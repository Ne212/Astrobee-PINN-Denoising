"""
=============================================================================
  train.py  —  Physics-Informed IMU Denoising Eğitim Scripti
=============================================================================
  Loss Bileşenleri:
    L1  Acceleration MSE    — a_pred ↔ true_accel (doğruluk)
    L2  Smoothness / TV     — Jerk + Jerk² minimizasyonu (pürüzsüzlük)
    L3  Kinematic           — İkinci sonlu fark pos_gt ↔ a_pred (tutarlılık)
    L4  Spectral            — FFT ile yüksek-frekans enerji cezası (frekans süzme)
    L5  Calibration Aux     — Fizik katmanı ara çıktısını da düzelt

  Optimizer : AdamW
  Scheduler : OneCycleLR (super-convergence: hızlı ısınma → cosine soğuma)
  AMP       : torch.cuda.amp  (GPU için %20-40 hız artışı)
  Warmup    : İlk N epoch'ta smooth/spectral ağırlıklarını kademeli artır
=============================================================================
"""

import os
import glob
import logging
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

from dataset_loader import build_dataloaders
from model import PhysicsIMUNet, count_parameters

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

def smooth_overlap_add(windows_array, seq_len=200, stride=50):
        """Kopuk pencereleri çan eğrisi ile üst üste bindirerek kesintisiz sinyal üretir."""
        N, _, C = windows_array.shape
        total_len = (N - 1) * stride + seq_len
        reconstructed = np.zeros((total_len, C))
        weight_sum = np.zeros((total_len, C))
        window = np.hanning(seq_len).reshape(-1, 1) 
        for i in range(N):
            start = i * stride
            end = start + seq_len
            reconstructed[start:end] += windows_array[i] * window
            weight_sum[start:end] += window
        reconstructed /= np.maximum(weight_sum, 1e-8)
        return reconstructed


# =============================================================================
#  LOSS FONKSİYONLARI
# =============================================================================

class PhysicsLoss(nn.Module):
    """
    Fizik-Bilgilendirilmiş IMU Denoising Kaybı.

    Neden 4 bileşen?
    ─────────────────
    Sadece MSE kullanan modeller "testere dişini koruyarak kaydırır":
    - L_smooth  → modeli ani değişimden DOĞRUDAN caydırır
    - L_spectral → testere dişinin FREKANS bileşenlerini hedef alır
    - L_kinematic → konum uyumluluğunu kinematik seviyede doğrular
    Hepsi birden modeli pürüzsüz, fiziksel ve doğru olmaya zorlar.

    Normalizasyon notu:
    ─────────────────────
    a_pred normalize uzayda (ya_std, ya_mean ile). Kinematik loss için
    pos_gt'nin ikinci farkı fiziksel birimde (m/s²) olduğundan ya_stats
    ile normalize edilir.
    """

    def __init__(
        self,
        w_accel:     float = 1.0,
        w_smooth:    float = 0.8,
        w_kinematic: float = 0.3,
        w_spectral:  float = 0.4,
        w_cal_aux:   float = 0.1,   # Fizik katmanı yardımcı kaybı
        dt:          float = 1.0 / 62.5,   # Astrobee IMU: 62.5 Hz
        cutoff_hz:   float = 3.0,          # Bu frekans üstü cezalandır
        ya_mean:     Optional[np.ndarray] = None,
        ya_std:      Optional[np.ndarray] = None,
    ):
        super().__init__()
        self.w_accel     = w_accel
        self.w_smooth    = w_smooth
        self.w_kinematic = w_kinematic
        self.w_spectral  = w_spectral
        self.w_cal_aux   = w_cal_aux
        self.dt          = dt
        self.cutoff_hz   = cutoff_hz

        # Kinematik loss için normalizasyon tamponu
        if ya_mean is not None and ya_std is not None:
            self.register_buffer(
                'ya_mean', torch.tensor(ya_mean, dtype=torch.float32)
            )
            self.register_buffer(
                'ya_std',  torch.tensor(ya_std,  dtype=torch.float32)
            )
        else:
            self.ya_mean = None
            self.ya_std  = None

    # ── L1: Acceleration MSE ─────────────────────────────────────────────────

    def _loss_accel(
        self, a_pred: torch.Tensor, a_true: torch.Tensor
    ) -> torch.Tensor:
        """
        Gelişmiş Tepe Odaklı MSE + Global Durağanlık Çıpası.
        """
        # Temel MSE (B, T, 3)
        base_loss = F.mse_loss(a_pred, a_true, reduction='none')
        
        # 1. Dinamik Tepe Ağırlığı (Tepeleri yakalamak için)
        # Katsayıyı 2.0'dan 3.0'a çekerek ivmeli hareketlere sadakati artırdık.
        weight_peak = 1.0 + 3.0 * torch.abs(a_true)
        
        # 2. Gelişmiş Durağanlık Çıpası (Stationary Anchor)
        # Üstel fonksiyonun içindeki katsayıyı (-20'den -15'e) biraz yumuşatarak 
        # sıfır civarındaki "güvenli bölgeyi" genişlettik. 
        # Çarpanı (5.0'dan 8.0'a) artırarak bias kaymalarına karşı toleransı sıfırladık.
        weight_zero = 8.0 * torch.exp(-15.0 * (a_true ** 2))
        
        # 3. Toplam Dinamik Ağırlık
        final_weight = weight_peak + weight_zero
        
        # Ağırlıklı Loss
        weighted_loss = base_loss * final_weight
        
        return weighted_loss.mean()

    # ── L2: Smoothness / Total Variation ─────────────────────────────────────

    def _loss_smoothness(
        self, a_pred: torch.Tensor, a_true: torch.Tensor
    ) -> torch.Tensor:
        """
        Kenara Duyarlı (Edge-Aware) Total Variation.
        Gerçek sinyalin (a_true) gerçekten aniden değiştiği yerlerde
        pürüzsüzlük cezasını dinamik olarak düşürür.
        """
        jerk_pred  = a_pred[:, 1:, :] - a_pred[:, :-1, :]
        jerk_true  = a_true[:, 1:, :] - a_true[:, :-1, :]
        
        jerk2_pred = jerk_pred[:, 1:, :] - jerk_pred[:, :-1, :]
        jerk2_true = jerk_true[:, 1:, :] - jerk_true[:, :-1, :]
        
        # Gerçek hareketin çok sert olduğu anlarda (jerk_true yüksekse),
        # weight sıfıra yaklaşır ve modele o anlık esneklik tanır.
        weight1 = torch.exp(-3.0 * torch.abs(jerk_true))
        weight2 = torch.exp(-3.0 * torch.abs(jerk2_true))
        
        loss_j1 = (weight1 * (jerk_pred ** 2)).mean()
        loss_j2 = (weight2 * (jerk2_pred ** 2)).mean()
        
        return loss_j1 + 0.5 * loss_j2

    # ── L3: Kinematic Consistency ─────────────────────────────────────────────

    def _loss_kinematic(self, a_pred: torch.Tensor, pos_gt: torch.Tensor) -> torch.Tensor:
        dt2 = self.dt ** 2
        # 1. Ham ivmeyi hesapla (m/s^2)
        # Boyut: (B, T-2, 3)
        a_from_pos = (pos_gt[:, 2:, :] - 2.0 * pos_gt[:, 1:-1, :] + pos_gt[:, :-2, :]) / dt2

        # 2. ÖNEMLİ: Normalizasyon kontrolü
        # ya_mean ve ya_std'nin varlığından ve cihazından emin olalım
        if self.ya_mean is not None and self.ya_std is not None:
            # Burası hayati: a_from_pos fiziksel birim (1-2 m/s^2), 
            # tahminler ise normalize (-5 ile +5 arası). 
            # İkisini de aynı teraziye alıyoruz.
            a_from_pos = (a_from_pos - self.ya_mean) / (self.ya_std + 1e-6)
        
        # 3. Kırpma (Alignment): a_pred (T), a_from_pos (T-2)
        # Ortadaki timestepleri karşılaştırıyoruz.
        target = a_pred[:, 1:-1, :]
        
        # HATA KONTROLÜ: Eğer hala 1800 geliyorsa, a_from_pos içinde NaN veya inf olabilir
        # Bunu önlemek için clamp ekleyelim
        a_from_pos = torch.clamp(a_from_pos, min=-50, max=50)

        return F.mse_loss(target, a_from_pos)

    # ── L4: Spectral Smoothness ───────────────────────────────────────────────

    def _loss_spectral(self, a_pred: torch.Tensor) -> torch.Tensor:
        """
        Frekans alanı yüksek-frekans enerji cezası.

        Adımlar:
          1. rfft ile a_pred'i frekans alanına taşı
          2. Her frekans için güç hesapla: |FFT|²
          3. cutoff_hz üzerindeki frekanslara sigmoid maske uygula
          4. Maskeli gücün ortalaması = loss

        Neden sigmoid maske?
          Ani kesme yerine yumuşak geçiş → gradient daha düzgün akar.
          Şarpness=10 → 0.1 Hz genişliğinde geçiş bandı.

        Bu loss DOĞRUDAN "testere dişi" frekans bileşenlerini hedef alır.
        Modelin çıktısı spectral açıdan low-pass filtrelenmiş olmak zorundadır.
        """
        B, T, C = a_pred.shape
        a_pred_f32 = a_pred.float() 
        
        # FFT zaman ekseni boyunca: (B, T//2+1, C)
        fft_out = torch.fft.rfft(a_pred_f32, dim=1)
        power   = torch.abs(fft_out) ** 2                     # (B, T//2+1, C)

        # Frekans ekseni (Hz cinsinden)
        freqs = torch.fft.rfftfreq(T, d=self.dt).to(a_pred.device)  # (T//2+1,)

        # Yumuşak yüksek-frekans maskesi
        # f < cutoff → 0 (ceza yok), f > cutoff → 1 (tam ceza)
        sharpness = 10.0
        freq_mask = torch.sigmoid(sharpness * (freqs - self.cutoff_hz))  # (T//2+1,)

        # Yüksek-frekans güç ortalaması
        loss = (power * freq_mask.unsqueeze(0).unsqueeze(-1)).mean()
        return loss

    # ── Ana Loss ──────────────────────────────────────────────────────────────

    def forward(
        self,
        a_pred: torch.Tensor,           # (B, T, 3) model çıktısı
        a_true: torch.Tensor,           # (B, T, 3) normalize smooth ivme
        pos_gt: torch.Tensor,           # (B, T, 3) GT konum (m)
        a_cal:  Optional[torch.Tensor] = None,   # (B, T, 3) fizik ara çıktı
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        
        
        
        # 1. Temel Kayıplar
        # Not: _loss_accel içine "Durağanlık Çıpası" eklediğimizi varsayıyoruz
        L_accel     = self._loss_accel(a_pred, a_true)
        L_smooth    = self._loss_smoothness(a_pred, a_true)
        L_kinematic = self._loss_kinematic(a_pred, pos_gt)
        L_spectral  = self._loss_spectral(a_pred)

        # 2. YENİ: Genel Bias (DC Offset) Kaybı
        # Tahmin ve gerçeğin zamansal ortalamaları arasındaki fark.
        # Bu, sinyalin genel olarak yukarı/aşağı kaymasını (bias) cezalandırır.
        L_bias = F.mse_loss(a_pred.mean(dim=1), a_true.mean(dim=1))

        # 3. Kalibrasyon katmanı yardımcı kaybı
        if a_cal is not None:
            L_cal = F.mse_loss(a_cal, a_true) * self.w_cal_aux
        else:
            L_cal = torch.tensor(0.0, device=a_pred.device)

        # 4. Toplam Kayıp (Yeni Katsayılarla)
        # self.w_bias değerini dışarıdan 4.0 veya 5.0 gibi tanımlayabilirsin.
        w_bias = 10.0 

        total = (
            self.w_accel     * L_accel     +
            self.w_smooth    * L_smooth    +
            self.w_kinematic * L_kinematic +
            self.w_spectral  * L_spectral  +
            w_bias           * L_bias      +  # <--- Genel bias düzeltici
            L_cal
        )

        return total, {
            'total'    : total.item(),
            'accel'    : L_accel.item(),
            'smooth'   : L_smooth.item(),
            'kinematic': L_kinematic.item(),
            'spectral' : L_spectral.item(),
            'bias'     : L_bias.item(),       # Sözlüğe ekledik
            'cal_aux'  : L_cal.item(),
        }
    



# =============================================================================
#  METRİK
# =============================================================================

class DenoiseMetrics:
    """
    Denoising kalite metrikleri.

    RMSE    : Ortalama tahmin hatası (küçük = iyi)
    SNR_dB  : Sinyal-gürültü oranı kazancı (büyük = iyi)
    Smooth  : Jerk std oranı pred/true (< 1.0 = pred daha pürüzsüz)
    Improve : Ham IMU'ya göre iyileşme yüzdesi
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self._preds = []
        self._trues = []
        self._raws  = []

    def update(
        self,
        pred: torch.Tensor,
        true: torch.Tensor,
        raw:  Optional[torch.Tensor] = None,
    ):
        self._preds.append(pred.detach().cpu().float().numpy())
        self._trues.append(true.detach().cpu().float().numpy())
        if raw is not None:
            self._raws.append(raw.detach().cpu().float().numpy())

    def compute(self) -> Dict[str, float]:
        preds = np.concatenate(self._preds).reshape(-1, 3)
        trues = np.concatenate(self._trues).reshape(-1, 3)
        diff  = preds - trues

        # RMSE per axis + mean
        rmse_ax = np.sqrt((diff ** 2).mean(0))                # (3,)
        rmse    = float(rmse_ax.mean())

        # SNR: signal power / noise power
        sig_pwr  = float((trues ** 2).mean())
        noise_pwr = float((diff ** 2).mean())
        snr_db   = float(10 * np.log10(sig_pwr / (noise_pwr + 1e-12)))

        # Smoothness index: pred jerk std / true jerk std
        jerk_pred = np.diff(preds, axis=0)
        jerk_true = np.diff(trues, axis=0)
        smooth_idx = float(np.std(jerk_pred) / (np.std(jerk_true) + 1e-12))

        result: Dict[str, float] = {
            'RMSE'   : rmse,
            'RMSE_x' : float(rmse_ax[0]),
            'RMSE_y' : float(rmse_ax[1]),
            'RMSE_z' : float(rmse_ax[2]),
            'SNR_dB' : snr_db,
            'Smooth' : smooth_idx,  # hedef: < 1.0
        }

        if self._raws:
            raws     = np.concatenate(self._raws).reshape(-1, 3)
            rmse_raw = float(np.sqrt(((raws - trues) ** 2).mean()))
            result['RMSE_raw']  = rmse_raw
            result['Improve_%'] = float((1 - rmse / (rmse_raw + 1e-12)) * 100)

        return result


# =============================================================================
#  TRAINER
# =============================================================================

class Trainer:
    """
    Eğitim / değerlendirme döngüsü yöneticisi.

    Özellikler
    ──────────
    - Loss ağırlığı ısınması: Smooth & Spectral yavaşça devreye girer
      (Başlangıçta sadece doğruluk, sonra pürüzsüzlük cezası eklenir.
       Aksi halde model önce "ne yapacağını" öğrenemeden cezalanır.)
    - AMP (Mixed Precision): fp16 + fp32 karışık → daha hızlı GPU
    - Gradient clipping: patlayan gradient koruması
    - OneCycleLR: agresif ısınma → kararlı soğuma = super-convergence
    - Early stopping + en iyi checkpoint kaydı
    """

    def __init__(
        self,
        model:          PhysicsIMUNet,
        criterion:      PhysicsLoss,
        optimizer:      torch.optim.Optimizer,
        scheduler:      torch.optim.lr_scheduler._LRScheduler,
        device:         torch.device,
        ckpt_dir:       str   = "checkpoints",
        patience:       int   = 40,
        max_grad_norm:  float = 1.0,
        use_amp:        bool  = True,
        warmup_epochs:  int   = 25,
    ):
        self.model         = model.to(device)
        self.criterion     = criterion
        self.optimizer     = optimizer
        self.scheduler     = scheduler
        self.device        = device
        self.patience      = patience
        self.max_grad_norm = max_grad_norm
        self.warmup_epochs = warmup_epochs

        # AMP sadece CUDA'da anlamlı
        self.use_amp = use_amp and (device.type == 'cuda')
        self.scaler  = torch.amp.GradScaler('cuda') if self.use_amp else None

        os.makedirs(ckpt_dir, exist_ok=True)
        self.ckpt_path = os.path.join(ckpt_dir, "best_pidn.pt")

        self.best_val_loss = float('inf')
        self.no_improve    = 0
        self.history       = {'train': [], 'val': []}

        # Orijinal ağırlıkları sakla (ısınma için)
        self._w_smooth   = criterion.w_smooth
        self._w_spectral = criterion.w_spectral

    # ─── Loss Ağırlığı Isınması ──────────────────────────────────────────────

    def _set_warmup_weights(self, epoch: int) -> None:
        """
        İlk warmup_epochs'ta smooth ve spectral ağırlıkları 0'dan hedef değere
        kademeli olarak artır. Bu sayede model önce "doğru yerde" öğrenir,
        sonra "düzgün" öğrenir.
        """
        if epoch <= self.warmup_epochs:
            factor = epoch / self.warmup_epochs
        else:
            factor = 1.0
        self.criterion.w_smooth   = self._w_smooth   * factor
        self.criterion.w_spectral = self._w_spectral * factor

    # ─── Tek Epoch ───────────────────────────────────────────────────────────

    def _run_epoch(
        self,
        loader:   DataLoader,
        training: bool,
        metrics:  DenoiseMetrics,
    ) -> Dict[str, float]:

        self.model.train(training)
        comp_sum: Dict[str, float] = {}
        n_batches = 0

        ctx = torch.enable_grad() if training else torch.no_grad()

        with ctx:
            for X_b, ya_b, yp_b in loader:
                X_b  = X_b.to(self.device)    # (B, T, 6)
                ya_b = ya_b.to(self.device)   # (B, T, 3)
                yp_b = yp_b.to(self.device)   # (B, T, 3)

                # Forward
                if self.use_amp:
                    with torch.amp.autocast('cuda'):
                        a_pred, a_cal = self.model(X_b)
                        loss, comps   = self.criterion(a_pred, ya_b, yp_b, a_cal)
                else:
                    a_pred, a_cal = self.model(X_b)
                    loss, comps   = self.criterion(a_pred, ya_b, yp_b, a_cal)

                # Backward
                if training:
                    self.optimizer.zero_grad(set_to_none=True)
                    if self.use_amp:
                        self.scaler.scale(loss).backward()
                        self.scaler.unscale_(self.optimizer)
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        loss.backward()
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                        self.optimizer.step()
                    
                    # --- KRİTİK DÜZELTME ---
                    # Scheduler her BATCH sonunda adım atmalıdır.
                    # Bu satırın 'if training' bloğunun içinde ve döngünün en sonunda olduğundan emin ol.
                    self.scheduler.step()

                # Birikim
                for k, v in comps.items():
                    comp_sum[k] = comp_sum.get(k, 0.0) + v
                n_batches += 1

                # Metrik: ham ivme = X_b'nin ilk 3 kanalı (denormalize ETME —
                # metrik normalize uzayda tutarlı karşılaştırma yapar)
                metrics.update(a_pred, ya_b, X_b[:, :, :3])

        return {k: v / max(n_batches, 1) for k, v in comp_sum.items()}

    # ─── Ana Eğitim Döngüsü ──────────────────────────────────────────────────

    def fit(
        self,
        train_loader: DataLoader,
        val_loader:   DataLoader,
        epochs:       int = 150,
    ) -> Dict:
        log.info("=" * 70)
        log.info("Eğitim başlıyor  [Physics-Informed IMU Denoising Network]")
        log.info(f"  Epochs       : {epochs}")
        log.info(f"  Early stop   : {self.patience}")
        log.info(f"  AMP          : {self.use_amp}")
        log.info(f"  Warmup epoch : {self.warmup_epochs}")
        log.info(f"  Device       : {self.device}")
        log.info("=" * 70)

        for epoch in range(1, epochs + 1):
            # Smooth & Spectral ağırlığını warmup takvimle güncelle
            self._set_warmup_weights(epoch)

            train_m = DenoiseMetrics()
            val_m   = DenoiseMetrics()

            train_c = self._run_epoch(train_loader, training=True,  metrics=train_m)
            val_c   = self._run_epoch(val_loader,   training=False, metrics=val_m)

            train_met = train_m.compute()
            val_met   = val_m.compute()

            self.history['train'].append({**train_c, **train_met})
            self.history['val'].append({**val_c, **val_met})

            lr   = self.optimizer.param_groups[0]['lr']
            w_sm = self.criterion.w_smooth
            w_sp = self.criterion.w_spectral

            log.info(
                f"Epoch {epoch:>4d}/{epochs} | "
                f"Tr {train_c['total']:.4f} "
                f"(acc={train_c['accel']:.4f} sm={train_c['smooth']:.4f} "
                f"sp={train_c['spectral']:.4f} kin={train_c['kinematic']:.4f}) | "
                f"Val {val_c['total']:.4f} | "
                f"RMSE={val_met['RMSE']:.5f} "
                f"SNR={val_met['SNR_dB']:.1f}dB "
                f"Smo={val_met['Smooth']:.3f} "
                f"Imp={val_met.get('Improve_%', 0):.1f}% | "
                f"LR={lr:.2e} w_sm={w_sm:.2f} w_sp={w_sp:.2f}"
            )

            # ── Checkpoint ────────────────────────────────────────────────
            if val_c['total'] < self.best_val_loss:
                self.best_val_loss = val_c['total']
                self.no_improve    = 0
                torch.save({
                    'epoch'      : int(epoch),
                    'model_state': self.model.state_dict(),
                    'optim_state': self.optimizer.state_dict(),
                    'val_loss'   : float(self.best_val_loss),
                    'val_metrics': {k: float(v) for k, v in val_met.items()},
                    'cal_matrix' : self.model.sensor_cal.calibration_matrix.tolist(),
                    'cal_bias'   : self.model.sensor_cal.bias.tolist(),
                }, self.ckpt_path)
                log.info(f"  ✓ Checkpoint → {self.ckpt_path}")
            else:
                self.no_improve += 1

            # ── Early Stopping ─────────────────────────────────────────────
            if self.no_improve >= self.patience:
                log.info(
                    f"Early stopping! "
                    f"({self.patience} epoch boyunca iyileşme yok)"
                )
                break

        log.info("Eğitim tamamlandı.")
        log.info(f"En iyi Val Loss : {self.best_val_loss:.6f}")

        # Öğrenilen kalibrasyon değerlerini raporla
        cal_m = self.model.sensor_cal.calibration_matrix.detach().cpu().numpy()
        cal_b = self.model.sensor_cal.bias.detach().cpu().numpy()
        log.info("Öğrenilen Kalibrasyon Matrisi (S):")
        log.info(f"  {cal_m.round(5)}")
        log.info(f"Öğrenilen Bias (b): {cal_b.round(6)}")

        return self.history

    # ─── Değerlendirme ───────────────────────────────────────────────────────

    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        """Test seti üzerinde metrik hesapla."""
        m = DenoiseMetrics()
        self._run_epoch(loader, training=False, metrics=m)
        return m.compute()

    def predict(
        self, loader: DataLoader, denormalize: bool = False, stats: Optional[Dict] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Tüm örnekler için temizlenmiş ivme tahmin et.

        Parametreler
        ────────────
        denormalize : True ise ya_stats kullanarak fiziksel birime çevir
        stats       : dataset stats sözlüğü (ya_mean, ya_std)

        Dönüş: (a_pred_array, a_true_array)  her biri (N, T, 3)
        """
        self.model.eval()
        preds_list, trues_list = [], []

        with torch.no_grad():
            for X_b, ya_b, _ in loader:
                X_b = X_b.to(self.device)
                a_pred, _ = self.model(X_b)
                preds_list.append(a_pred.cpu().numpy())
                trues_list.append(ya_b.numpy())

        preds = np.concatenate(preds_list, axis=0)   # (N, T, 3)
        trues = np.concatenate(trues_list, axis=0)   # (N, T, 3)

        if denormalize and stats is not None:
            preds = preds * stats['ya_std'] + stats['ya_mean']
            trues = trues * stats['ya_std'] + stats['ya_mean']

        return preds, trues


# =============================================================================
#  ANA FONKSİYON
# =============================================================================

def main():
    # ─────────────────────────────────────────────────────────────────────────
    # YAPILANDIRMA  —  kendi verilerinize göre güncelleyin
    # ─────────────────────────────────────────────────────────────────────────
    CSV_PATHS    = sorted(glob.glob("*.csv"))

    # Veri
    SEQ_LEN      = 200
    STRIDE       = 30        # 50 → %75 örtüşme; daha az veri için 100 yapın
    BATCH_SIZE   = 64

    # Eğitim
    EPOCHS       = 400
    LR           = 1e-3    # OneCycleLR için max_lr
    WEIGHT_DECAY = 5e-4
    PATIENCE     = 40
    WARMUP_EPOCHS = 5       # Smooth/Spectral loss kademeli devreye giriş

    # Model
    CNN_HIDDEN   = 128
    LSTM_HIDDEN  = 128
    LSTM_LAYERS  = 2
    DROPOUT      = 0.4

    # Loss ağırlıkları
    W_ACCEL      = 1000
    W_SMOOTH     = 0.1       # Yüksek tut — pürüzsüzlük öncelikli
    W_KINEMATIC  = 0.0
    W_SPECTRAL   = 25.0      # Testere dişi frekanslarını doğrudan hedef al
    W_CAL_AUX   = 1.0

    # Fizik
    SAMPLE_RATE  = 62.5      # Hz  (Astrobee IMU)
    DT           = 1.0 / SAMPLE_RATE
    CUTOFF_HZ    = 6.0       # Bu frekansın üstü gürültü olarak kabul edilir
    # ─────────────────────────────────────────────────────────────────────────

    if not CSV_PATHS:
        log.error("CSV bulunamadı! 'data/' klasörüne CSV dosyalarınızı koyun.")
        log.error("CSV kolonları: ax,ay,az,gx,gy,gz,px,py,pz,true_ax,true_ay,true_az")
        return

    os.makedirs("checkpoints", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Cihaz: {device}")

    # ── Veri ──────────────────────────────────────────────────────────────────
    train_dl, val_dl, test_dl, stats = build_dataloaders(
        csv_paths   = CSV_PATHS,
        seq_len     = SEQ_LEN,
        stride      = STRIDE,
        batch_size  = BATCH_SIZE,
        stats_path  = "checkpoints/stats.pkl",
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = PhysicsIMUNet(
        seq_len     = SEQ_LEN,
        cnn_hidden  = CNN_HIDDEN,
        lstm_hidden = LSTM_HIDDEN,
        lstm_layers = LSTM_LAYERS,
        dropout     = DROPOUT,
    )
    log.info(f"Model parametresi: {count_parameters(model):,}")

    # ── Loss ──────────────────────────────────────────────────────────────────
    criterion = PhysicsLoss(
        w_accel     = W_ACCEL,
        w_smooth    = W_SMOOTH,
        w_kinematic = W_KINEMATIC,
        w_spectral  = W_SPECTRAL,
        w_cal_aux   = W_CAL_AUX,
        dt          = DT,
        cutoff_hz   = CUTOFF_HZ,
        ya_mean     = stats['ya_mean'],
        ya_std      = stats['ya_std'],
    )

    # ── Optimizer + Scheduler ─────────────────────────────────────────────────
    optimizer = AdamW(
        model.parameters(),
        lr           = LR,
        weight_decay = WEIGHT_DECAY,
        betas        = (0.9, 0.999),
    )

    # OneCycleLR: Super-convergence
    #   - Başlangıç LR  = max_lr / div_factor        = 1e-3 / 25 = 4e-5
    #   - Tepe LR       = max_lr                     = 1e-3  (epoch × pct_start'da)
    #   - Bitiş LR      = başlangıç / final_div      = 4e-5 / 1000 = 4e-8
    scheduler = OneCycleLR(
        optimizer,
        max_lr          = 2e-3,
        epochs          = EPOCHS,
        steps_per_epoch = len(train_dl),
        pct_start       = 0.3,     # %30 ısınma
        anneal_strategy = 'cos',
        div_factor      = 25.0,
        final_div_factor = 1000.0,
    )

    # ── Eğitim ────────────────────────────────────────────────────────────────
    trainer = Trainer(
        model         = model,
        criterion     = criterion,
        optimizer     = optimizer,
        scheduler     = scheduler,
        device        = device,
        ckpt_dir      = "checkpoints",
        patience      = PATIENCE,
        use_amp       = True,
        warmup_epochs = WARMUP_EPOCHS,
    )
    trainer.criterion.to(device)

    history = trainer.fit(train_dl, val_dl, epochs=EPOCHS)

    # ── Test Değerlendirme ────────────────────────────────────────────────────
    log.info("En iyi checkpoint yükleniyor ...")
    ckpt = torch.load(
        "checkpoints/best_pidn.pt",
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(ckpt['model_state'])

    test_metrics = trainer.evaluate(test_dl)
    log.info("=" * 70)
    log.info("TEST SONUÇLARI:")
    log.info(f"  RMSE       : {test_metrics['RMSE']:.5f}")
    log.info(f"  SNR        : {test_metrics['SNR_dB']:.2f} dB")
    log.info(f"  Smoothness : {test_metrics['Smooth']:.4f}  (< 1.0 = pred daha pürüzsüz)")
    if 'Improve_%' in test_metrics:
        log.info(f"  Ham IMU'ya göre iyileşme: %{test_metrics['Improve_%']:.1f}")
    log.info("=" * 70)

    # ── Tahminleri Kaydet ─────────────────────────────────────────────────────
    preds, trues = trainer.predict(test_dl, denormalize=True, stats=stats)
    
    # 2. HAM (Kirli) Veriyi DataLoader'dan Çek
    all_inputs = []
    for batch in test_dl:
        x = batch[0] # Girdi tensörü (Genelde: ax, ay, az, gx, gy, gz)
        all_inputs.append(x.cpu().numpy())
    raw_inputs = np.concatenate(all_inputs, axis=0)

    # İvme verilerini (ilk 3 sütun) al ve Denormalize et
    raw_accel = raw_inputs[:, :, :3]
    
    # Not: stats sözlüğündeki anahtar isimleri X_mean veya mean olabilir, güvenli çekiyoruz:
    mean_x = np.array(stats.get('X_mean', stats.get('mean', np.zeros(6))))[:3]
    std_x  = np.array(stats.get('X_std',  stats.get('std',  np.ones(6))))[:3]
    raw_accel = (raw_accel * std_x) + mean_x

    # 3. Pencereleri Eritip Kesintisiz (Continuous) Sinyaller Oluştur
    preds_continuous = smooth_overlap_add(preds, seq_len=200, stride=50)
    trues_continuous = smooth_overlap_add(trues, seq_len=200, stride=50)
    raw_continuous   = smooth_overlap_add(raw_accel, seq_len=200, stride=50)

    # 4. YENİ: Tüm Kesintisiz Verileri Kaydet
    np.save("checkpoints/test_predictions_continuous.npy", preds_continuous)
    np.save("checkpoints/test_groundtruth_continuous.npy", trues_continuous)
    np.save("checkpoints/test_raw_continuous.npy", raw_continuous) # KİRLİ VERİ KAYDEDİLDİ!
    
    # Eski kopuk versiyonları da (ne olur ne olmaz diye) kaydedebilirsin:
    np.save("checkpoints/test_predictions.npy", preds)
    np.save("checkpoints/test_groundtruth.npy", trues)

    log.info(f"Orijinal kopuk veriler kaydedildi: {preds.shape}")
    log.info(f"Kesintisiz tahminler kaydedildi: checkpoints/test_predictions_continuous.npy  {preds_continuous.shape}")
    log.info(f"Kesintisiz HAM veri kaydedildi: checkpoints/test_raw_continuous.npy  {raw_continuous.shape}")

    return history, test_metrics


if __name__ == "__main__":
    main()
