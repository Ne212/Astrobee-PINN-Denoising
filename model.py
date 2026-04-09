"""
=============================================================================
  model.py  —  Physics-Informed IMU Denoising Network (PIDN)
=============================================================================

Mimari özeti:
  Ham IMU (B,T,6)
        ↓
  SensorCalibrationLayer  ← Fizik Katmanı: bias + ölçek + çapraz eksen
        ↓ a_cal (B,T,3)
  [X || a_cal] (B,T,9) ──→ MultiScaleDilatedEncoder ──→ (B,T,128)
                                     ↓
                            Bidirectional LSTM ──→ (B,T,256)
                                     ↓
                            [ctx || a_cal] (B,T,259)
                                     ↓
                             MLP Correction Head ──→ (B,T,3)
                                     ↓
                       a_pred = a_cal + correction  ← Residual

Residual tasarımı: Fizik katmanı temel tahmini (a_cal) verir;
sinir ağı sadece düzeltme öğrenir. Bu:
  - Eğitimi hızlandırır (sıfırdan öğrenmek yerine fizikten başlar)
  - Fiziksel tutarlılığı garantiler (büyük ayrışma cezalandırılır)
  - Gradyan akışını iyileştirir (doğrudan yol: a_cal → output)
=============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


# =============================================================================
#  1. Fizik Katmanı: Sensör Kalibrasyon Modeli
# =============================================================================

class SensorCalibrationLayer(nn.Module):
    """
    Doğrusal sensör fizik modeli:  a_cal = L @ (a_raw − b)

    Parametreler
    ────────────
    b : (3,) bias vektörü  — başlangıç: sıfır
    L : (3,3) alt üçgen matris  — başlangıç: identity

      [ exp(d0)    0        0    ]
    L = [ l_10    exp(d1)   0    ]
      [ l_20    l_21    exp(d2) ]

    Neden alt üçgen?
      - exp(diagonal) → ölçek faktörleri her zaman pozitif
      - alt elemanlar → çapraz eksen kuplaj (cross-axis coupling)
      - Identity başlangıcı → eğitim başında değişiklik yok (kararlı)

    Fiziksel karşılık:
      Bir MEMS ivme ölçerin gerçek modeli:
        a_raw = R @ a_true + b + gürültü
      Bu katman R_inv ve b'yi öğrenir.
    """

    def __init__(self):
        super().__init__()
        # log(diagonal) — exp ile pozitiflik garantisi; başlangıç 0 → exp(0)=1
        self.log_diag = nn.Parameter(torch.zeros(3))
        # Alt üçgen dışı köşegen elemanları (çapraz eksen kuplaj)
        self.off_diag = nn.Parameter(torch.zeros(3))
        # Bias: ivme ölçer sıfır harekette sıfır okumuyorsa burası öğrenir
        self.bias = nn.Parameter(torch.zeros(3))

    @property
    def calibration_matrix(self) -> torch.Tensor:
        """3×3 alt üçgen kalibrasyon matrisi."""
        diag = torch.exp(self.log_diag)   # (3,) — her zaman pozitif
        L = torch.zeros(3, 3, device=self.bias.device, dtype=self.bias.dtype)
        L[0, 0] = diag[0]
        L[1, 1] = diag[1];  L[1, 0] = self.off_diag[0]
        L[2, 2] = diag[2];  L[2, 0] = self.off_diag[1]; L[2, 1] = self.off_diag[2]
        return L  # (3, 3)

    def forward(self, a_raw: torch.Tensor) -> torch.Tensor:
        """
        a_raw : (B, T, 3)
        return: (B, T, 3)  kalibre edilmiş ivme
        """
        a_centered = a_raw - self.bias                     # broadcast: (B,T,3) - (3,)
        # matmul: (..., 3) @ (3, 3).T  → (..., 3)
        return torch.matmul(a_centered, self.calibration_matrix.T)

    def extra_repr(self) -> str:
        diag = torch.exp(self.log_diag).detach().cpu().numpy().round(4).tolist()
        bias = self.bias.detach().cpu().numpy().round(5).tolist()
        return f"scale_diag={diag}, bias={bias}"


# =============================================================================
#  2. CNN Encoder: Dilated Residual Konvolüsyon
# =============================================================================

class _DilatedResBlock(nn.Module):
    """
    Tek bir dilated 1D konvolüsyon residual bloğu.
    Sequence uzunluğunu DEĞIŞTIRMEZ.
    """

    def __init__(self, channels: int, kernel_size: int = 5, dilation: int = 1):
        super().__init__()
        # Asimetrik padding: sadece geçmiş timestep'leri görür (causal)
        # Not: causal padding yerine simetrik padding kullanıyoruz çünkü
        # eğitim sırasında geleceğe bakmak overfitting değil — bu bir
        # offline denoising problemi, gerçek zamanlı tahmin değil.
        pad = (kernel_size - 1) * dilation // 2
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size,
                      padding=pad, dilation=dilation, bias=False),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Conv1d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(channels),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        return self.act(x + self.block(x))


class MultiScaleDilatedEncoder(nn.Module):
    """
    Artan dilation ile genişleyen alıcı alan encoder.

    Dilation = [1, 2, 4, 8, 16]:
      Her katman bir öncekinin 2 katı uzun geçmişe bakar.
      T=200, kernel=5 ile toplam alıcı alan ≈ 80 timestep.
      62.5 Hz'de bu ≈ 1.28 saniye — yeterince uzun hareketi yakalar.

    Neden strided conv (downsample) YOK?
      Sequence uzunluğu korunursa decoder gerekmez → daha basit,
      daha kararlı, her timestep bağımsız çıktı üretir.
    """

    def __init__(self, in_ch: int = 9, hidden: int = 128):
        super().__init__()
        # Giriş projeksiyon: (B, in_ch, T) → (B, hidden, T)
        self.input_proj = nn.Sequential(
            nn.Conv1d(in_ch, hidden, kernel_size=1, bias=False),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
        )
        # Dilated residual katmanlar: alıcı alanı genişletir
        self.blocks = nn.ModuleList([
            _DilatedResBlock(hidden, kernel_size=5, dilation=d)
            for d in [1, 2, 4, 8, 16]
        ])
        # Son projeksiyon: channel-wise mixing
        self.output_proj = nn.Sequential(
            nn.Conv1d(hidden, hidden, kernel_size=1, bias=False),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x:   (B, T, in_ch)
        out: (B, T, hidden)
        """
        out = self.input_proj(x.transpose(1, 2))  # (B, hidden, T)
        for block in self.blocks:
            out = block(out)
        out = self.output_proj(out)
        return out.transpose(1, 2)                 # (B, T, hidden)


# =============================================================================
#  3. Ana Model: PhysicsIMUNet
# =============================================================================

class PhysicsIMUNet(nn.Module):
    """
    Physics-Informed IMU Denoising Network (PIDN).

    Neden bu mimari testere dişini çözer?
    ──────────────────────────────────────
    1. Dilated CNN: Lokal gürültüyü yüksek alıcı alan sayesinde bağlam
       içinde değerlendirir; ani spike'ları "gerçek hareket" sanmaz.

    2. Bidirectional LSTM: Geleceği de görerek hareketin süreklilik
       tutarlılığını kontrol eder. "Bu t anındaki titreşim devam ediyor mu?"

    3. Fizik Residual: a_pred = a_cal + δ
       Ağ sıfırdan ivme öğrenmek yerine fiziksel kalibrasyonu düzeltir.
       Büyük δ = şüpheli → Loss fonksiyonu bunu cezalandırır.

    4. Spectral + Smoothness Loss (train.py'de): Modeli, çıktısının
       frekans içeriğini doğrudan kısıtlamaya zorlar.
    """

    def __init__(
        self,
        seq_len:     int   = 200,
        cnn_hidden:  int   = 128,
        lstm_hidden: int   = 128,
        lstm_layers: int   = 2,
        dropout:     float = 0.2,
    ):
        super().__init__()
        self.seq_len = seq_len

        # ── 1. Sensör Kalibrasyon Katmanı (Fizik) ────────────────────────
        self.sensor_cal = SensorCalibrationLayer()

        # ── 2. Multi-Scale CNN Encoder ────────────────────────────────────
        # Giriş: 6 (ham IMU) + 3 (kalibre ivme) = 9 kanal
        # Kalibre ivmeyi ek kanal olarak vermek:
        # - Fizik katmanının çıktısını CNN'e de tanıtır
        # - Skip-connection benzeri bilgi akışı sağlar
        self.encoder = MultiScaleDilatedEncoder(in_ch=9, hidden=cnn_hidden)

        # ── 3. Bidirectional LSTM ─────────────────────────────────────────
        self.lstm = nn.LSTM(
            input_size   = cnn_hidden,
            hidden_size  = lstm_hidden,
            num_layers   = lstm_layers,
            batch_first  = True,
            bidirectional = True,
            dropout = dropout if lstm_layers > 1 else 0.0,
        )
        lstm_out_dim = lstm_hidden * 2     # BiLSTM → ×2

        # ── 4. MLP Correction Head ────────────────────────────────────────
        # Giriş: LSTM çıktısı + a_cal (residual skip)
        # Çıktı: 3 eksen düzeltme vektörü
        head_in = lstm_out_dim + 3
        self.head = nn.Sequential(
            nn.Linear(head_in, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 3),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """
        Ağırlık başlatma stratejisi:
        - LSTM: Orthogonal init → uzun bağımlılıklarda gradient kaybolmasını önler
        - MLP son katman: çok küçük uniform → başlangıçta sıfıra yakın düzeltme
          (a_pred ≈ a_cal başlangıçta, ağ yavaş yavaş düzeltmeyi öğrenir)
        """
        for name, param in self.lstm.named_parameters():
            if 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
        # Son katman küçük başlatma: başlangıçta model "saf fizik" gibi davranır
        nn.init.uniform_(self.head[-1].weight, -1e-4, 1e-4)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parametreler
        ────────────
        x : (B, T, 6)  normalize edilmiş ham IMU [ax,ay,az,gx,gy,gz]

        Dönüş
        ─────
        a_pred : (B, T, 3)  temizlenmiş ivme (normalize uzayda)
        a_cal  : (B, T, 3)  fizik katmanı çıktısı (yardımcı loss için)
        """
        a_raw = x[:, :, :3]                      # ham ivme kanalları

        # ── Fizik Kalibrasyonu ────────────────────────────────────────────
        a_cal = self.sensor_cal(a_raw)            # (B, T, 3)

        # ── CNN Feature Extraction ────────────────────────────────────────
        enc_in  = torch.cat([x, a_cal], dim=-1)  # (B, T, 9)
        features = self.encoder(enc_in)           # (B, T, 128)

        # ── Temporal Context ──────────────────────────────────────────────
        ctx, _ = self.lstm(features)              # (B, T, 256)

        # ── Fizik + Sinir Ağı Residual ────────────────────────────────────
        combined   = torch.cat([ctx, a_cal], dim=-1)   # (B, T, 259)
        correction = self.head(combined)                # (B, T, 3)

        # Residual: ağ sadece düzeltmeyi öğrenir, fizik temel tahmini verir
        a_pred = a_cal + correction                     # (B, T, 3)

        return a_pred, a_cal

    def denoise(self, x: torch.Tensor) -> torch.Tensor:
        """Çıkarım (inference) için kısayol — sadece temizlenmiş ivme döner."""
        self.eval()
        with torch.no_grad():
            a_pred, _ = self.forward(x)
        return a_pred


# =============================================================================
#  Model Özeti
# =============================================================================

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    model = PhysicsIMUNet(seq_len=200)
    n = count_parameters(model)
    print(f"Toplam eğitilebilir parametre: {n:,}")

    # Sahte batch ile ileri geçiş testi
    dummy = torch.randn(4, 200, 6)
    a_pred, a_cal = model(dummy)
    print(f"Giriş şekli  : {dummy.shape}")
    print(f"a_pred şekli : {a_pred.shape}")
    print(f"a_cal şekli  : {a_cal.shape}")

    # Kalibrasyon matrisi
    print(f"\nSensör kalibrasyon matrisi (başlangıç = identity):")
    print(model.sensor_cal.calibration_matrix.data)
