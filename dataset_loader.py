"""
=============================================================================
  dataset_loader.py  —  Astrobee IMU Denoising Dataset
=============================================================================
  Giriş X : (B, 200, 6)   ham [ax, ay, az, gx, gy, gz]
  Hedef ya : (B, 200, 3)   smooth [true_ax, true_ay, true_az]
  Hedef yp : (B, 200, 3)   konum  [px, py, pz]  — fiziksel birim (m), normalize edilmez
=============================================================================
"""

import os
import glob
import logging
import pickle
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split

log = logging.getLogger(__name__)


# =============================================================================
#  Dataset
# =============================================================================

class AstrobeeIMUDataset(Dataset):
    """
    Sliding-window tabanlı IMU Denoising Dataset.

    Normalizasyon stratejisi
    ────────────────────────
    X  (ham IMU)      : z-score  → gürültülü ivme ile jiroskop arasındaki
                        büyüklük farkını dengeler
    ya (gerçek ivme)  : z-score  → loss ölçeğini dengeler; fizik kaybı
                        için ya_std/ya_mean Trainer'a iletilir
    yp (konum)        : normalize EDİLMEZ → kinematik kaba kullanılan ikinci
                        türev fiziksel birimde (m/s²) kalmalıdır
    """

    IMU_COLS   = ['ax', 'ay', 'az', 'gx', 'gy', 'gz']
    ACCEL_COLS = ['true_ax', 'true_ay', 'true_az']
    POS_COLS   = ['px', 'py', 'pz']

    def __init__(
        self,
        csv_paths: List[str],
        seq_len:   int            = 200,
        stride:    int            = 50,     # ≥ 1; 50 → %75 örtüşme
        stats:     Optional[Dict] = None,   # dışarıdan verilen normalizasyon
        fit_stats: bool           = True,
    ):
        self.seq_len = seq_len
        self.stride  = stride

        # Ham pencereler (liste → sonra stack)
        _X:  List[np.ndarray] = []
        _ya: List[np.ndarray] = []
        _yp: List[np.ndarray] = []

        for path in csv_paths:
            if not os.path.isfile(path):
                log.warning(f"Bulunamadı, atlanıyor: {path}")
                continue

            df = pd.read_csv(path)

            # Kolon doğrulama
            required = set(self.IMU_COLS + self.ACCEL_COLS + self.POS_COLS)
            missing  = required - set(df.columns)
            if missing:
                log.warning(f"{os.path.basename(path)}: eksik kolon {missing} — atlanıyor.")
                continue

            X_arr  = df[self.IMU_COLS].values.astype(np.float32)
            ya_arr = df[self.ACCEL_COLS].values.astype(np.float32)
            yp_arr = df[self.POS_COLS].values.astype(np.float32)

            # Sliding window
            n = len(X_arr)
            n_win = 0
            for start in range(0, n - seq_len + 1, stride):
                end = start + seq_len
                _X.append(X_arr[start:end])
                _ya.append(ya_arr[start:end])
                _yp.append(yp_arr[start:end])
                n_win += 1

            log.info(f"  {os.path.basename(path)}: {n} satır → {n_win} pencere")

        if not _X:
            raise RuntimeError("Hiçbir geçerli pencere oluşturulamadı. CSV yollarını kontrol edin.")

        self.X  = np.stack(_X,  axis=0)   # (N, T, 6)
        self.ya = np.stack(_ya, axis=0)   # (N, T, 3)
        self.yp = np.stack(_yp, axis=0)   # (N, T, 3)

        log.info(f"Toplam pencere: {len(self.X)}")

        # Normalizasyon istatistikleri
        if stats is not None:
            self.stats = stats
        elif fit_stats:
            self.stats = self._compute_stats()
        else:
            # Kimlik dönüşümü (normalizasyon yok)
            self.stats = {
                'X_mean':  np.zeros(6, dtype=np.float32),
                'X_std':   np.ones(6,  dtype=np.float32),
                'ya_mean': np.zeros(3, dtype=np.float32),
                'ya_std':  np.ones(3,  dtype=np.float32),
            }

    # ─── İstatistik Hesaplama ────────────────────────────────────────────────

    def _compute_stats(self) -> Dict[str, np.ndarray]:
        X_flat  = self.X.reshape(-1, 6)
        ya_flat = self.ya.reshape(-1, 3)

        X_mean  = X_flat.mean(0);   X_std  = X_flat.std(0)
        ya_mean = ya_flat.mean(0);  ya_std = ya_flat.std(0)

        # Sıfır std: sabit feature'ı bölme hatasından koru
        X_std  = np.where(X_std  < 1e-8, 1.0, X_std).astype(np.float32)
        ya_std = np.where(ya_std < 1e-8, 1.0, ya_std).astype(np.float32)

        log.info(f"X  mean: {X_mean.round(5)}")
        log.info(f"X  std : {X_std.round(5)}")
        log.info(f"ya mean: {ya_mean.round(5)}")
        log.info(f"ya std : {ya_std.round(5)}")

        return {
            'X_mean':  X_mean.astype(np.float32),
            'X_std':   X_std,
            'ya_mean': ya_mean.astype(np.float32),
            'ya_std':  ya_std,
        }

    # ─── Dataset Arayüzü ────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Dönüş:
          X  : (seq_len, 6)  normalize edilmiş ham IMU
          ya : (seq_len, 3)  normalize edilmiş smooth ivme  (loss hedefi)
          yp : (seq_len, 3)  ham konum m cinsinden          (kinematik loss)
        """
        X  = (self.X[idx]  - self.stats['X_mean'])  / self.stats['X_std']
        ya = (self.ya[idx] - self.stats['ya_mean']) / self.stats['ya_std']
        yp = self.yp[idx].copy()  # fiziksel birim, normalize ETME

        return (
            torch.from_numpy(X.astype(np.float32)),
            torch.from_numpy(ya.astype(np.float32)),
            torch.from_numpy(yp.astype(np.float32)),
        )


# =============================================================================
#  DataLoader Fabrikası
# =============================================================================

def build_dataloaders(
    csv_paths:    Optional[List[str]] = None, # None gelirse klasörü tara
    seq_len:      int   = 200,
    stride:       int   = 50,
    batch_size:   int   = 32,
    train_ratio:  float = 0.70,
    val_ratio:    float = 0.15,
    num_workers:  int   = 2,
    stats_path:   Optional[str] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    """
    Tüm CSV'lerden Dataset oluştur, Train/Val/Test'e böl.
    """
    # EĞER csv_paths BOŞSA KLASÖRDEKİ TÜM CSV'LERİ BUL
    if csv_paths is None:
        csv_paths = glob.glob("*.csv")
        if not csv_paths:
            raise FileNotFoundError("Mevcut dizinde hiç CSV dosyası bulunamadı!")

    log.info(f"Dataset yükleniyor ({len(csv_paths)} dosya) ...")
    full_ds = AstrobeeIMUDataset(
        csv_paths, seq_len=seq_len, stride=stride, fit_stats=True
    )
    stats = full_ds.stats

    if stats_path:
        os.makedirs(os.path.dirname(stats_path) or ".", exist_ok=True)
        with open(stats_path, 'wb') as f:
            pickle.dump(stats, f)
        log.info(f"Normalizasyon istatistikleri kaydedildi: {stats_path}")

    N       = len(full_ds)
    n_train = int(N * train_ratio)
    n_val   = int(N * val_ratio)
    n_test  = N - n_train - n_val

    train_ds, val_ds, test_ds = random_split(
        full_ds,
        [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42),
    )

    log.info(f"Bölüm → Train: {n_train} | Val: {n_val} | Test: {n_test}")

    _kw = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    train_dl = DataLoader(train_ds, shuffle=True,  drop_last=True,  **_kw)
    val_dl   = DataLoader(val_ds,   shuffle=False, drop_last=False, **_kw)
    test_dl  = DataLoader(test_ds,  shuffle=False, drop_last=False, **_kw)

    return train_dl, val_dl, test_dl, stats
