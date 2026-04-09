import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq

# 1. KESİNTİSİZ Verileri Yükle
preds = np.load("checkpoints/test_predictions_continuous.npy") # (N_total, 3)
trues = np.load("checkpoints/test_groundtruth_continuous.npy") # (N_total, 3)
raws  = np.load("checkpoints/test_raw_continuous.npy")         # (N_total, 3) KİRLİ VERİ

# 2. Gösterilecek Zaman Aralığını Belirle
START_SEC = 10  # Başlangıç saniyesi (İstediğin gibi değiştirebilirsin)
END_SEC = 30    # Bitiş saniyesi (Örn: 20 saniyelik bir kesit)
fs = 62.5       # Örnekleme frekansı (Hz)

start_idx = int(START_SEC * fs)
end_idx = int(END_SEC * fs)
N_slice = end_idx - start_idx

# Zaman ekseni (sn)
t = np.arange(start_idx, end_idx) / fs

# İncelemek istediğimiz eksen (0: X, 1: Y, 2: Z)
axis = 2
axis_name = ['X', 'Y', 'Z'][axis]

# İlgili zaman aralığını ve ekseni kes
y_raw  = raws[start_idx:end_idx, axis]
y_true = trues[start_idx:end_idx, axis]
y_pred = preds[start_idx:end_idx, axis]

plt.figure(figsize=(16, 12))

# --- GRAFİK 1: Zaman Ekseni İvme Karşılaştırması ---
plt.subplot(3, 1, 1)
# Arka plana soluk mavi ile kirli veriyi çiziyoruz
plt.plot(t, y_raw, color='royalblue', alpha=0.4, label='Ham (Kirli) Sensör Verisi', linewidth=1.5)
plt.plot(t, y_true, 'g-', label='Yer Gerçeği (True)', linewidth=2.5)
plt.plot(t, y_pred, 'r-', label='YZ Temizlenmiş İvme', linewidth=2.0)

plt.axhline(0, color='black', linestyle='--', linewidth=1) # Sıfır Çıpası
plt.title(f"Astrobee Kesintisiz Uçuş İvmesi ({axis_name}-Ekseni, {START_SEC}s - {END_SEC}s)", fontsize=13, fontweight='bold')
plt.ylabel("İvme (m/s²)")
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3)

# --- GRAFİK 2: Hata Analizi (Residuals) ---
plt.subplot(3, 1, 2)
error = y_pred - y_true
plt.fill_between(t, error, color='red', alpha=0.3, label='Kalan Hata (Pred - True)')
plt.axhline(0, color='black', linestyle='--', linewidth=1)

plt.title("Zaman Bazlı Tahmin Hatası", fontsize=12)
plt.ylabel("Hata (m/s²)")
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3)

# --- GRAFİK 3: FREKANS ANALİZİ (FFT) ---
plt.subplot(3, 1, 3)
yf_raw  = np.abs(rfft(y_raw))
yf_true = np.abs(rfft(y_true))
yf_pred = np.abs(rfft(y_pred))
xf = rfftfreq(N_slice, 1/fs)

# Frekans Spektrumlarını Çizdir
plt.semilogy(xf, yf_raw, color='royalblue', alpha=0.4, label='Ham Veri Spektrumu (Yüksek Gürültü)')
plt.semilogy(xf, yf_true, 'g', label='İdeal Spektrum (Low-Freq Only)', linewidth=2)
plt.semilogy(xf, yf_pred, 'r', label='YZ Çıktı Spektrumu', alpha=0.8, linewidth=2)

# Eğitimde 6.0 Hz üstünü gürültü kabul ettiğimiz için çizgiyi 6'ya çektik
plt.axvline(6.0, color='blue', linestyle='--', label='Cut-off (6Hz)') 

plt.title("Frekans Spektrumu Analizi (Gürültü Reddi)", fontsize=12)
plt.xlabel("Frekans (Hz)")
plt.ylabel("Güç (Magnitude)")
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"Astrobee_Analiz_{axis_name}_Ekseni.png", dpi=300)
plt.show()