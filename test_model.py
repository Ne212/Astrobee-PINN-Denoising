import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from model import AstrobeePINN
from dataset_loader import AstrobeeIMUDataset

def test_and_plot(csv_file):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Modeli Yükle
    model = AstrobeePINN().to(device)
    model.load_state_dict(torch.load("astrobee_pinn_model.pth"))
    model.eval()

    # 2. Veriyi Hazırla (Test için daha uzun bir pencere alalım)
    dataset = AstrobeeIMUDataset(csv_file, window_size=600, step_size=600)
    
    # 5. pencereyi test edelim (Hareketin olduğu bir anı yakalamak için)
    sample_idx = 5 if len(dataset) > 5 else 0
    sample = dataset[sample_idx]
    
    imu_input = sample['imu'].unsqueeze(0).to(device)
    true_accel_gt = sample['true_accel_gt'].numpy() # Gerçek temiz ivme (Referans)

    # 3. Tahmin Yürüt
    with torch.no_grad():
        a_clean_pred, params = model(imu_input)
        D, DT, sigma, creep = params

    # CPU'ya çek
    imu_dirty = imu_input.cpu().squeeze().numpy()
    imu_fixed = a_clean_pred.cpu().squeeze().numpy()
    
    # 4. GRAFİK ÇİZİMİ
    plt.figure(figsize=(14, 10))
    
    # --- Üst Grafik: İvme Karşılaştırması (X Ekseni) ---
    plt.subplot(2, 1, 1)
    plt.plot(imu_dirty[:, 0], label='Ham İvme (Gürültülü/Sayısal)', color='lightskyblue', alpha=0.6)
    plt.plot(true_accel_gt[:, 0], label='Hedef Temiz İvme (True Ax)', color='green', linestyle='--', linewidth=2)
    plt.plot(imu_fixed[:, 0], label='YZ Düzeltilmiş İvme', color='red', linewidth=1.5)
    
    plt.title(f"Astrobee İvme Düzeltme Analizi ({csv_file})")
    plt.xlabel("Zaman Adımı (Sample)")
    plt.ylabel("İvme (m/s²)")
    plt.legend(loc='upper right')
    plt.grid(True, which='both', linestyle='--', alpha=0.5)

    # --- Alt Grafik: Tahmin Edilen Fiziksel Parametreler ---
    plt.subplot(2, 1, 2)
    plt.plot(creep.cpu().squeeze().numpy()[:, 0], label='Tahmin Edilen Sünme (Creep X)', color='darkgreen')
    plt.plot(DT.cpu().squeeze().numpy() / 15.0, label='Tahmin Edilen Termal Etki (Normalize)', color='darkorange')
    plt.plot(D.cpu().squeeze().numpy(), label='Sensör Yorgunluğu (D)', color='purple')
    
    plt.title("Yapay Zekanın Arka Planda Tahmin Ettiği Fiziksel Katsayılar")
    plt.xlabel("Zaman Adımı (Sample)")
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # İstediğin bir CSV dosyasıyla test et
    test_and_plot("iva_hatch_inspection2_dataset.csv")