"""
WESAD Veri Seti - Çok Modlu Öznitelik Çıkarım (Feature Extraction) Modülü

Bu modül, pencerelenmiş ham fizyolojik sinyalleri (EDA, BVP, TEMP, ACC) 
istatistiksel ve sinyal işleme tabanlı 24 farklı özniteliğe dönüştürür. 

Çıkarılan Öznitelikler:
- EDA (8): Ortalama, sapma, aralık, eğim, tepe noktası sayısı vb.
- BVP (6): Ortalama, enerji, RMSSD, ZCR (Zero Crossing Rate) vb.
- TEMP (4): Termal değişim hızı ve varyans özellikleri.
- ACC (6): Hareket şiddeti ve enerji yoğunluğu özellikleri.

Bu işlem, derin öğrenme modeline destekleyici bilgi sağlamak veya 
geleneksel makine öğrenmesi modelleri için girdi oluşturmak amacıyla kullanılır.
"""

import numpy as np

# --- Öznitelik İsim Tanımlamaları ---
FEATURE_NAMES = [
    # EDA (8 adet)
    "EDA_mean", "EDA_std", "EDA_range", "EDA_slope", "EDA_peak_count", 
    "EDA_abs_diff_mean", "EDA_positive_slope_ratio", "EDA_scl_p10",
    # BVP (6 adet)
    "BVP_mean", "BVP_std", "BVP_energy", "BVP_rmssd", "BVP_zcr", "BVP_peak_interval_std",
    # TEMP (4 adet)
    "TEMP_mean", "TEMP_std", "TEMP_slope", "TEMP_diff_var",
    # ACC (6 adet)
    "ACC_mean", "ACC_std", "ACC_energy", "ACC_mad", "ACC_iqr", "ACC_range",
]

def extract_features_from_window(window_slow, window_fast):
    """
    Tek bir zaman penceresi üzerinden 24 adet öznitelik hesaplar.
    
    Argümanlar:
        window_slow (np.array): 4 Hz örneklenmiş (EDA, TEMP, ACC) sinyalleri.
        window_fast (np.array): 64 Hz örneklenmiş BVP sinyali.
    """
    features = []

    # Sinyal kanallarını ayrıştır
    eda_sig  = window_slow[:, 0]
    temp_sig = window_slow[:, 1]
    acc_sig  = window_slow[:, 2]  # Magnitude verisi
    bvp_sig  = window_fast.flatten()

    # --- EDA (Elektrodermal Aktivite) Öznitelikleri ---
    features.append(np.mean(eda_sig))
    features.append(np.std(eda_sig))
    features.append(np.max(eda_sig) - np.min(eda_sig))
    # Eğim (Linear Regression Slope) 
    features.append(np.polyfit(np.arange(len(eda_sig)), eda_sig, 1)[0])
    # Tepe noktası (Local Maxima) sayısı
    eda_peaks = np.where((eda_sig[1:-1] > eda_sig[:-2]) & (eda_sig[1:-1] > eda_sig[2:]))[0]
    features.append(len(eda_peaks))
    # Türevsel özellikler
    diff_eda = np.diff(eda_sig)
    features.append(np.mean(np.abs(diff_eda)))
    features.append(np.sum(diff_eda > 0) / len(diff_eda))
    features.append(np.percentile(eda_sig, 10))

    # --- BVP (Kan Hacmi Pulsasyonu) Öznitelikleri ---
    features.append(np.mean(bvp_sig))
    features.append(np.std(bvp_sig))
    features.append(np.sum(np.square(bvp_sig)))  # Enerji
    features.append(np.sqrt(np.mean(np.square(np.diff(bvp_sig)))))  # RMSSD (Kardiyovasküler değişkenlik)
    # Zero Crossing Rate (ZCR)
    bvp_centered = bvp_sig - np.mean(bvp_sig)
    features.append(np.mean(bvp_centered[1:] * bvp_centered[:-1] < 0))
    # BVP Tepe noktaları arası değişkenlik
    bvp_peaks = np.where((bvp_sig[1:-1] > bvp_sig[:-2]) & (bvp_sig[1:-1] > bvp_sig[2:]))[0]
    features.append(np.std(np.diff(bvp_peaks)) if len(bvp_peaks) > 1 else 0)

    # --- TEMP (Cilt Sıcaklığı) Öznitelikleri ---
    features.append(np.mean(temp_sig))
    features.append(np.std(temp_sig))
    features.append(np.polyfit(np.arange(len(temp_sig)), temp_sig, 1)[0]) # Sıcaklık eğimi
    features.append(np.var(np.diff(temp_sig)))

    # --- ACC (İvmeölçer) Öznitelikleri ---
    features.append(np.mean(acc_sig))
    features.append(np.std(acc_sig))
    features.append(np.sum(np.square(acc_sig)))
    features.append(np.mean(np.abs(acc_sig - np.mean(acc_sig))))  # Mean Absolute Deviation
    # Interquartile Range (IQR)
    features.append(np.percentile(acc_sig, 75) - np.percentile(acc_sig, 25))
    features.append(np.max(acc_sig) - np.min(acc_sig))

    return np.array(features)

def extract_dataset_features(x_slow, x_fast):
    """Veri setindeki tüm pencereler için öznitelik matrisini oluşturur."""
    feature_list = []
    total_windows = x_slow.shape[0]
    
    print(f"[INFO] {total_windows} pencere işleniyor...")

    for i in range(total_windows):
        feat = extract_features_from_window(x_slow[i], x_fast[i])
        feature_list.append(feat)
        if (i + 1) % 1000 == 0:
            print(f"   > Tamamlanan: {i + 1}/{total_windows}")

    return np.array(feature_list)

def build_feature_dataset(input_path="dl_multimodal_dataset.npz"):
    """Ham veri setini yükler, öznitelikleri çıkarır ve yeni bir dosya olarak kaydeder."""
    try:
        data = np.load(input_path)
        
        print("\n--- Öznitelik Çıkarım İşlemi Başladı ---")
        
        # Eğitim ve Test setleri için ayrı ayrı işlem yap
        x_train_feat = extract_dataset_features(data["X_train_slow"], data["X_train_fast"])
        x_test_feat  = extract_dataset_features(data["X_test_slow"], data["X_test_fast"])

        save_path = "multimodal_feature_dataset.npz"
        np.savez(
            save_path,
            X_train=x_train_feat,
            y_train=data["y_train"],
            X_test=x_test_feat,
            y_test=data["y_test"],
            feature_names=FEATURE_NAMES
        )

        print(f"\n✅ Başarılı: Öznitelik matrisi oluşturuldu -> {save_path}")
        print(f"Boyut: {x_train_feat.shape}")

    except Exception as e:
        print(f"❌ Hata: Veri seti işlenemedi: {e}")

if __name__ == "__main__":
    build_feature_dataset()