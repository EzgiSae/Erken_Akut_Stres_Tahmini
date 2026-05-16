"""
WESAD Veri Seti - Derin Öğrenme İçin Çok Modlu (Multimodal) Veri Kümesi İnşası

Bu modül, ön işlemeden geçmiş fizyolojik sinyalleri kayar pencere (sliding window) 
yöntemiyle dilimler ve hibrit derin öğrenme modelleri (CNN-LSTM) için 
girdi tensörlerini oluşturur.

Özellikler:
- Çok Modlu Yapı: Yavaş (4 Hz) ve Hızlı (64 Hz) kanalları eş zamanlı pencereler.
- Etiket Karar Mekanizması: Pencere içindeki baskın duruma göre etiket ataması.
- İkili Sınıflandırma: Erken Uyarı (EW=0) ve Akut Stres (Stress=1) dönüşümü.

Geliştiren: Ezgi Sarıca
"""

import numpy as np
from preprocess import preprocess_subject

def build_multimodal_windows(data, window_seconds=30, slow_fs=4, fast_fs=64, step_seconds=1):
    """
    Sinyalleri zaman pencerelerine böler ve çok modlu giriş tensörlerini hazırlar.
    
    Argümanlar:
        data (dict): Ön işlemeden geçmiş sinyalleri içeren sözlük.
        window_seconds (int): Pencere uzunluğu (Saniye).
        slow_fs (int): Yavaş kanalların frekansı (4 Hz).
        fast_fs (int): Hızlı kanalın frekansı (64 Hz).
        step_seconds (int): Kaydırma (stride) miktarı (Saniye).
        
    Döndürür:
        tuple: (X_slow, X_fast, y_raw) - Ham etiketli tensör dizileri.
    """
    
    # Pencere ve adım boyutlarını örnek sayısına (sample count) dönüştür
    slow_window_size = window_seconds * slow_fs
    slow_step_size = step_seconds * slow_fs
    ratio = int(fast_fs / slow_fs)
    
    # Yavaş kanalları (EDA, TEMP, ACC) tek bir matriste birleştir
    slow_features = np.column_stack([data["EDA"], data["TEMP"], data["ACC"]])
    fast_features = data["BVP"]
    labels = data["Label"]

    X_slow, X_fast, y = [], [], []

    # Kayar pencere döngüsü
    for slow_start in range(0, len(slow_features) - slow_window_size + 1, slow_step_size):
        slow_end = slow_start + slow_window_size
        fast_start, fast_end = slow_start * ratio, slow_end * ratio

        window_slow = slow_features[slow_start:slow_end]
        window_fast = fast_features[fast_start:fast_end]
        window_labels = labels[slow_start:slow_end]

        # Etiket Karar Mekanizması: 
        # Pencerenin en az %20'si Erken Uyarı (200) ise pencereyi EW olarak işaretle.
        # Aksi halde son örneğin etiketini kabul et.
        ew_ratio = np.mean(window_labels == 200)
        if ew_ratio >= 0.2:
            window_label = 200
        else:
            window_label = labels[slow_end - 1]

        # Sadece ilgilenilen sınıfları (EW ve Stress) veri kümesine dahil et
        if window_label in [200, 2]:
            X_slow.append(window_slow)
            # Hızlı kanal için kanal boyutunu (channel dimension) ekle
            X_fast.append(window_fast.reshape(-1, 1))
            y.append(window_label)

    return np.array(X_slow), np.array(X_fast), np.array(y)

def build_subject_dataset(subject_id, ew_seconds=60, slow_fs=4, fast_fs=64,
                          window_seconds=30, step_seconds=1):
    """
    Belirli bir denek için tüm işlem hattını yürüterek final veri kümesini oluşturur.
    """
    # 1. Ön işleme hattını çalıştır
    data = preprocess_subject(subject_id, slow_fs=slow_fs, fast_fs=fast_fs, ew_seconds=ew_seconds)

    # 2. Kayar pencere tensörlerini oluştur
    X_slow, X_fast, y_raw = build_multimodal_windows(
        data, window_seconds, slow_fs, fast_fs, step_seconds
    )

    # 3. Etiketleri model eğitimine uygun ikili (binary) formata dönüştür
    # EW (200) -> 0, Stress (2) -> 1
    label_map = {200: 0, 2: 1}
    y = np.array([label_map[label] for label in y_raw])

    return X_slow, X_fast, y

if __name__ == "__main__":
    # Test ve doğrulama senaryosu
    SUBJECT_ID = "S2"
    print(f"\n[INFO] {SUBJECT_ID} için veri kümesi yapılandırılıyor...")
    
    xs, xf, labels = build_subject_dataset(SUBJECT_ID)

    print("-" * 30)
    print(f"Pencere Sayısı: {len(labels)}")
    print(f"X_slow (4Hz)  : {xs.shape}")
    print(f"X_fast (64Hz) : {xf.shape}")
    
    unique, counts = np.unique(labels, return_counts=True)
    dist = dict(zip(unique, counts))
    print(f"Sınıf Dağılımı: 0 (EW): {dist.get(0, 0)}, 1 (Stres): {dist.get(1, 0)}")