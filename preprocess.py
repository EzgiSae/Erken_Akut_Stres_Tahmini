"""
WESAD Veri Seti - Multimodal Ön İşleme ve Sinyal Senkronizasyon Modülü

Bu modül, farklı örnekleme frekanslarına sahip fizyolojik sinyalleri 
ortak bir zaman düzleminde senkronize eder, gürültü azaltma için 
normalizasyon uygular ve multimodal model eğitimine hazır hale getirir.

İşlemler:
- Sinyallerin hedef frekanslara (4 Hz ve 64 Hz) yeniden örneklenmesi.
- Label verisinin 'Majority Voting' yöntemiyle hizalanması.
- Denek bazlı (Subject-wise) Z-Score normalizasyonu.
- İvmeölçer (ACC) verisinden vektörel büyüklük (Magnitude) çıkarımı.

Geliştiren: Ezgi Sarıca
"""

import numpy as np
from scipy import signal
from math import gcd

from data_loader import load_wesad_subject
from label_builder import build_early_warning_labels

def resample_signal(data, original_fs, target_fs):
    """
    Sinyali, spektral bozulmayı önlemek için polyphase resampling yöntemiyle 
    hedef örnekleme frekansına dönüştürür.
    """
    original_fs, target_fs = int(original_fs), int(target_fs)
    if original_fs == target_fs:
        return data

    # En büyük ortak bölen üzerinden up/down katsayılarını hesapla
    common_divisor = gcd(original_fs, target_fs)
    up = target_fs // common_divisor
    down = original_fs // common_divisor

    return signal.resample_poly(data, up, down)

def resample_labels(labels, original_fs, target_fs):
    """
    Etiket dizisini indirgerken bilgi kaybını önlemek için 'Majority Voting' kullanır.
    Her pencere içerisindeki en sık tekrar eden etiketi baskın etiket olarak seçer.
    """
    original_fs, target_fs = int(original_fs), int(target_fs)
    if original_fs == target_fs:
        return labels

    factor = int(original_fs / target_fs)
    new_labels = []

    # Belirlenen faktör aralığında en sık geçen etiketi tespit et
    for i in range(0, len(labels), factor):
        window = labels[i:i + factor]
        if len(window) > 0:
            majority_label = np.bincount(window.astype(int)).argmax()
            new_labels.append(majority_label)

    return np.array(new_labels)

def subject_normalize(data):
    """
    Denekler arası fizyolojik temel seviye farklılıklarını gidermek için 
    Z-Score normalizasyonu (Standardizasyon) uygular.
    """
    # Sıfıra bölünme hatasını önlemek için küçük bir epsilon (1e-8) eklenmiştir
    return (data - np.mean(data)) / (np.std(data) + 1e-8)

def preprocess_subject(subject_id, slow_fs=4, fast_fs=64, ew_seconds=60):
    """
    Belirtilen denek için tüm ön işleme ve senkronizasyon adımlarını yürütür.
    
    Çıktı Yapısı:
    - EDA, TEMP, ACC, Label: slow_fs (4 Hz) frekansına senkronize edilir.
    - BVP: fast_fs (64 Hz) frekansında korunur (Yüksek çözünürlüklü kanal).
    """
    # 1. Ham veriyi ve etiketleri yükle
    wrist, labels, fs = load_wesad_subject(subject_id)

    # 2. Erken Uyarı (Early Warning) mantığını uygula
    new_labels, _, _ = build_early_warning_labels(
        labels=labels,
        label_fs=fs["Label"],
        early_warning_seconds=ew_seconds
    )

    # 3. Yavaş Kanallar: EDA, TEMP ve ACC (Magnitude) -> 4 Hz
    eda = resample_signal(wrist["EDA"], fs["EDA"], slow_fs)
    temp = resample_signal(wrist["TEMP"], fs["TEMP"], slow_fs)
    
    # 3 eksenli ACC verisinden hareket şiddetini temsil eden magnitude hesapla
    acc = wrist["ACC"]
    acc_mag = np.sqrt(np.sum(np.square(acc), axis=1))
    acc_mag = resample_signal(acc_mag, fs["ACC"], slow_fs)

    # 4. Hızlı Kanal: BVP -> 64 Hz (Kardiyovasküler dinamiklerin korunması için)
    bvp = resample_signal(wrist["BVP"], fs["BVP"], fast_fs)

    # 5. Etiketlerin yavaş kanal frekansına (4 Hz) indirgenmesi
    labels_resampled = resample_labels(
        new_labels,
        original_fs=fs["Label"],
        target_fs=slow_fs
    )

    # 6. Zaman Senkronizasyonu (Temporal Alignment)
    # Hızlı ve yavaş kanallar arasındaki örnekleme oranını belirle
    ratio = int(fast_fs / slow_fs)
    
    # Tüm sinyalleri en kısa olanın uzunluğuna göre kırparak hizala
    min_len_slow = min(
        len(eda), len(temp), len(acc_mag), 
        len(labels_resampled), len(bvp) // ratio
    )

    # 7. Normalizasyon ve Veri Paketleme
    processed_data = {
        "EDA": subject_normalize(eda[:min_len_slow]),
        "TEMP": subject_normalize(temp[:min_len_slow]),
        "ACC": subject_normalize(acc_mag[:min_len_slow]),
        "BVP": subject_normalize(bvp[:min_len_slow * ratio]),
        "Label": labels_resampled[:min_len_slow]
    }

    return processed_data