"""
WESAD Veri Seti - Erken Uyarı (Early Warning) Etiketi Oluşturucu

Bu modül, WESAD veri setindeki ham etiket dizilerini analiz ederek stres 
başlangıcını (onset) tespit eder ve bu başlangıçtan önceki belirli bir 
zaman dilimini 'Erken Uyarı (Early Warning)' fazı olarak işaretler.

Geliştiren: Ezgi Sarıca
Proje: Akut Stres Tahmini ve Erken Uyarı Sistemi
"""

import numpy as np
from data_loader import load_wesad_subject

# --- Sabit Tanımlamalar ---
WESAD_STRESS_LABEL = 2
EARLY_WARNING_LABEL = 200
# WESAD Etiket Haritası (Raporlama ve görselleştirme için)
LABEL_MAP = {
    0: "Transient",
    1: "Baseline",
    2: "Stress",
    3: "Amusement",
    4: "Meditation",
    EARLY_WARNING_LABEL: "Early-Warning"
}

def find_stress_onset(labels):
    """
    Ham etiket dizisi içerisindeki ilk stres (label=2) başlangıç indeksini tespit eder.
    
    Argümanlar:
        labels (np.array): WESAD ham etiket dizisi.
        
    Döndürür:
        int: İlk stres onset indeksi. Stres durumu yoksa None döner.
    """
    # Vektörize edilmiş işlem: Etiket değişim noktalarını tespit et
    stress_mask = (labels == WESAD_STRESS_LABEL).astype(int)
    diff = np.diff(stress_mask)
    
    # 0'dan 1'e geçiş (başlangıç) noktalarını bul
    onsets = np.where(diff == 1)[0]
    
    return onsets[0] + 1 if len(onsets) > 0 else None

def build_early_warning_labels(labels, label_fs=700, early_warning_seconds=60):
    """
    Stres başlangıcından önceki belirli bir süreyi 'Early-Warning' (200) olarak etiketler.
    
    Argümanlar:
        labels (np.array): Ham etiket dizisi.
        label_fs (int): Etiket örnekleme frekansı (Varsayılan: 700 Hz).
        early_warning_seconds (int): Onset öncesi kapsanacak saniye süresi.
        
    Döndürür:
        tuple: (new_labels, stress_onset_idx, ew_start_idx)
    """
    new_labels = np.array(labels).copy()
    stress_onset_idx = find_stress_onset(new_labels)

    if stress_onset_idx is None:
        return new_labels, None, None

    # Zaman penceresini örnek sayısına (sample count) çevir
    ew_window_size = int(early_warning_seconds * label_fs)
    ew_start_idx = max(0, stress_onset_idx - ew_window_size)

    # Belirlenen aralığa yeni etiketi ata
    new_labels[ew_start_idx:stress_onset_idx] = EARLY_WARNING_LABEL

    return new_labels, stress_onset_idx, ew_start_idx

def summarize_label_counts(labels):
    """
    Veri kümesindeki etiket dağılımını teknik rapor formatında özetler.
    """
    unique, counts = np.unique(labels, return_counts=True)
    return {LABEL_MAP.get(l, f"Unknown({l})"): c for l, c in zip(unique, counts)}

def debug_early_warning(subject_id="S2", early_warning_seconds=60):
    """
    Etiket oluşturma mantığını doğrulamak için kullanılan test fonksiyonu.
    """
    try:
        _, labels, fs_dict = load_wesad_subject(subject_id)
        
        new_labels, onset_idx, start_idx = build_early_warning_labels(
            labels, fs_dict["Label"], early_warning_seconds
        )

        if onset_idx is None:
            print(f"Bilgi: {subject_id} deneginde stres durumu bulunamadı.")
            return

        print(f"\n--- {subject_id} Analizi ---")
        print(f"Stres Baslangici: {onset_idx / fs_dict['Label'] / 60:.2f}. dakika")
        print(f"EW Penceresi: {early_warning_seconds} sn")
        
        dist = summarize_label_counts(new_labels)
        print("Etiket Dagilimi:")
        for label_name, count in dist.items():
            print(f"  - {label_name}: {count}")

    except Exception as e:
        print(f"Hata: {e}")

if __name__ == "__main__":
    # Test ve doğrulama
    debug_early_warning(subject_id="S2", early_warning_seconds=60)