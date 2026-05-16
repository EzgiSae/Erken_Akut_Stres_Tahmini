"""
WESAD Veri Seti - Denekler Arası Veri Birleştirme ve Kümeleme Modülü

Bu modül, bireysel denek bazında hazırlanan pencerelenmiş verileri toplar, 
eğitim ve test kümelerini oluşturur ve model eğitimine hazır 
sıkıştırılmış (.npz) formatta diske kaydeder.

Bilimsel Yaklaşım:
- Denek Bazlı Ayrım (Subject-wise Split): Modelin genelleme yeteneğini 
  test etmek amacıyla eğitim ve test kümeleri farklı deneklerden oluşturulmuştur.
"""

import numpy as np
from dataset_builder import build_subject_dataset

def aggregate_data(subject_list):
    """
    Belirlenen denek listesi için verileri oluşturur ve birleştirir.
    """
    x_slow_list, x_fast_list, y_list = [], [], []

    for sid in subject_list:
        print(f"   > {sid} verisi işleniyor...")
        try:
            xs, xf, y = build_subject_dataset(sid)
            x_slow_list.append(xs)
            x_fast_list.append(xf)
            y_list.append(y)
        except Exception as e:
            print(f"   ! Hata: {sid} işlenirken sorun oluştu: {e}")

    return (
        np.concatenate(x_slow_list, axis=0),
        np.concatenate(x_fast_list, axis=0),
        np.concatenate(y_list, axis=0)
    )

def main():
    train_subjects = ["S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10", "S11", "S13", "S14"]
    test_subjects  = ["S15", "S16", "S17"]
    save_path = "dl_multimodal_dataset.npz"

    print("\n--- Çok Modlu Veri Seti Oluşturma İşlemi Başladı ---")

    # 1. Eğitim Kümesi
    print(f"\n[1/3] Eğitim kümesi hazırlanıyor ({len(train_subjects)} denek)...")
    x_train_slow, x_train_fast, y_train = aggregate_data(train_subjects)

    # 2. Test Kümesi
    print(f"\n[2/3] Test kümesi hazırlanıyor ({len(test_subjects)} denek)...")
    x_test_slow, x_test_fast, y_test = aggregate_data(test_subjects)

    # 3. Kaydetme (Hata düzeltildi. Değişken isimleri senkronize edildi)
    print(f"\n[3/3] Veriler kaydediliyor: '{save_path}'")
    
    np.savez(
        save_path,
        X_train_slow=x_train_slow,
        X_train_fast=x_train_fast,
        y_train=y_train,
        X_test_slow=x_test_slow,
        X_test_fast=x_test_fast,
        y_test=y_test
    )

    print("\n✅ İşlem başarıyla tamamlandı.")

if __name__ == "__main__":
    main()