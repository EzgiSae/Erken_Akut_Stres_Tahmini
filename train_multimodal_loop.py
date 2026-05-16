"""
WESAD Veri Seti - İstatistiksel Model Değerlendirme ve Yinelemeli Eğitim Modülü

Bu modül, multimodal modelin performans tutarlılığını ölçmek amacıyla 
belirlenen sayıda (varsayılan: 5) bağımsız eğitim turu gerçekleştirir. 

Analiz Kapsamı:
- Modelin her turdaki Accuracy, Precision, Recall ve F1-Score değerlerini toplar.
- Elde edilen metriklerin aritmetik ortalamasını ve standart sapmasını hesaplar.
- Modelin başlangıç ağırlıklarından bağımsız olarak genelleme başarısını doğrular.

Bu dosya, akademik raporlardaki istatistiksel tabloların oluşturulması için temel teşkil eder.
"""

import numpy as np
from train_multimodal_single import load_final_datasets, build_multimodal_model
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from keras.callbacks import EarlyStopping

def main(runs=5):
    # 1. Veri Setinin Yüklenmesi
    (X_tr_slow, X_tr_fast, X_tr_feat, y_train, 
     X_te_slow, X_te_fast, X_te_feat, y_test) = load_final_datasets()

    # Öznitelik normalizasyonu (Eğitim seti parametreleri ile)
    scaler = StandardScaler()
    X_tr_feat = scaler.fit_transform(X_tr_feat)
    X_te_feat = scaler.transform(X_te_feat)

    # Metrikleri depolamak için listeler
    results = {
        "acc": [],
        "ew_prec": [], "ew_rec": [], "ew_f1": [],
        "st_prec": [], "st_rec": [], "st_f1": []
    }

    print(f"\n--- İstatistiksel Analiz Başladı: {runs} Bağımsız Eğitim Turu ---")

    for run in range(runs):
        print(f"\n[Tur {run+1}/{runs}] Model eğitiliyor...")
        
        # Her turda taze bir model nesnesi oluşturulur
        model = build_multimodal_model((120,3), (1920,1), (24,))
        early_stop = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
        
        model.fit(
            x=[X_tr_slow, X_tr_fast, X_tr_feat],
            y=y_train,
            validation_split=0.2,
            epochs=50,
            batch_size=32,
            class_weight={0: 2.5, 1: 1.0},
            callbacks=[early_stop],
            verbose=0 # Konsol kalabalığını önlemek için sessiz mod
        )

        # Tahmin ve Metrik Hesaplama
        y_prob = model.predict([X_te_slow, X_te_fast, X_te_feat], verbose=0)
        y_pred = (y_prob > 0.5).astype(int)
        
        report = classification_report(
            y_test, y_pred, 
            target_names=["EW", "Stress"], 
            output_dict=True, 
            zero_division=0
        )

        # Sonuçları kaydet
        results["acc"].append(report["accuracy"])
        results["ew_prec"].append(report["EW"]["precision"])
        results["ew_rec"].append(report["EW"]["recall"])
        results["ew_f1"].append(report["EW"]["f1-score"])
        results["st_prec"].append(report["Stress"]["precision"])
        results["st_rec"].append(report["Stress"]["recall"])
        results["st_f1"].append(report["Stress"]["f1-score"])

    # 2. İstatistiksel Özet Raporu
    print("\n" + "="*50)
    print(f"RESMİ PERFORMANS RAPORU ({runs} Tur Ortalaması)")
    print("="*50)
    
    def print_stat(name, data_list):
        mean = np.mean(data_list)
        std = np.std(data_list)
        print(f"{name:<12}: {mean:.4f} ± {std:.4f}")

    print_stat("Genel Accuracy", results["acc"])
    print("-" * 30)
    print("Erken Uyarı (EW) Sınıfı:")
    print_stat("  Precision", results["ew_prec"])
    print_stat("  Recall", results["ew_rec"])
    print_stat("  F1-Score", results["ew_f1"])
    print("-" * 30)
    print("Akut Stres Sınıfı:")
    print_stat("  Precision", results["st_prec"])
    print_stat("  Recall", results["st_rec"])
    print_stat("  F1-Score", results["st_f1"])
    print("="*50 + "\n")

if __name__ == "__main__":
    # Raporlama için 5 tur idealdir
    main(runs=5)