# WESAD Veri Seti Üzerinde Akut Stres Tahmini ve Erken Uyarı Sistemi

Bu proje, giyilebilir sensör teknolojilerinden elde edilen çok modlu (multimodal) fizyolojik sinyalleri kullanarak akut stres durumlarını henüz gerçekleşmeden önce tahmin etmeyi amaçlayan bir derin öğrenme sistemidir. Literatürdeki mevcut çalışmaların aksine, sistem stres anını tespit etmek yerine (detection), stres başlangıcından önceki 60 saniyelik "Erken Uyarı" (Early Warning) fazına odaklanmaktadır.

---

## 1. Proje Mimari Yapısı

Sistem, farklı örnekleme frekanslarına sahip verileri eş zamanlı işleyebilen **Üç Dallı Hibrit Mimari (Tri-Branch Hybrid Architecture)** üzerine kurulmuştur:

**Yavaş Kanal (Slow Stream):**  
- EDA (Elektrodermal Aktivite), TEMP (Cilt Sıcaklığı) ve ACC (İvmeölçer) sinyallerini 4 Hz frekansında işler.  
- Mekânsal öznitelikler için 1D-CNN, zamansal bağımlılıklar için LSTM katmanlarını kullanır.

**Hızlı Kanal (Fast Stream):**  
- Kardiyovasküler dinamikleri yakalamak adına BVP (Kan Hacmi Pulsasyonu) sinyalini 64 Hz frekansında, derinleştirilmiş 1D-CNN ve LSTM blokları ile analiz eder.

**Uzman Kolu (Expert Branch):**  
- Sinyallerden çıkarılan 24 adet istatistiksel özniteliği (Mean, Std, RMSSD, Slope, Peak Count vb.) içeren vektörü Dense katmanları üzerinden sisteme entegre eder.

---

## 2. Veri Seti Özellikleri

- **Veri Seti:** WESAD (giyilebilir stres tespiti alanında referans kabul edilen veri seti)  
- **Özne Sayısı:** 15 denek (S2 - S17)  
- **Cihaz:** Bilek tipi Empatica E4 sensörleri  
- **Pencereleme:** 60 saniyelik kayar pencere (sliding window)  
- **Etiketleme:** Stres başlangıcından (onset) önceki 60 saniyelik süre %20 eşik değeriyle Erken Uyarı (EW) olarak tanımlanmıştır

---

## 3. Kurulum ve Gereksinimler

Proje Python 3.11+ ortamında geliştirilmiştir. Gerekli kütüphaneleri yüklemek için:

```bash
pip install numpy scikit-learn tensorflow keras pandas scipy
```
---

## 4. Kullanım Adımları (Pipeline)

Sistemi uçtan uca çalıştırmak için aşağıdaki işlem sırası takip edilmelidir:

- **Veri Hazırlama:** Ham WESAD verilerinin senkronize edilmesi ve birleştirilmesi
```bash
python dataset_aggregator.py
```
- **Öznitelik Çıkarımı:** İstatistiksel uzman özniteliklerin hesaplanması
```bash
python feature_extraction.py
```
- **Model Eğitimi:** İstatistiksel kararlılık için 5 turlu bağımsız eğitim döngüsü
```bash
python train_multimodal_loop.py
```
---

## 5. Performans Metrikleri

Modelin istatistiksel analiz sonuçları, özellikle Erken Uyarı (EW) sınıfındaki düşük veri temsiline rağmen stabil performans göstermektedir:

| Metrik                    | Değer (Ortalama ± Std) |
|----------------------------|-----------------------:|
| Genel Accuracy             | %85.18 ± 5.9          |
| Akut Stres F1-Score        | 0.9116 ± 0.04         |
| Erken Uyarı (EW) F1-Score  | 0.4674 ± 0.08         |
| EW Recall                  | 0.5794 ± 0.22         |
---

## 6. Akademik Bağlam

Bu çalışma, OSTİM Teknik Üniversitesi Bilgisayar Mühendisliği Bölümü bünyesinde, Dr. Öğr. Üyesi Barış Taha ULUDAĞ danışmanlığında yürütülmektedir. Proje kapsamında geliştirilen EW etiketleme prosedürü ve hibrit mimari, giyilebilir sağlık teknolojilerinde proaktif stres yönetimi için temel teşkil etmektedir.

- **Yazar:** Ezgi Sarıca
- **E-posta:** ezgi.sarica@outlook.com
- **Proje Durumu:** Aktif Geliştirme Aşamasında (V7 Final)
