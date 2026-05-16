Giyilebilir Cihaz Verileri ile Akut Stres Erken TahminiBu çalışma, WESAD veri seti üzerinde, Empatica E4 bileklik sensörlerinden elde edilen fizyolojik sinyalleri kullanarak akut stresin başlamasından önceki 60 saniyelik "Erken Uyarı" (Early Warning - EW) dönemini tahmin etmeyi amaçlar.  Projenin Özeti ve Temel BulgularGeleneksel stres tespiti çalışmalarının aksine bu proje, stresin varlığını değil, yaklaşmakta olan stres tepkisini proaktif olarak tahmin etmeye odaklanmıştır.  Kritik Keşif: Stres öncesi dönemin (WESAD'da label=0) büyük oranda "tanımsız/geçiş" etiketlerinden oluştuğu deneysel olarak kanıtlanmıştır.  Model Başarısı: Nihai üç dallı hibrit model, stres tespitinde %95 F1 skoruna ulaşırken, erken uyarı tahmininde 0.46 ± 0.08 F1 değeri ile gerçekçi bir üst sınır belirlemiştir.  Sistem Mimarisi ve PipelineProje, modüler bir sinyal işleme ve derin öğrenme hattı (pipeline) üzerine kurulmuştur.  Kullanılan Sensörler ve FrekanslarSinyalFizyolojik AnlamÖrnekleme (Hz)İşleme YöntemiEDASempatik Aktivasyon4 HzResampling BVPKalp Hacim Nabzı (HRV)64 HzÖzgün Frekans Korundu ACCHareket Şiddeti32 Hz -> 4 HzMagnitude Dönüşümü TEMPCilt Sıcaklığı4 HzZ-Score Normalizasyonu Üç Dallı Hibrit Model MimarisiModel, sinyal heterojenliğini yönetmek için "Late Fusion" stratejisini kullanan üç ayrı koldan oluşur:  Yavaş Kanal (4 Hz): EDA, TEMP ve ACC verilerini 1D-CNN ve LSTM katmanları ile işler.  Hızlı Kanal (64 Hz): BVP sinyalini yüksek çözünürlükte (kernel=7) analiz ederek kardiyovasküler dinamikleri yakalar.  Uzman Kanalı: Sinyallerden çıkarılan 24 boyutlu istatistiksel öznitelik vektörünü modele dahil eder.  Bilimsel Katkılar (Özgün Değer)Bu çalışma literatürde aşağıdaki 5 temel boşluğu doldurmaktadır:  1. EW Etiketleme Çerçevesi: Oransal pencere kuralı (%20 EW eşiği) ile tekrarlanabilir metodoloji.
2. Undefined Label Analizi: Stres öncesi "tanımsız etiket" kısıtlamasının 15 denek üzerinde ilk kez belgelenmesi.
3. Çok Frekanslı Mimari: BVP ve yavaş sinyalleri doğal frekanslarında işleyen özgün tasarım.
4. Kapsamlı Ablasyon: Pencere süresi (15-90s) ve adım boyutu üzerine nicel analizler.
5. Bilek Tabanlı Tahmin: WESAD üzerinde sadece bilek sensörleriyle yapılan ilk kapsamlı EW çalışmaları arasındadır.  💻 Kurulum ve ÇalıştırmaBash# Bağımlılıkları yükleyin
pip install -r requirements.txt

# Veri setini hazırlayın
python dataset_aggregator.py

# Modeli tek seferlik eğitin ve test edin
python train_multimodal_single.py

# 5 tekrarlı istatistiksel analizi çalıştırın
python train_multimodal_loop.py