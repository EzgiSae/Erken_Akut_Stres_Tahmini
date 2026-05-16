"""
WESAD Veri Seti - Hibrit CNN-LSTM Çok Modlu Eğitim Modülü (60s Güncel Versiyon)

Mimari Özellikleri (60s Update):
- Branch 1 (Slow): 4 Hz * 60 sn = 240 örnek (EDA, TEMP, ACC)
- Branch 2 (Fast): 64 Hz * 60 sn = 3840 örnek (BVP)
- Branch 3 (Expert): 24 boyutlu öznitelik vektörü
"""

import numpy as np
from keras.models import Model
from keras.layers import (Input, Conv1D, MaxPooling1D, LSTM, Dense, 
                          Dropout, Concatenate, BatchNormalization, 
                          SpatialDropout1D)
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.regularizers import l2
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

def load_final_datasets():
    """Ham sinyal ve öznitelik dosyalarını yükler."""
    # NOT: dataset_aggregator.py'yi 60s için çalıştırdığından emin ol!
    raw_data = np.load("dl_multimodal_dataset.npz")
    feat_data = np.load("multimodal_feature_dataset.npz")
    
    return (
        raw_data["X_train_slow"], raw_data["X_train_fast"], feat_data["X_train"], raw_data["y_train"],
        raw_data["X_test_slow"], raw_data["X_test_fast"], feat_data["X_test"], raw_data["y_test"]
    )

def build_multimodal_model(slow_shape, fast_shape, feat_shape):
    """
    Üç kollu hibrit model mimarisini inşa eder.
    
    Parametreler (60s için güncellendi):
        slow_shape: (240, 3) - EDA, TEMP, ACC (4 Hz * 60 sn)
        fast_shape: (3840, 1) - BVP (64 Hz * 60 sn)
        feat_shape: (24,) - İstatistiksel öznitelikler
    """
    reg = l2(0.0005) 

    # --- KOL 1: Yavaş Kanallar (Slow Signals - 4 Hz) ---
    input_slow = Input(shape=slow_shape, name="Slow_Input")
    s = Conv1D(32, 3, activation='relu', padding='same', kernel_regularizer=reg)(input_slow)
    s = BatchNormalization()(s)
    s = SpatialDropout1D(0.3)(s)
    s = MaxPooling1D(2)(s)
    s = LSTM(32, dropout=0.3, recurrent_dropout=0.3)(s)

    # --- KOL 2: Hızlı Kanal (Fast Signal - BVP 64 Hz) ---
    input_fast = Input(shape=fast_shape, name="Fast_Input")
    f = Conv1D(64, 7, activation='relu', padding='same', kernel_regularizer=reg)(input_fast)
    f = BatchNormalization()(f)
    f = SpatialDropout1D(0.3)(f)
    f = MaxPooling1D(4)(f)
    f = Conv1D(128, 5, activation='relu', padding='same', kernel_regularizer=reg)(f)
    f = MaxPooling1D(4)(f)
    f = LSTM(64, dropout=0.3)(f)

    # --- KOL 3: Uzman Öznitelikler (Expert Features - 24D) ---
    input_feat = Input(shape=feat_shape, name="Feat_Input")
    feat = Dense(64, activation='relu', kernel_regularizer=reg)(input_feat)
    feat = BatchNormalization()(feat)
    feat = Dropout(0.4)(feat)

    # --- Öznitelik Füzyonu ve Karar Katmanları ---
    combined = Concatenate()([s, f, feat])
    x = Dense(128, activation='relu', kernel_regularizer=reg)(combined)
    x = Dropout(0.5)(x)
    x = Dense(32, activation='relu')(x)
    output = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=[input_slow, input_fast, input_feat], outputs=output)
    model.compile(
        optimizer=Adam(learning_rate=0.0002), 
        loss='binary_crossentropy', 
        metrics=['accuracy']
    )
    return model

def main():
    # 1. Veri Yükleme ve Hazırlık
    try:
        (X_tr_slow, X_tr_fast, X_tr_feat, y_train, 
         X_te_slow, X_te_fast, X_te_feat, y_test) = load_final_datasets()
    except KeyError as e:
        print(f"Hata: .npz dosyalarında beklenen anahtarlar bulunamadı: {e}")
        return

    # Öznitelik vektörlerini normalize et
    scaler = StandardScaler()
    X_tr_feat = scaler.fit_transform(X_tr_feat)
    X_te_feat = scaler.transform(X_te_feat)

    # 2. Model Kurulumu (60s parametreleri buraya işlendi)
    # Girdi boyutları: Slow (240, 3), Fast (3840, 1), Feat (24,)
    model = build_multimodal_model((240, 3), (3840, 1), (24,))
    
    # 3. Eğitim Yapılandırması
    early_stop = EarlyStopping(
        monitor='val_loss', 
        patience=7, 
        restore_best_weights=True
    )

    print("\n--- Çok Modlu Hibrit Model Eğitimi Başlıyor (60s Window) ---")
    
    history = model.fit(
        x=[X_tr_slow, X_tr_fast, X_tr_feat],
        y=y_train,
        validation_split=0.2,
        epochs=50,
        batch_size=32,
        class_weight={0: 2.5, 1: 1.0},
        callbacks=[early_stop],
        verbose=1
    )

    # 4. Performans Değerlendirme
    print("\n--- Nihai Test Performansı (Eşik: 0.5) ---")
    y_prob = model.predict([X_te_slow, X_te_fast, X_te_feat])
    y_pred = (y_prob > 0.5).astype(int)
    
    print(classification_report(
        y_test, y_pred, 
        target_names=["Erken Uyari (EW)", "Akut Stres"], 
        zero_division=0
    ))
    
    print("Karmaşıklık Matrisi (Confusion Matrix):")
    print(confusion_matrix(y_test, y_pred))

if __name__ == "__main__":
    main()