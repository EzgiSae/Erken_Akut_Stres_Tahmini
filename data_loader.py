"""
Bu dosya, WESAD veri setindeki subject bazlı .pkl dosyalarını yükler.

Kullanılan bilek sensörleri:
- EDA  : Electrodermal Activity, 4 Hz
- BVP  : Blood Volume Pulse, 64 Hz
- TEMP : Skin Temperature, 4 Hz
- ACC  : 3 eksenli ivmeölçer, 32 Hz

Label verisi WESAD içinde 700 Hz örnekleme frekansı ile tutulmaktadır.
"""

import pickle
import numpy as np


def load_wesad_subject(subject_id):
    """
    Belirtilen WESAD subject verisini yükler.

    Parameters
    ----------
    subject_id : str
        Örneğin: "S2", "S3", "S15"

    Returns
    -------
    wrist_data_clean : dict
        Bilek sensör verilerini içerir:
        {
            "EDA":  np.array,
            "TEMP": np.array,
            "BVP":  np.array,
            "ACC":  np.array, shape=(N, 3)
        }

    labels : np.array
        WESAD label dizisi. Örnekleme frekansı 700 Hz'dir.

    fs_dict : dict
        Her sinyalin örnekleme frekansını içerir.
    """

    path = f"data/{subject_id}/{subject_id}.pkl"

    with open(path, "rb") as f:
        data = pickle.load(f, encoding="latin1")

    wrist_data = data["signal"]["wrist"]
    labels = np.array(data["label"]).flatten()

    fs_dict = {
        "EDA": 4,
        "TEMP": 4,
        "BVP": 64,
        "ACC": 32,
        "Label": 700
    }

    wrist_data_clean = {
        "EDA": wrist_data["EDA"].flatten(),
        "TEMP": wrist_data["TEMP"].flatten(),
        "BVP": wrist_data["BVP"].flatten(),
        "ACC": wrist_data["ACC"]
    }

    return wrist_data_clean, labels, fs_dict


def debug_subject(subject_id="S2"):
    """
    Veri yükleme işlemini kontrol etmek için debug fonksiyonu.
    Subject'e ait sinyal uzunluklarını, sampling rate bilgilerini ve label değerlerini yazdırır. 
    """

    wrist, labels, fs = load_wesad_subject(subject_id)

    print(f"\n--- {subject_id} DEBUG ---")

    print("\nSampling Rates:")
    print(fs)

    print("\nSignal Lengths:")
    for key, value in wrist.items():
        print(f"{key}: {len(value)}")

    print(f"\nLabels Length: {len(labels)}")

    print("\nUnique Labels:")
    print(np.unique(labels))


if __name__ == "__main__":
    debug_subject("S2")