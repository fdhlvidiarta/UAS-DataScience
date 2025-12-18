import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer


def load_dan_pra_proses_data(path_data):

    # Dataset SERVO tidak memiliki header bawaan
    kolom = ["motor", "screw", "pgain", "vgain", "kelas"]
    df = pd.read_csv(path_data, header=None, names=kolom)

    # fitur (X) dan target (y)
    X = df.drop("kelas", axis=1)
    y = df["kelas"]

    fitur_kategorikal = ["motor", "screw"]
    fitur_numerik = ["pgain", "vgain"]

    # Pipeline preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ("kategori", OneHotEncoder(handle_unknown="ignore"), fitur_kategorikal),
            ("numerik", StandardScaler(), fitur_numerik),
        ]
    )

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test, preprocessor
