import numpy as np
import os
import joblib
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from preprocess import load_dan_pra_proses_data


def main():
    # Load dan preprocessing data
    X_train, X_test, y_train, y_test, preprocessor = load_dan_pra_proses_data(
        "data/servo.data"
    )

    # Model baseline: Decision Tree Regressor (non-linear sederhana)
    model = Pipeline(
        steps=[
            ("pra_proses", preprocessor),
            ("model", DecisionTreeRegressor(
                max_depth=5,
                random_state=42
            )),
        ]
    )

    # Training model
    model.fit(X_train, y_train)

    # Prediksi data uji
    y_pred = model.predict(X_test)

    # Evaluasi model
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print("=== HASIL MODEL BASELINE (Decision Tree Regressor) ===")
    print("Model ini digunakan sebagai pembanding non-linear sederhana")
    print(f"RMSE : {rmse:.4f}")
    print(f"R²   : {r2:.4f}")

    # Simpan model
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/model_baseline.pkl")
    print("Model baseline berhasil disimpan di models/model_baseline.pkl")


if __name__ == "__main__":
    main()
