import numpy as np
import os
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from preprocess import load_dan_pra_proses_data


def main():
    # Load dan preprocessing data
    X_train, X_test, y_train, y_test, preprocessor = load_dan_pra_proses_data(
        "data/servo.data"
    )

    # Model Advanced Machine Learning: Random Forest Regressor
    model = Pipeline(
        steps=[
            ("pra_proses", preprocessor),
            ("model", RandomForestRegressor(
                n_estimators=200,
                max_depth=None,
                random_state=42
            )),
        ]
    )

    print("=== PROSES TRAINING MODEL ADVANCED MACHINE LEARNING ===")
    print("Model     : Random Forest Regressor")
    print("Keterangan: Ensemble learning untuk menangkap hubungan non-linear\n")

    # Training model
    model.fit(X_train, y_train)

    # Prediksi data uji
    y_pred = model.predict(X_test)

    # Evaluasi model
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print("=== HASIL EVALUASI MODEL ADVANCED ML ===")
    print(f"RMSE : {rmse:.4f}")
    print(f"R²   : {r2:.4f}")

    # Simpan model
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/model_rf.pkl")
    print("Model Random Forest berhasil disimpan di models/model_rf.pkl")


if __name__ == "__main__":
    main()
