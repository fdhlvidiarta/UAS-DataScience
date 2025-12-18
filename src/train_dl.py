import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import mean_squared_error, r2_score
from preprocess import load_dan_pra_proses_data


def main():
    # Load dan preprocessing data
    X_train, X_test, y_train, y_test, preprocessor = load_dan_pra_proses_data(
        "data/servo.data"
    )

    # Transform data (fit hanya pada data latih)
    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)

    # Arsitektur Neural Network (MLP)
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(
            64,
            activation="relu",
            input_shape=(X_train.shape[1],)
        ),
        tf.keras.layers.Dropout(0.2),  # Regularisasi
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(1)  # Output regresi
    ])

    model.compile(
        optimizer="adam",
        loss="mean_squared_error",
        metrics=["mean_absolute_error"]
    )

    print("\n=== PROSES TRAINING MODEL DEEP LEARNING (MLP) ===")
    print("Keterangan:")
    print("- Epoch : 1 kali seluruh data latih diproses oleh model")
    print("- Loss  : Nilai kesalahan prediksi model\n")

    # Callback Early Stopping
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )

    # Training model
    history = model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=50,
        batch_size=16,
        callbacks=[early_stop],
        verbose=1
    )

    # Prediksi data uji
    y_pred = model.predict(X_test).flatten()

    # Evaluasi model
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print("\n=== HASIL EVALUASI MODEL DEEP LEARNING ===")
    print(f"RMSE : {rmse:.4f}")
    print(f"R²   : {r2:.4f}")

    # Simpan model
    os.makedirs("models", exist_ok=True)
    model.save("models/model_mlp.h5")
    print("Model deep learning berhasil disimpan di models/model_mlp.h5")

    # Visualisasi loss
    os.makedirs("images", exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="Loss Data Latih")
    plt.plot(history.history["val_loss"], label="Loss Data Validasi")
    plt.xlabel("Epoch (Iterasi Pelatihan)")
    plt.ylabel("Nilai Loss")
    plt.title("Perubahan Nilai Loss Selama Proses Training Model MLP")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("images/training_loss_mlp.png")
    plt.show()

    print("Grafik loss disimpan di images/training_loss_mlp.png")


if __name__ == "__main__":
    main()
