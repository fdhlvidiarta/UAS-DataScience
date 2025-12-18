# 📘 Servo Motor Performance Prediction

Proyek Machine Learning dan Deep Learning untuk memprediksi performa sistem servo motor berdasarkan parameter kontrol menggunakan pendekatan regresi.

## 👤 Informasi

- **Nama:** Fadhil Vidiarta
- **NIM:** 233307047
- **Repo:** https://github.com/fdhlvidiarta/UAS-DataScience-.git
- **Video:** https://youtu.be/8iN2i0yBS8s

---

## 1. 🎯 Ringkasan Proyek

Proyek ini bertujuan untuk memprediksi performa sistem servo motor berdasarkan parameter kontrol menggunakan dataset Servo dari UCI Machine Learning Repository. Tahapan yang dilakukan:

- Melakukan Exploratory Data Analysis (EDA)
- Melakukan data preparation (endcoding & scaling)
- Pembangun 3 model Regresi: **Decision Tree (Baseline)**, **Random Forest (Advanced)**, **Neural Network / MLP (Deep Learning)**
- Evaluasi model menggunakan RMSE dan R² Score
- Menentukan model terbaik berdasarkan performa

---

## 2. 📄 Problem & Goals

**Problem Statements:**

1. Bagaimana memprediksi performa sistem servo motor berdasarkan parameter kontrol?
2. Apakah model machine learning non-linear lebih baik dibanding baseline?
3. Apakah deep learning efektif untuk data tabular berukuran kecil?

**Goals:**

1. Membangun model regresi performa servo motor
2. Membandingkan performa baseline, advanced, dan deep learning
3. Menentukan model terbaik berdasarkan metrik evaluasi
4. Menghasilkan sistem yang reproducible

---

## 📁 Struktur Folder

```
zoo-classification/
│
├── data/                   # Dataset
│   └── servo.data          # Data servo
│
├── notebooks/              # Jupyter notebooks
│   └── ML_Project.ipynb    # Notebook utama proyek
│
├── src/                    # Source code
│
├── models/                 # Saved models
│   ├── model_baseline.pkl  # Decision Tree model
│   ├── model_rf.pkl        # Random Forest model
│   └── model_mlp.h5        # Neural Network model
│
├── images/
│   ├── eda_boxplot_fitur.png
│   ├── eda_distribusi_target.png
│   ├── eda_korelasi.png
│   └── training_loss_mlp.png
│
├── requirements.txt        # Dependencies
├── .gitignore
└── README.md
```

---

## 3. 📊 Dataset

- **Sumber:** [UCI Machine Learning Repository - servo Dataset](https://archive.ics.uci.edu/ml/datasets/servo)
- **Jumlah Data:** 167 instances
- **Jumlah Fitur:** 5
- **Tipe:** Tabular (Regresi)
- **Target:** Performa sistem servo

### Fitur Utama

| Nama Fitur | Tipe Data   | Deskripsi                      |
| ---------- | ----------- | ------------------------------ |
| motor      | Kategorikal | Jenis motor servo              |
| screw      | Kategorikal | Jenis screw                    |
| pgain      | Numerik     | Proportional gain              |
| vgain      | Numerik     | Velocity gain                  |
| kelas      | Numerik     | Performa sistem servo (target) |

---

## 4. 🔧 Data Preparation

### 4.1 Data Cleaning

- Tidak ada missing values
- Tidak ada duplicate data

### 4.2 Feature Engineering

- One-Hot Encoding untuk fitur kategorikal
- StandardScaler untuk fitur numerik

### 4.3 Data Splitting

- Training set: 80%
- Test set: 20%
- Random state: 42

---

## 5. 🤖 Modeling

### Model 1 – Baseline: Decision Tree

- Model sederhana untuk pembanding
- Mampu menangkap hubungan non-linear dasar
- Digunakan sebagai pembanding awal

### Model 2 – Advanced: Random Forest

- Ensemble learning
- Lebih stabil dan akurat
- Cocok untuk data tabular kecil–menengah

### Model 3 – Deep Learning: Neural Network (MLP)

- Multilayer Perceptron
- 2 hidden layer + Dropout
- Early stopping untuk mencegah overfitting

---

## 6. 🧪 Evaluation

**Metrik:** RMSE dan R² Score

### Hasil Perbandingan Model

| Model                    | RSME   | R2     | Training Time |
| ------------------------ | ------ | ------ | ------------- |
| Baseline (Decision) Tree | 0.8005 | 0.7351 | < 1 detik     |
| Advanced (Random Forest) | 0.5801 | 0.8609 | ~ 2 detik     |
| Deep Learning (MLP)      | 0.6395 | 0.8309 | ~ 15 detik    |

---

## 7. 🏁 Kesimpulan

### Model Terbaik: Random Forest Regressor

Berdasarkan hasil evaluasi menggunakan metrik RMSE dan R² Score, model Random Forest Regressor menunjukkan performa terbaik dalam memprediksi performa sistem servo motor dibandingkan model lainnya.
**Random Forest**:

- Random Forest memiliki RMSE terendah (0.5801) dan R² tertinggi (0.8609)

### Alasan:

1. Random Forest mampu menangkap hubungan non-linear yang kompleks antar parameter kontrol servo
2. Memberikan keseimbangan terbaik antara akurasi, stabilitas, dan waktu training
3. Lebih efektif dibanding deep learning pada dataset tabular berukuran kecil
4. Tidak menunjukkan indikasi overfitting yang signifikan

### Key Insights:

- Parameter pgain dan vgain merupakan faktor paling berpengaruh terhadap performa sistem servo
- Model ensemble (Random Forest) secara konsisten mengungguli model baseline
- Deep learning tidak selalu memberikan performa terbaik pada data tabular kecil
- Pendekatan machine learning klasik masih sangat relevan untuk sistem kontrol berbasis data terstruktur

---

## 8. 🔮 Future Work

- [x] Mengumpulkan lebih banyak data servo
- [x] Menambah variasi kondisi dan parameter servo
- [x] Hyperparameter tuning lebih ekstensif (Grid Search / Random Search)
- [x] Eksperimen arsitektur Deep Learning yang lebih kompleks
- [x] Perbandingan MLP dengan dan tanpa preprocessing manual
- [ ] Deployment ke web application (Streamlit/Gradio)
- [ ] Membuat API dengan Flask/FastAPI
- [ ] Model explainability dengan SHAP/LIME

---

## 9. 🔁 Reproducibility

### Instalasi Dependencies

```bash
python -m venv venv
venv\Scripts\activate
source venv/bin/activate
pip install -r requirements.txt

```

### Menjalankan Project

```bash

git clone https://github.com/fdhlvidiarta/UAS-DataScience-.git
cd UAS-DataScience-

# Download dataset servo.data dari UCI Machine Learning Repository
# Letakkan file servo.data ke dalam folder:
data/servo.data

# Jalankan Jupyter Notebook
jupyter notebook notebooks/ML_Project.ipynb
```

### Google Colab

1. Upload `ML_Project.ipynb` ke Google Colab
2. Upload `servo.data` ke Colab atau mount Google Drive
3. Install dependencies: `!pip install pandas numpy scikit-learn matplotlib seaborn tensorflow`
4. Run all cells

---

## 📚 Referensi

- UCI Machine Learning Repository: Servo Dataset
- Quinlan, J. R. (1993). Servo Dataset. UCI Machine Learning Repository
- Scikit-learn Documentation
- TensorFlow/Keras Documentation

---
