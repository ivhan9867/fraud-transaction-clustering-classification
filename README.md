# Financial Transaction Fraud Analysis Clustering & Classification

Capstone project untuk kelas **[Belajar Machine Learning untuk Pemula](https://www.dicoding.com/)** di Dicoding.
Membangun end-to-end machine learning pipeline pada dataset transaksi keuangan untuk mengelompokkan pola transaksi (clustering) dan memprediksi kelompok tersebut (classification).

## 📌 Latar Belakang

Dataset ini berisi **2.512 transaksi keuangan** dengan atribut seperti nominal, lokasi, metode pembayaran, jenis transaksi, dan waktu transaksi   cocok untuk eksplorasi deteksi anomali/fraud pada aktivitas keuangan.

Karena dataset tidak berlabel, project ini dibagi menjadi dua tahap:
1. **Clustering**   mengelompokkan transaksi ke dalam segmen/pola tertentu tanpa label (unsupervised).
2. **Classification**   menggunakan hasil clustering sebagai label (`Target`), lalu melatih model supervised untuk memprediksi segmen tersebut pada data baru.

## 🛠️ Tech Stack

- Python, pandas, numpy
- scikit-learn (KMeans, DecisionTreeClassifier, RandomForestClassifier, GridSearchCV)
- yellowbrick (KElbowVisualizer)
- matplotlib, seaborn
- joblib (model persistence)

## 🔍 Tahap 1   Clustering

Notebook: [`notebooks/1_clustering.ipynb`](notebooks/1_clustering.ipynb)

- EDA: struktur data, missing value, statistik deskriptif, matriks korelasi, distribusi tiap fitur
- Preprocessing: pembersihan data, drop kolom identifier (TransactionID, AccountID, dll), encoding fitur kategorikal, scaling fitur numerik, binning
- Penentuan jumlah cluster optimal dengan **Elbow Method** (`KElbowVisualizer`)
- Model: **K-Means Clustering**, dibandingkan dengan reduksi dimensi **PCA**
- **Silhouette Score: 0.57**
- Interpretasi tiap cluster berdasarkan karakteristik rata-rata/min/max fitur, hasil di-*inverse transform* kembali ke skala aslinya untuk dianalisis
- Output: `data/data_clustering_inverse.csv` (data asli + kolom `Target` hasil clustering)

## 🌳 Tahap 2   Classification

Notebook: [`notebooks/2_classification.ipynb`](notebooks/2_classification.ipynb)

- Split data (`train_test_split`) menggunakan kolom `Target` dari hasil clustering
- Model utama: **Decision Tree Classifier**
- Model pembanding: **Random Forest Classifier**
- **Hyperparameter tuning** dengan GridSearchCV pada model terbaik
- Evaluasi dengan accuracy, precision, recall, dan F1-score untuk seluruh model

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| Decision Tree | 1.00 | 1.00 | 1.00 | 1.00 |
| Tuned Model | 1.00 | 1.00 | 1.00 | 1.00 |

> Catatan: akurasi mendekati sempurna karena label (`Target`) berasal dari hasil clustering pada tahap sebelumnya, sehingga pola antar cluster relatif mudah dipisahkan oleh model klasifikasi.

## 📁 Struktur Repo

```
├── notebooks/
│   ├── 1_clustering.ipynb
│   └── 2_classification.ipynb
├── models/              # model hasil training (.h5 via joblib)
├── data/                # dataset hasil preprocessing & clustering
└── README.md
```

## 🚀 Cara Menjalankan

```bash
pip install pandas numpy scikit-learn yellowbrick matplotlib seaborn joblib
jupyter notebook notebooks/1_clustering.ipynb
```

Jalankan notebook clustering terlebih dahulu untuk menghasilkan `data_clustering_inverse.csv`, baru lanjut ke notebook classification.

## 👤 Author

**Ivhan Afika Prila**   capstone project ini dikerjakan sebagai submission akhir kelas Dicoding "Belajar Machine Learning untuk Pemula".
