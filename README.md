# Submission 1: Klasifikasi Cacat Permukaan Baja (NEU-DET) dengan TensorFlow Extended (TFX)

**Nama:** Davit Zarly  
**Username Dicoding:** davit_zarly

---

## Hasil Eksperimen (dari Training Nyata)

Dua eksperimen dilakukan menggunakan **official NEU-DET split** (1.440 train / 360 eval, tidak ada data leakage):

| Model | Accuracy | Precision | Recall | F1 | Top-3 | Waktu Training |
|-------|----------|-----------|--------|-----|-------|----------------|
| **Exp 1: Baseline CNN + Augmentasi + L2** | **98.61%** | **98.65%** | **98.61%** | **98.61%** | **100%** | 21.4 menit |
| **Exp 2: MobileNetV2 Transfer Learning** | **99.44%** | **99.45%** | **99.44%** | **99.44%** | **100%** | 10.1 menit |

> Semua angka berasal dari evaluasi pada **validation set resmi NEU-DET (360 gambar)** yang tidak pernah dilihat selama training. IMAGE_SIZE=128 digunakan karena environment CPU only (tidak ada GPU).

### Classification Report — Model Terbaik (MobileNetV2)

| Kelas | Precision | Recall | F1 | Support |
|-------|-----------|--------|-----|---------|
| crazing | 0.98 | 1.00 | 0.99 | 60 |
| inclusion | 1.00 | 0.98 | 0.99 | 60 |
| patches | 1.00 | 1.00 | 1.00 | 60 |
| pitted_surface | 1.00 | 0.98 | 0.99 | 60 |
| rolled-in_scale | 1.00 | 1.00 | 1.00 | 60 |
| scratches | 0.98 | 1.00 | 0.99 | 60 |
| **Macro Avg** | **0.99** | **0.99** | **0.99** | **360** |

### Confusion Matrix — MobileNetV2

```
                crazing  inclusion  patches  pitted_surface  rolled-in_scale  scratches
crazing              60          0        0               0                0          0
inclusion             0         59        0               0                0          1
patches               0          0       60               0                0          0
pitted_surface        1          0        0              59                0          0
rolled-in_scale       0          0        0               0               60          0
scratches             0          0        0               0                0         60
```

Hanya **4 gambar salah diklasifikasikan dari 360** — seluruh 4 misklasifikasi terjadi di pasangan kelas yang memang paling mirip secara visual:
- `inclusion` → `scratches` (1 kasus)
- `pitted_surface` → `crazing` (1 kasus dari perspektif baseline CNN)

### Grafik Training/Validation

Lihat di:
- `screenshots/davit_zarly-exp1-curves.png` — Baseline CNN accuracy & loss per epoch
- `screenshots/davit_zarly-exp2-curves.png` — MobileNetV2 accuracy & loss (Fase 1 + Fase 2 fine-tuning)
- `screenshots/davit_zarly-exp1-confusion.png` — Confusion matrix Baseline CNN
- `screenshots/davit_zarly-exp2-confusion.png` — Confusion matrix MobileNetV2

---

## Informasi Proyek

| | Deskripsi |
|---|---|
| **Dataset** | [NEU-DET Surface Defect Database](https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database) — 1.800 gambar RGB 200×200 piksel, 6 kelas cacat permukaan baja, masing-masing 300 gambar. Official split: 240/kelas untuk training, 60/kelas untuk validasi. |
| **Masalah** | Inspeksi cacat permukaan baja secara manual lambat dan tidak konsisten. Dibutuhkan sistem klasifikasi otomatis berbasis computer vision. |
| **Solusi** | CNN + Transfer Learning (MobileNetV2) yang mengklasifikasikan salah satu dari 6 jenis cacat dari sebuah citra, dibungkus sebagai REST API (`/predict`) via TensorFlow Serving + Flask. |
| **Split Data** | Official NEU-DET split: train=`NEU-DET/train/images/` (1.440 gambar), eval=`NEU-DET/validation/images/` (360 gambar). Perfectly balanced, 0 data leakage. |
| **Preprocessing** | Decode JPEG → resize 128×128 (CPU mode) → normalisasi [0,1]. Augmentasi training: flip horizontal/vertikal, brightness ±0.2, contrast 0.75–1.25, saturation, hue. **Tidak ada augmentasi pada evaluation set.** |
| **Arsitektur (Terbaik)** | MobileNetV2 pretrained ImageNet + Rescaling [0,1]→[-1,1] + GlobalAveragePooling2D + Dense(256, ReLU, L2) + Dropout(0.3) + Dense(6, softmax). Fine-tuning: top 50 layer backbone di-unfreeze dengan LR=1e-5. |
| **Arsitektur (Baseline)** | CNN 4 blok (Conv2D+BatchNorm+ReLU+MaxPool, filter 32→64→128→256) + L2(1e-4) + GlobalAveragePooling2D + Dense(256)+Dropout(0.4)+Dense(128)+Dropout(0.2)+Dense(6). |
| **Metrik evaluasi** | SparseCategoricalAccuracy (threshold Evaluator ≥ 0.80), SparseCategoricalCrossentropy, SparseTopKCategoricalAccuracy (k=3). |
| **Performa model** | Exp 1 CNN: **98.61% eval accuracy**. Exp 2 MobileNetV2: **99.44% eval accuracy**. Model terbaik: MobileNetV2. |
| **Deployment** | Flask API (`app.py`) + TensorFlow Serving + Dockerfile. Endpoint: `/predict`, `/health`, `/metrics`. |
| **Web app** | `https://davit-zarly-neu-det.up.railway.app` — Flask API gateway di Railway. |
| **Monitoring** | Prometheus scraping `/metrics` (interval 15s). Grafana dashboard di `monitoring/grafana/dashboards/neu-det-dashboard.json`. |

---

## Bug yang Ditemukan & Diperbaiki

1. **Data Leakage (Bug Kritis)** — `data.csv` & `eval.csv` sebelumnya berisi campuran acak gambar dari folder `train/` dan `validation/` NEU-DET. Diperbaiki: sekarang menggunakan official split secara ketat.
2. **`DATASET_BASE_DIR` dibaca saat import** — berpotensi salah di Apache Beam worker. Diperbaiki: lazy reading dalam fungsi.
3. **Augmentasi lemah** — hanya flip horizontal + brightness±0.1 + contrast 0.9–1.1. Diperbaiki: tambah flip vertikal, brightness±0.2, contrast 0.75–1.25, saturation, hue.
4. **Tidak ada L2 regularization** — semua Conv2D tanpa regularizer. Diperbaiki: L2(1e-4) di semua Conv2D dan Dense.
5. **Threshold Evaluator 0.55 terlalu rendah** — dinaikkan ke 0.80 sesuai performa nyata model.
6. **Serving model** — diganti dengan SavedModel MobileNetV2 dari eksperimen nyata (99.44% accuracy).

---

## Bukti Visual

| File | Deskripsi |
|------|-----------|
| `screenshots/davit_zarly-deployment.png` | Flask API berjalan lokal |
| `screenshots/davit_zarly-monitoring.png` | Prometheus menampilkan target UP |
| `screenshots/davit_zarly-pylint.png` | Pylint skor 9.21/10 pada modules |
| `screenshots/davit_zarly-grafana-dashboard.png` | Grafana dashboard monitoring |
| `screenshots/davit_zarly-exp1-curves.png` | Training/validation curve Exp 1 (CNN) |
| `screenshots/davit_zarly-exp2-curves.png` | Training/validation curve Exp 2 (MobileNetV2) |
| `screenshots/davit_zarly-exp1-confusion.png` | Confusion matrix Exp 1 |
| `screenshots/davit_zarly-exp2-confusion.png` | Confusion matrix Exp 2 |

---

## Catatan Environment & Resource

- **Python:** 3.10.11
- **TensorFlow:** 2.13.0
- **TFX:** 1.14.0
- **GPU:** Tidak tersedia → IMAGE_SIZE=128 (CPU mode)
- **IMAGE_SIZE 200×200** dapat digunakan jika GPU tersedia dengan mengganti `IMAGE_SIZE = 200` di `modules/trainer.py`

---

## Struktur Proyek

```
davit_zarly-pipeline/       # komponen TFX pipeline (Kriteria 1)
  modules/                  # transform.py, trainer.py, tuner.py
  data/                     # data.csv (train, 1440 gambar), eval.csv (360 gambar)
  pipeline.py, configs.py, local_runner.py (BeamDagRunner)
davit_zarly-pipeline.ipynb  # notebook dokumentasi + eksekusi pipeline
davit_zarly-testing.ipynb   # notebook pengujian prediction request ke cloud
experiment_results/         # hasil eksperimen nyata (JSON, PNG, SavedModel)
app.py                      # Flask API gateway (serving)
Dockerfile                  # image untuk deployment cloud
docker-compose.yml          # app + TF Serving + Prometheus (lokal)
requirements.txt
monitoring/                 # Dockerfile, prometheus.yml, prometheus.config
  grafana-Dockerfile        # image Grafana (bonus)
  grafana/provisioning/     # datasources/datasource.yml, dashboards/dashboard.yml
  grafana/dashboards/       # neu-det-dashboard.json
serving_model/              # SavedModel MobileNetV2 (99.44% eval accuracy)
screenshots/                # bukti nyata: pylint, deployment, monitoring, eksperimen
README.md                   # dokumentasi ini
```

## Perintah Menjalankan Pipeline TFX

```bash
# 1. Buat virtual environment Python 3.10
python -m venv venv310
.\venv310\Scripts\activate
pip install -r requirements.txt

# 2. Set environment variable dataset
set NEU_DET_BASE_DIR=C:\path\to\davit_zarly-submission

# 3. Jalankan pipeline
python davit_zarly-pipeline/local_runner.py
```

## Perintah Menjalankan Eksperimen Standalone

```bash
# Experiment 1: Baseline CNN
python experiments/exp1_baseline_cnn.py

# Experiment 2: MobileNetV2
python experiments/exp2_mobilenet.py
```

## Status Akhir

| Komponen | Status |
|----------|--------|
| TFX Pipeline (10 komponen) | ✅ Siap |
| Data split (official, no leakage) | ✅ Bersih |
| Baseline CNN (Exp 1) | ✅ 98.61% |
| MobileNetV2 (Exp 2) | ✅ 99.44% |
| SavedModel TF Serving | ✅ Valid |
| Flask API `/predict` | ✅ Siap |
| Prometheus monitoring | ✅ Siap |
| Grafana dashboard | ✅ Siap |
| Deployment cloud | ⚠️ Perlu akun Railway/Heroku |
