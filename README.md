# TA-05 – Klasifikasi Penumpang Titanic dengan Decision Tree (Flask)

Studi kasus: prediksi keselamatan penumpang Titanic (target `survived`) dari `pclass, sex, age, sibsp, parch, fare, embarked`.

## Struktur
- `notebook/TA05_Decision_Tree.ipynb` – EDA, preprocessing, Model A (overfit) vs Model B (pruned), visualisasi, feature importance, simpan model
- `model/model_b.joblib`, `model/meta.json` – model terbaik (max_depth=3) + metadata fitur
- `app.py`, `templates/index.html` – aplikasi Flask
- `reports/` – gambar hasil (untuk laporan)
- `Dockerfile`, `Procfile`, `requirements.txt` – konfigurasi deploy

## Hasil singkat
| Model | Akurasi Train | Akurasi Test | Gap |
|---|---|---|---|
| A (max_depth=None, depth 18) | 97,9% | 76,1% | 21,8 |
| B (max_depth=3) | 83,0% | 79,5% | 3,5 |

## Jalankan lokal
```
pip install -r requirements.txt
python app.py            # http://localhost:5000
```

## Deploy (Vela / platform PaaS lain)
App membaca port dari environment variable `PORT` dan punya endpoint health check `/health`.
- **Opsi Docker:** build dari `Dockerfile` (port 8000 default, ubah lewat env `PORT`).
- **Opsi build-pack/Procfile:** start command `gunicorn app:app --bind 0.0.0.0:$PORT --workers 2`.
- Pastikan folder `model/` ikut ter-push ke GitHub (jangan di-ignore).
