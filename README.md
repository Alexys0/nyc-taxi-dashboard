# Dashboard NYC Yellow Taxi 2025 (Final Project)

Dashboard Streamlit untuk Final Project Big Data Analytics: permintaan Yellow Taxi New York City 2025,
prediksi pickup satu jam ke depan (scikit-learn dan Spark MLlib), dan jaringan mobilitas antarzona (graph).

Halaman:

1. **Ringkasan**: volume trip, peta pickup per zona, pola hari × jam.
2. **Prediksi Demand**: akurasi model vs baseline, eksplorasi aktual vs prediksi per zona, error per tingkat keramaian dan per hari.
3. **Jaringan Mobilitas**: arus bersih, PageRank, komunitas, koridor terbesar, tujuan trip bandara, pergeseran weekend.
4. **Insight Bisnis**: empat temuan dengan rekomendasi.

## Sumber data

Semua file di `data/` dibuat oleh `prepare_dashboard_data.py` dari satu run pipeline yang terverifikasi
(nama run ada di `data/meta.json`). Angkanya sama dengan laporan. Script ini memeriksa bahwa total trip dan
MAE cocok dengan hasil run sebelum menulis file. Script hanya bisa dijalankan di dalam folder proyek lengkap;
di repo dashboard, `data/` sudah tersedia.

## Menjalankan lokal

```
pip install -r requirements.txt
streamlit run app.py
```

Di proyek BDA (Windows), dari root proyek: `.venv-dashboard\Scripts\streamlit.exe run SourceCode\dashboard\app.py`.

## Deploy ke Streamlit Community Cloud

1. Buat repo GitHub publik baru, lalu upload isi folder ini (`app.py`, `requirements.txt`, `data/`,
   `.streamlit/`, `README.md`, `prepare_dashboard_data.py`) ke root repo.
2. Buka https://share.streamlit.io, login dengan GitHub, pilih **Create app**, lalu pilih repo, branch `main`,
   dan file `app.py`.
3. Klik **Deploy**. Link `https://<nama-app>.streamlit.app` bisa dibagikan ke dosen.
