# Perubahan utama
- Fitur bocor dibuang: `Demand Forecast`, `effective_demand`, `stock_to_demand_ratio` (+ `Inventory Level`/`Units Ordered` hari yang sama).
- `backend/features.py`: definisi fitur dipakai bersama training & serving (std ddof=1, is_payday sama).
- Split 70/15/15 (early stopping di val, laporan di test) + baseline pembanding.
- Jika XGBoost tidak mengalahkan baseline, serving otomatis pakai moving average terbaik (lihat `serving_method` di `artifacts/model_spec.json`).
- Pengali skenario manual dihapus; skenario kini mengubah fitur input model.
- Safety stock memakai sigma galat dari residual validasi, `on_order` dipakai, lead time/service level bisa diatur (env / request).
- Artifacts disimpan sebagai JSON (bukan joblib); SKU/toko asing -> -1 + warning; batch predict; CORS dibatasi.

# Cara pakai
pip install -r requirements.txt
python train_model.py            # latih ulang
uvicorn backend.main:app --port 8000   # jalankan dari folder root proyek
