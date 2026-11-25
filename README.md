\# SmartStock AI

\### AI-Based Demand Forecasting and Inventory Management System



SmartStock AI adalah aplikasi berbasis Artificial Intelligence (AI) untuk membantu proses peramalan permintaan produk (\*demand forecasting\*) dan pengelolaan persediaan (\*inventory management\*).



Aplikasi ini menggunakan algoritma \*\*XGBoost Regressor\*\* untuk memprediksi kebutuhan produk berdasarkan histori permintaan, karakteristik produk, informasi kalender, dan variabel skenario.



Tujuan utama proyek ini adalah membantu pengambilan keputusan persediaan agar lebih terukur, mengurangi risiko kekurangan stok (\*stockout\*), dan mendukung perencanaan pemesanan barang.



\---



\## Key Features



\- \*\*Demand Forecasting\*\* — Memprediksi permintaan produk menggunakan model Machine Learning.

\- \*\*Multiple Model Versions\*\* — Menyediakan model forecasting V1 dan V2.

\- \*\*Scenario Simulation\*\* — Menguji perubahan skenario promosi, diskon, cuaca, dan epidemi.

\- \*\*Inventory Management\*\* — Menghitung Safety Stock, Reorder Point, dan Suggested Order Quantity.

\- \*\*Dashboard Visualization\*\* — Menampilkan hasil prediksi dan informasi persediaan.

\- \*\*Multi-Store Support\*\* — Mendukung data dari beberapa cabang toko.

\- \*\*CSV Data Upload\*\* — Memproses dataset melalui file CSV.



\## Technology Stack



| Component | Technology |

|---|---|

| Programming Language | Python, JavaScript |

| Machine Learning | XGBoost |

| Data Processing | Pandas, NumPy |

| Backend | FastAPI |

| Frontend | HTML, CSS, JavaScript |

| Data Visualization | Chart.js |

| API Testing | FastAPI Swagger UI |

| Version Control | Git, GitHub |



\## Machine Learning Model



SmartStock AI memiliki dua versi model forecasting:



\### Model V1

\- Menggunakan XGBoost Regressor.

\- Menggunakan fitur historis dan karakteristik produk.

\- Berfungsi sebagai model awal (\*baseline implementation\*).



\### Model V2

\- Menggunakan XGBoost Regressor.

\- Menggunakan 21 fitur untuk proses prediksi.

\- Menerapkan fitur lag dan rolling statistics.

\- Menggunakan temporal train-validation-test split.

\- Mendukung simulasi skenario eksternal.



\### Model Evaluation



Hasil evaluasi model V2 pada test set:



| Metric | Result |

|---|---:|

| MAE | 25.75 |

| RMSE | 33.36 |

| WMAPE | 25.72% |

| Best Baseline WMAPE | 31.44% |



Model V2 menghasilkan WMAPE lebih rendah dibandingkan baseline moving average 7 hari.



\*\*Evaluation note:\*\* Evaluasi ini menggunakan fitur eksternal yang diamati pada periode test. Hasil tersebut merupakan evaluasi bersyarat dan belum tentu mencerminkan akurasi operasional ketika seluruh fitur masa depan belum diketahui.



\## Inventory Management



Sistem menyediakan beberapa indikator untuk membantu pengelolaan stok:



\- \*\*Safety Stock\*\* — Persediaan pengaman untuk mengantisipasi ketidakpastian permintaan.

\- \*\*Reorder Point (ROP)\*\* — Batas persediaan yang menjadi indikator untuk melakukan pemesanan ulang.

\- \*\*Suggested Order Quantity\*\* — Rekomendasi jumlah barang yang perlu dipesan berdasarkan perhitungan sistem.



\## Project Structure



```text

SmartStock-AI/

│

├── artifacts/

│   ├── encoders.json

│   ├── model\_spec.json

│   └── xgboost\_stock\_model.json

│

├── artifacts\_v2/

│   ├── encoders\_v2.json

│   ├── model\_spec\_v2.json

│   └── xgboost\_stock\_model\_v2.json

│

├── backend/

│   ├── \_\_init\_\_.py

│   ├── features.py

│   ├── features\_v2.py

│   ├── inventory\_policy.py

│   ├── main.py

│   └── main\_v2.py

│

├── css/

│   └── styles.css

│

├── dataset/

│   ├── retail\_store\_inventory.csv

│   └── sales\_data.csv

│

├── js/

│   ├── api.js

│   ├── app.js

│   ├── chart-config.js

│   └── table.js

│

├── artifacts/

├── artifacts\_v2/

├── index.html

├── requirements.txt

├── train\_model.py

├── train\_model\_v2.py

└── README.md

```



\## Installation and Setup



\### 1. Clone Repository



```bash

git clone https://github.com/GandhiArta/SmartStock-AI.git

cd SmartStock-AI

```



\### 2. Create Virtual Environment



```bash

python -m venv venv

```



Activate the environment on Windows:



```bash

venv\\Scripts\\activate

```



\### 3. Install Dependencies



```bash

pip install -r requirements.txt

```



\### 4. Run Backend V1



```bash

python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000

```



API documentation:



http://127.0.0.1:8000/docs



\### 5. Run Backend V2



Open another terminal in the project directory, activate the virtual environment, then run:



```bash

python -m uvicorn backend.main\_v2:app --host 127.0.0.1 --port 8001

```



API documentation:



http://127.0.0.1:8001/docs



\### 6. Run Frontend



Open `index.html` using a local development server, such as the Live Server extension in Visual Studio Code.



The frontend should connect to the backend running on the configured API address.



\## Dataset



The project includes CSV datasets containing information such as:



\- Date

\- Store ID

\- Product ID

\- Category

\- Region

\- Inventory Level

\- Units Sold

\- Units Ordered

\- Price

\- Discount

\- Weather Condition

\- Promotion

\- Competitor Pricing

\- Seasonality

\- Epidemic

\- Demand



The forecasting target used by Model V2 is \*\*Demand\*\*.



\## Units Sold vs Demand



These two variables have different meanings:



\- \*\*Units Sold:\*\* The number of product units actually sold.

\- \*\*Demand:\*\* The number of product units requested or needed by customers.



When available inventory is insufficient, actual sales may be lower than customer demand. Therefore, forecasting Demand can help a business anticipate inventory requirements rather than relying only on historical sales.



\## Scenario Simulation



Model V2 supports scenario inputs such as:



\- Promotion

\- Discount

\- Weather Condition

\- Epidemic



These inputs allow users to compare forecasting outputs under different assumptions.



Scenario results represent model predictions under the supplied inputs, not proof of causal relationships.



\## Project Status



\- \[x] Machine Learning Model V1

\- \[x] Machine Learning Model V2

\- \[x] Backend API V1

\- \[x] Backend API V2

\- \[x] CSV Upload

\- \[x] Scenario Recalculation

\- \[x] Inventory Policy Calculation

\- \[x] Frontend Integration

\- \[x] Scenario Testing

\- \[x] GitHub Repository



\## Repository



GitHub: https://github.com/GandhiArta/SmartStock-AI



\---



\*\*SmartStock AI — Smarter Demand Forecasting, Better Inventory Decisions.\*\*

