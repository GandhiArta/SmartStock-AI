# SmartStock AI V2

An AI-powered inventory management system that predicts product demand and provides inventory recommendations to support better stock planning and reduce inventory inefficiencies.

## Features

- **Demand Forecasting** – Predict product demand using an XGBoost machine learning model.
- **Inventory Recommendations** – Generate inventory recommendations based on predicted demand and inventory policies.
- **Sales Data Upload** – Upload sales data through the application.
- **Dynamic Recalculation** – Recalculate forecasts and inventory recommendations when new data is uploaded.
- **Interactive Dashboard** – View demand forecasts and inventory information through a web interface.

## Tech Stack

**Backend**
- Python
- FastAPI
- XGBoost
- Pandas
- NumPy

**Frontend**
- HTML
- CSS
- JavaScript

## Project Structure

```text
smartstock-frontend/
├── artifacts_v2/
│   ├── encoders_v2.json
│   ├── model_spec_v2.json
│   └── xgboost_stock_model_v2.json
├── backend/
│   ├── __init__.py
│   ├── features_v2.py
│   ├── inventory_policy.py
│   └── main_v2.py
├── css/
│   └── styles.css
├── dataset/
│   └── sales_data.csv
├── js/
│   ├── api.js
│   ├── app.js
│   ├── chart-config.js
│   └── table.js
├── .gitignore
├── README.md
├── index.html
├── requirements.txt
└── train_model_v2.py
```

## Requirements

- Python 3.9 or newer
- pip

## Installation

1. Clone the repository:

```bash
git clone YOUR_REPOSITORY_URL
cd smartstock-frontend
```

2. Create a virtual environment:

```bash
python -m venv venv
```

3. Activate the virtual environment.

Windows CMD:

```cmd
venv\Scripts\activate
```

4. Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the Application

### 1. Start the Backend

Run the following command from the project root directory:

```bash
uvicorn backend.main_v2:app --reload --port 8001
```

The backend API will be available at:

```text
http://127.0.0.1:8001
```

API documentation:

```text
http://127.0.0.1:8001/docs
```

### 2. Open the Frontend

Open `index.html` in your browser using a local development server.

Make sure the backend is running before using the application.

## Model Training

The project uses the V2 XGBoost model.

To train the model again, run:

```bash
python train_model_v2.py
```

The model artifacts are stored in the `artifacts_v2/` directory.

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/model-info` | Retrieve model information |
| POST | `/api/upload-csv` | Upload sales data |
| POST | `/api/recalculate` | Recalculate forecasts and inventory recommendations |

## Production Model

SmartStock AI currently uses **V2** as its production model.

The model and its associated preprocessing artifacts are stored in `artifacts_v2/`.

## License

No license has been specified yet.