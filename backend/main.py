
import io
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import xgboost as xgb
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .features import FEATURES, Encoders, calendar_features, history_features
from .inventory_policy import calculate_reorder_policy


app = FastAPI(title="SmartStock AI API")

_default_origins = ",".join(
    f"http://{h}:{p}"
    for h in ("127.0.0.1", "localhost")
    for p in (3000, 5173, 5500, 8000, 8080)
)

ALLOWED_ORIGINS = [
    o.strip()
    for o in os.getenv("CORS_ORIGINS", _default_origins).split(",")
    if o.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"
HORIZON = 14

DEFAULT_LEAD_TIME = int(os.getenv("LEAD_TIME_DAYS", "3"))
DEFAULT_REVIEW_PERIOD = int(os.getenv("REVIEW_PERIOD_DAYS", "0"))
DEFAULT_SERVICE_LEVEL = float(os.getenv("SERVICE_LEVEL", "0.95"))

# Load artifacts
try:
    model_spec = json.loads(
        (ARTIFACTS_DIR / "model_spec.json").read_text(encoding="utf-8")
    )

    encoders = Encoders(
        json.loads(
            (ARTIFACTS_DIR / "encoders.json").read_text(encoding="utf-8")
        )
    )

    model = xgb.XGBRegressor()
    model.load_model(ARTIFACTS_DIR / "xgboost_stock_model.json")

except Exception as e:
    raise RuntimeError(
        f"Artifacts tidak bisa dimuat ({e}). "
        "Jalankan python train_model.py terlebih dahulu."
    ) from e


if model_spec.get("features") != FEATURES:
    raise RuntimeError(
        "Daftar fitur model_spec.json berbeda dengan backend/features.py. "
        "Latih ulang model."
    )

SERVING_METHOD = model_spec.get("serving_method", "xgboost")
RESID_STD = model_spec.get("residual_std", {})
RESID_STD_GLOBAL = float(model_spec.get("residual_std_global", 1.0))

print(f"Artifacts dimuat. Metode serving: {SERVING_METHOD}")


def model_info() -> Dict[str, Any]:
    metrics = model_spec.get("metrics_test", {})

    return {
        "serving_method": SERVING_METHOD,
        "wmape_xgboost": metrics.get("xgboost"),
        "wmape_best_baseline": model_spec.get("best_baseline_wmape"),
        "beats_baseline": model_spec.get("beats_baseline", False),
        "scenario_effective": False,
        "target": "Demand",
    }


def forecast_items(
    items: List[Dict[str, Any]],
    exo: Dict[str, bool],
    policy_cfg: Dict[str, Any],
):
    """Forecast Demand 14 hari menggunakan model XGBoost secara rekursif."""

    states = []

    for item in items:
        hist = [
            float(v)
            for v in (
                item.get("full_history")
                or item.get("historical_demand")
                or []
            )
        ]

        warnings = []

        if not hist:
            warnings.append(
                "Riwayat Demand kosong; ramalan menggunakan histori minimal."
            )

        codes = {
            "Product ID": encoders.code(
                "Product ID", item["sku_id"]
            ),
            "Store ID": encoders.code(
                "Store ID", item["store_id"]
            ),
            "Category": encoders.code(
                "Category", item.get("category", "")
            ),
            "Region": encoders.code(
                "Region", item.get("region", "")
            ),
        }

        unknown = [
            col for col, value in codes.items()
            if value == -1
        ]

        if unknown:
            warnings.append(
                "Kategori tidak dikenal encoder: "
                + ", ".join(unknown)
            )

        states.append(
            {
                "hist": hist,
                "codes": codes,
                "warnings": warnings,
                "last_date": pd.to_datetime(
                    item.get("last_date", "2024-01-30")
                ),
                "preds": [],
            }
        )

    for step in range(HORIZON):
        rows = []

        for state in states:
            date = state["last_date"] + pd.Timedelta(days=step + 1)

            cal = calendar_features(date)
            hf = history_features(state["hist"])

            row = {
                **{
                    f"{col}_code": value
                    for col, value in state["codes"].items()
                },
                **cal,
                **hf,
            }

            rows.append(row)

        if not rows:
            break

        X = pd.DataFrame(rows)[FEATURES]

        if SERVING_METHOD == "xgboost":
            preds = np.maximum(
                0.0,
                model.predict(X),
            )

        elif SERVING_METHOD == "moving_average_7":
            preds = np.array([row["rm_7"] for row in rows])

        else:
            preds = np.array([row["rm_28"] for row in rows])

        for state, prediction in zip(states, preds):
            prediction = max(0.0, float(prediction))

            state["preds"].append(round(prediction, 2))
            state["hist"].append(prediction)

    results = []

    lead = int(policy_cfg["lead_time"])
    review = int(policy_cfg["review_period"])

    protection_days = min(
        HORIZON,
        max(1, lead + review),
    )

    for item, state in zip(items, states):
        forecast_dates = [
            (
                state["last_date"]
                + pd.Timedelta(days=i + 1)
            ).strftime("%Y-%m-%d")
            for i in range(HORIZON)
        ]

        key = f"{item['store_id']}|{item['sku_id']}"

        sigma = float(
            RESID_STD.get(key, RESID_STD_GLOBAL)
        )

        if key not in RESID_STD:
            state["warnings"].append(
                "Sigma galat menggunakan nilai global."
            )

        policy = calculate_reorder_policy(
            demand_lead=sum(
                state["preds"][:protection_days]
            ),
            sigma_daily=sigma,
            on_hand=int(item.get("current_stock", 0)),
            on_order=int(item.get("on_order", 0)),
            lead_time=lead,
            review_period=review,
            service_level=float(policy_cfg["service_level"]),
        )

        results.append(
            (
                forecast_dates,
                state["preds"],
                policy,
                state["warnings"],
            )
        )

    return results


def make_policy_cfg(
    lead_time=None,
    review_period=None,
    service_level=None,
):
    return {
        "lead_time": (
            DEFAULT_LEAD_TIME
            if lead_time is None
            else lead_time
        ),
        "review_period": (
            DEFAULT_REVIEW_PERIOD
            if review_period is None
            else review_period
        ),
        "service_level": (
            DEFAULT_SERVICE_LEVEL
            if service_level is None
            else service_level
        ),
    }


class RecalculateRequest(BaseModel):
    items: List[Dict[str, Any]]
    exogenous: Dict[str, bool] = {}
    lead_time: Optional[int] = None
    review_period: Optional[int] = None
    service_level: Optional[float] = None


@app.get("/api/model-info")
def get_model_info():
    return {
        **model_info(),
        "metrics_test": model_spec.get("metrics_test", {}),
        "trained_at": model_spec.get("trained_at"),
    }


@app.post("/api/recalculate")
def recalculate_forecast(req: RecalculateRequest):
    try:
        cfg = make_policy_cfg(
            req.lead_time,
            req.review_period,
            req.service_level,
        )

        results = forecast_items(
            req.items,
            req.exogenous,
            cfg,
        )

    except (KeyError, ValueError) as e:
        raise HTTPException(
            status_code=400,
            detail=f"Input tidak valid: {e}",
        )

    updated = []

    for item, result in zip(req.items, results):
        forecast_dates, preds, policy, warnings = result

        new_item = dict(item)

        new_item.update(
            {
                "forecast_dates": forecast_dates,
                "predictions": preds,
                "inventory_policy": policy,
                "warnings": warnings,
            }
        )

        updated.append(new_item)

    return {
        "message": "Forecast Demand berhasil dikalkulasi ulang.",
        "model_info": model_info(),
        "processed_items": updated,
    }


@app.post("/api/upload-csv")
async def upload_and_predict(file: UploadFile = File(...)):
    if not (file.filename or "").lower().endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="File harus berformat .csv",
        )

    try:
        df = pd.read_csv(
            io.BytesIO(await file.read())
        )

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"CSV tidak bisa dibaca: {e}",
        )

    required_columns = [
        "Date",
        "Product ID",
        "Store ID",
        "Category",
        "Region",
        "Demand",
    ]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise HTTPException(
            status_code=400,
            detail=(
                "Kolom CSV tidak lengkap. "
                f"Kolom yang hilang: {missing}"
            ),
        )

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce",
    )

    df["Demand"] = pd.to_numeric(
        df["Demand"],
        errors="coerce",
    )

    for col in ["Inventory Level", "Units Ordered"]:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce",
            ).fillna(0)

    df = df.dropna(
        subset=["Date", "Demand"]
    ).sort_values("Date")

    if df.empty:
        raise HTTPException(
            status_code=400,
            detail="Tidak ada data valid.",
        )

    stores = sorted(
        df["Store ID"].astype(str).unique().tolist()
    )

    all_items = []

    for store in stores:
        df_store = df[
            df["Store ID"].astype(str) == store
        ]

        products = sorted(
            df_store["Product ID"].astype(str).unique().tolist()
        )

        for sku in products:
            data = (
                df_store[
                    df_store["Product ID"].astype(str) == sku
                ]
                .sort_values("Date")
            )

            last = data.iloc[-1]

            history = data["Demand"].astype(float).tolist()

            item = {
                "sku_id": sku,
                "store_id": store,
                "category": str(last["Category"]),
                "region": str(last["Region"]),
                "last_date": last["Date"].strftime("%Y-%m-%d"),
                "current_stock": int(
                    last.get("Inventory Level", 0)
                ),
                "on_order": int(
                    last.get("Units Ordered", 0)
                ),
                "full_history": history,
                "historical_dates": (
                    data.tail(14)["Date"]
                    .dt.strftime("%Y-%m-%d")
                    .tolist()
                ),
                "historical_demand": (
                    data.tail(14)["Demand"]
                    .astype(float)
                    .tolist()
                ),
            }

            all_items.append(item)

    try:
        results = forecast_items(
            all_items,
            {},
            make_policy_cfg(),
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Gagal meramal: {e}",
        )

    store_data = {
        store: [] for store in stores
    }

    for item, result in zip(all_items, results):
        forecast_dates, preds, policy, warnings = result

        item.update(
            {
                "forecast_dates": forecast_dates,
                "predictions": preds,
                "inventory_policy": policy,
                "warnings": warnings,
            }
        )

        store_data[item["store_id"]].append(item)

    return {
        "message": (
            f"Berhasil memproses {len(stores)} cabang "
            f"dari file {file.filename}"
        ),
        "stores": stores,
        "store_data": store_data,
        "model_accuracy": model_spec.get("accuracy"),
        "model_info": model_info(),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
    )
