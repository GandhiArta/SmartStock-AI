
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
from pydantic import BaseModel, Field

from .features_v2 import (
    FEATURES,
    CAT_COLS,
    Encoders,
    calendar_features,
    history_features,
)
from .inventory_policy import calculate_reorder_policy


# =========================================================
# APPLICATION
# =========================================================

app = FastAPI(title="SmartStock AI API V2")

_default_origins = ",".join(
    f"http://{h}:{p}"
    for h in ("127.0.0.1", "localhost")
    for p in (3000, 5173, 5500, 8000, 8080)
)

ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        "CORS_ORIGINS", _default_origins
    ).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# =========================================================
# CONFIGURATION
# =========================================================

ARTIFACTS_DIR = (
    Path(__file__).resolve().parent.parent / "artifacts_v2"
)

HORIZON = 14

DEFAULT_LEAD_TIME = int(
    os.getenv("LEAD_TIME_DAYS", "3")
)

DEFAULT_REVIEW_PERIOD = int(
    os.getenv("REVIEW_PERIOD_DAYS", "0")
)

DEFAULT_SERVICE_LEVEL = float(
    os.getenv("SERVICE_LEVEL", "0.95")
)


# =========================================================
# LOAD V2 MODEL
# =========================================================

try:
    model_spec = json.loads(
        (ARTIFACTS_DIR / "model_spec_v2.json").read_text(
            encoding="utf-8"
        )
    )

    encoder_data = json.loads(
        (ARTIFACTS_DIR / "encoders_v2.json").read_text(
            encoding="utf-8"
        )
    )

    encoders = Encoders(encoder_data)

    model = xgb.XGBRegressor()
    model.load_model(
        ARTIFACTS_DIR / "xgboost_stock_model_v2.json"
    )

except Exception as e:
    raise RuntimeError(
        f"Artifacts V2 tidak bisa dimuat: {e}. "
        "Jalankan python train_model_v2.py terlebih dahulu."
    ) from e


if model_spec.get("features") != FEATURES:
    raise RuntimeError(
        "Fitur model_spec_v2.json tidak cocok "
        "dengan backend/features_v2.py."
    )


SERVING_METHOD = model_spec.get(
    "serving_method", "xgboost"
)

RESID_STD = model_spec.get("residual_std", {})

RESID_STD_GLOBAL = float(
    model_spec.get("residual_std_global", 1.0)
)

print("Artifacts V2 berhasil dimuat.")
print("Serving method:", SERVING_METHOD)
print("Jumlah fitur:", len(FEATURES))


# =========================================================
# MODEL INFORMATION
# =========================================================

def model_info() -> Dict[str, Any]:
    metrics = model_spec.get("metrics_test", {})

    return {
        "model_version": "v2",
        "model_type": "XGBRegressor",
        "serving_method": SERVING_METHOD,
        "wmape_xgboost": metrics.get("xgboost"),
        "wmape_best_baseline": model_spec.get(
            "best_baseline_wmape"
        ),
        "beats_baseline": model_spec.get(
            "beats_baseline", False
        ),
        "scenario_effective": True,
        "target": "Demand",
        "features_count": len(FEATURES),
        "trained_at": model_spec.get("trained_at"),
    }


# =========================================================
# SCENARIO VALIDATION
# =========================================================

def validate_scenario(exo: Dict[str, Any]) -> None:
    epidemic = exo.get("epidemic")

    if epidemic is not None:
        if epidemic not in (0, 1, False, True):
            raise ValueError(
                "Epidemic harus bernilai 0 atau 1."
            )

    weather_map = exo.get("weather_by_region", {})

    if not isinstance(weather_map, dict):
        raise ValueError(
            "weather_by_region harus berupa object."
        )

    valid_weather = {
        "Cloudy", "Rainy", "Snowy", "Sunny"
    }

    for region, weather in weather_map.items():
        if weather not in valid_weather:
            raise ValueError(
                f"Weather Condition tidak valid untuk "
                f"{region}: {weather}"
            )

    item_scenarios = exo.get("items", {})

    if not isinstance(item_scenarios, dict):
        raise ValueError(
            "items pada exogenous harus berupa object."
        )

    for key, values in item_scenarios.items():
        if not isinstance(values, dict):
            raise ValueError(
                f"Skenario item {key} harus berupa object."
            )

        promotion = values.get("promotion")

        if promotion is not None and promotion not in (
            0, 1, False, True
        ):
            raise ValueError(
                f"Promotion untuk {key} harus 0 atau 1."
            )

        discount = values.get("discount")

        if discount is not None:
            discount = float(discount)

            if not 0 <= discount <= 100:
                raise ValueError(
                    f"Discount untuk {key} harus "
                    "berada pada rentang 0 sampai 100."
                )


# =========================================================
# FORECAST ENGINE
# =========================================================

def forecast_items(
    items: List[Dict[str, Any]],
    exo: Dict[str, Any],
    policy_cfg: Dict[str, Any],
):
    """
    Forecast Demand 14 hari menggunakan XGBoost V2.

    Skenario eksternal:
    - epidemic: global
    - weather_by_region: berdasarkan Region
    - items: berdasarkan Store ID|Product ID
    """

    validate_scenario(exo)

    states = []

    global_epidemic = exo.get("epidemic")

    weather_by_region = exo.get(
        "weather_by_region", {}
    )

    item_scenarios = exo.get("items", {})

    for item in items:
        hist = [
            float(value)
            for value in (
                item.get("full_history")
                or item.get("historical_demand")
                or []
            )
        ]

        warnings = []

        if not hist:
            warnings.append(
                "Riwayat Demand kosong."
            )

        codes = {
            col: encoders.code(col, item.get(field, ""))
            for col, field in [
                ("Product ID", "sku_id"),
                ("Store ID", "store_id"),
                ("Category", "category"),
                ("Region", "region"),
            ]
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

        scenario_defaults = item.get(
            "scenario_defaults", {}
        )

        if not isinstance(scenario_defaults, dict):
            scenario_defaults = {}

        item_key = (
            f"{item['store_id']}|{item['sku_id']}"
        )

        override = item_scenarios.get(
            item_key, {}
        )

        if not isinstance(override, dict):
            override = {}

        region = str(item.get("region", ""))

        weather = weather_by_region.get(
            region,
            scenario_defaults.get(
                "weather", "Cloudy"
            ),
        )

        promotion = override.get(
            "promotion",
            scenario_defaults.get("promotion", 0),
        )

        discount = override.get(
            "discount",
            scenario_defaults.get("discount", 0),
        )

        epidemic = (
            global_epidemic
            if global_epidemic is not None
            else scenario_defaults.get("epidemic", 0)
        )

        weather_code = encoders.code(
            "Weather Condition", weather
        )

        seasonality_code_warning = False

        if weather_code == -1:
            warnings.append(
                f"Weather Condition tidak dikenal: {weather}"
            )

        if global_epidemic is None:
            warnings.append(
                "Epidemic memakai nilai terakhir dari CSV."
            )

        if region not in weather_by_region:
            warnings.append(
                f"Weather untuk {region} memakai "
                "nilai terakhir dari CSV atau default."
            )

        if item_key not in item_scenarios:
            warnings.append(
                "Promotion dan Discount memakai "
                "nilai terakhir dari CSV atau default."
            )

        codes["Weather Condition"] = weather_code

        states.append({
            "hist": hist,
            "codes": codes,
            "warnings": warnings,
            "last_date": pd.to_datetime(
                item.get("last_date")
            ),
            "weather": weather,
            "promotion": float(promotion),
            "discount": float(discount),
            "epidemic": float(epidemic),
            "preds": [],
        })

    # -----------------------------------------------------
    # RECURSIVE FORECAST
    # -----------------------------------------------------

    for step in range(HORIZON):
        rows = []

        for state in states:
            date = (
                state["last_date"]
                + pd.Timedelta(days=step + 1)
            )

            cal = calendar_features(date)
            hf = history_features(state["hist"])

            row = {
                **{
                    f"{col}_code": value
                    for col, value in state["codes"].items()
                },
                **cal,
                **hf,
                "Promotion": state["promotion"],
                "Discount": state["discount"],
                "Epidemic": state["epidemic"],
            }

            rows.append(row)

        if not rows:
            break

        X = pd.DataFrame(
            rows, columns=FEATURES
        )

        if SERVING_METHOD == "xgboost":
            preds = np.maximum(
                0.0,
                model.predict(X),
            )

        elif SERVING_METHOD == "moving_average_7":
            preds = np.array([
                row["rm_7"] for row in rows
            ])

        else:
            preds = np.array([
                row["rm_28"] for row in rows
            ])

        for state, prediction in zip(states, preds):
            prediction = max(
                0.0, float(prediction)
            )

            state["preds"].append(
                round(prediction, 2)
            )

            state["hist"].append(prediction)

    # -----------------------------------------------------
    # INVENTORY POLICY
    # -----------------------------------------------------

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

        key = (
            f"{item['store_id']}|{item['sku_id']}"
        )

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
            service_level=float(
                policy_cfg["service_level"]
            ),
        )

        results.append((
            forecast_dates,
            state["preds"],
            policy,
            state["warnings"],
        ))

    return results


# =========================================================
# INVENTORY POLICY CONFIGURATION
# =========================================================

def make_policy_cfg(
    lead_time=None,
    review_period=None,
    service_level=None,
):
    lead = (
        DEFAULT_LEAD_TIME
        if lead_time is None else lead_time
    )

    review = (
        DEFAULT_REVIEW_PERIOD
        if review_period is None else review_period
    )

    service = (
        DEFAULT_SERVICE_LEVEL
        if service_level is None else service_level
    )

    if not 1 <= int(lead) <= 365:
        raise ValueError(
            "Lead time harus antara 1 sampai 365 hari."
        )

    if not 0 <= int(review) <= 365:
        raise ValueError(
            "Review period harus antara 0 sampai 365 hari."
        )

    if not 0.5 <= float(service) <= 0.999:
        raise ValueError(
            "Service level harus antara 0.5 sampai 0.999."
        )

    return {
        "lead_time": int(lead),
        "review_period": int(review),
        "service_level": float(service),
    }


# =========================================================
# REQUEST MODEL
# =========================================================

class RecalculateRequest(BaseModel):
    items: List[Dict[str, Any]]
    exogenous: Dict[str, Any] = Field(
        default_factory=dict
    )
    lead_time: Optional[int] = None
    review_period: Optional[int] = None
    service_level: Optional[float] = None


# =========================================================
# MODEL INFO ENDPOINT
# =========================================================

@app.get("/api/model-info")
def get_model_info():
    return {
        **model_info(),
        "metrics_test": model_spec.get(
            "metrics_test", {}
        ),
        "trained_at": model_spec.get("trained_at"),
    }


# =========================================================
# RECALCULATE ENDPOINT
# =========================================================

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

    except (KeyError, ValueError, TypeError) as e:
        raise HTTPException(
            status_code=400,
            detail=f"Input tidak valid: {e}",
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Gagal menghitung forecast: {e}",
        )

    updated = []

    for item, result in zip(req.items, results):
        dates, preds, policy, warnings = result

        new_item = dict(item)

        new_item.update({
            "forecast_dates": dates,
            "predictions": preds,
            "inventory_policy": policy,
            "warnings": warnings,
        })

        updated.append(new_item)

    return {
        "message": "Forecast V2 berhasil dihitung ulang.",
        "model_info": model_info(),
        "processed_items": updated,
    }


# =========================================================
# CSV UPLOAD ENDPOINT
# =========================================================

@app.post("/api/upload-csv")
async def upload_and_predict(
    file: UploadFile = File(...),
):
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
                f"Kolom hilang: {missing}"
            ),
        )

    df["Date"] = pd.to_datetime(
        df["Date"], errors="coerce"
    )

    df["Demand"] = pd.to_numeric(
        df["Demand"], errors="coerce"
    )

    for col in [
        "Inventory Level",
        "Units Ordered",
        "Promotion",
        "Discount",
        "Epidemic",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col], errors="coerce"
            )

    df = df.dropna(
        subset=[
            "Date",
            "Demand",
            "Product ID",
            "Store ID",
            "Category",
            "Region",
        ]
    ).sort_values("Date")

    if df.empty:
        raise HTTPException(
            status_code=400,
            detail="Tidak ada data valid.",
        )

    if (df["Demand"] < 0).any():
        raise HTTPException(
            status_code=400,
            detail="Demand tidak boleh negatif.",
        )

    stores = sorted(
        df["Store ID"].astype(str).unique().tolist()
    )

    # Weather terbaru untuk setiap Region.
    weather_by_region = {}

    if "Weather Condition" in df.columns:
        weather_data = df.dropna(
            subset=["Weather Condition"]
        ).sort_values("Date")

        for region, group in weather_data.groupby(
            "Region"
        ):
            weather_by_region[str(region)] = str(
                group.iloc[-1]["Weather Condition"]
            )

    # Epidemic terbaru pada tanggal terakhir.
    latest_epidemic = 0

    if "Epidemic" in df.columns:
        epidemic_data = df.dropna(
            subset=["Epidemic"]
        ).sort_values("Date")

        if not epidemic_data.empty:
            latest_epidemic = int(
                epidemic_data.iloc[-1]["Epidemic"]
            )

    all_items = []

    for store in stores:
        df_store = df[
            df["Store ID"].astype(str) == store
        ]

        products = sorted(
            df_store["Product ID"]
            .astype(str)
            .unique()
            .tolist()
        )

        for sku in products:
            data = (
                df_store[
                    df_store["Product ID"].astype(str) == sku
                ]
                .sort_values("Date")
            )

            last = data.iloc[-1]

            history = (
                data["Demand"].astype(float).tolist()
            )

            def latest_value(column, default):
                if column not in data.columns:
                    return default

                values = data[column].dropna()

                if values.empty:
                    return default

                return values.iloc[-1]

            scenario_defaults = {
                "weather": str(
                    latest_value(
                        "Weather Condition", "Cloudy"
                    )
                ),
                "promotion": int(
                    latest_value("Promotion", 0)
                ),
                "discount": float(
                    latest_value("Discount", 0)
                ),
                "epidemic": latest_epidemic,
            }

            item = {
                "sku_id": sku,
                "store_id": store,
                "category": str(last["Category"]),
                "region": str(last["Region"]),
                "last_date": last["Date"].strftime(
                    "%Y-%m-%d"
                ),
                "current_stock": int(
                    last.get("Inventory Level", 0)
                    if pd.notna(
                        last.get("Inventory Level", 0)
                    )
                    else 0
                ),
                "on_order": int(
                    last.get("Units Ordered", 0)
                    if pd.notna(
                        last.get("Units Ordered", 0)
                    )
                    else 0
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
                "scenario_defaults": scenario_defaults,
            }

            all_items.append(item)

    default_exogenous = {
        "epidemic": latest_epidemic,
        "weather_by_region": weather_by_region,
        "items": {},
    }

    try:
        results = forecast_items(
            all_items,
            default_exogenous,
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
        dates, preds, policy, warnings = result

        item.update({
            "forecast_dates": dates,
            "predictions": preds,
            "inventory_policy": policy,
            "warnings": warnings,
        })

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
        "default_exogenous": default_exogenous,
    }


# =========================================================
# RUN SERVER
# =========================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8001,
    )
