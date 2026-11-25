
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from backend.features_v2 import (
    CAT_COLS,
    FEATURES,
    TARGET,
    Encoders,
    PAYDAY_DAYS,
    seasonality_from_month,
)


BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "dataset" / "sales_data.csv"
ARTIFACTS_DIR = BASE_DIR / "artifacts_v2"
ARTIFACTS_DIR.mkdir(exist_ok=True)


def wmape(y, p):
    y = np.asarray(y)
    p = np.asarray(p)
    denominator = max(1e-9, np.abs(y).sum())
    return float(np.abs(y - p).sum() / denominator * 100)


def metrics(y, p):
    y = np.asarray(y)
    p = np.asarray(p)

    return {
        "mae": float(np.mean(np.abs(y - p))),
        "rmse": float(np.sqrt(np.mean((y - p) ** 2))),
        "wmape": wmape(y, p),
    }


print("1. Membaca dataset:", DATASET_PATH)

if not DATASET_PATH.exists():
    raise FileNotFoundError(
        f"Dataset tidak ditemukan: {DATASET_PATH}"
    )

df = pd.read_csv(DATASET_PATH)

required = [
    "Date",
    "Store ID",
    "Product ID",
    "Category",
    "Region",
    "Demand",
    "Promotion",
    "Discount",
    "Weather Condition",
    "Seasonality",
    "Epidemic",
]

missing = [col for col in required if col not in df.columns]

if missing:
    raise ValueError(f"Kolom dataset tidak ditemukan: {missing}")


# --------------------------------------------------
# 2. Membersihkan dan memvalidasi dataset
# --------------------------------------------------

df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

numeric_columns = [
    "Demand",
    "Promotion",
    "Discount",
    "Epidemic",
]

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=required).copy()

if df.empty:
    raise ValueError("Dataset kosong setelah proses pembersihan.")

for col in ["Store ID", "Product ID", "Category", "Region",
            "Weather Condition", "Seasonality"]:
    df[col] = df[col].astype(str).str.strip()

if not df["Promotion"].isin([0, 1]).all():
    raise ValueError("Promotion harus bernilai 0 atau 1.")

if not df["Epidemic"].isin([0, 1]).all():
    raise ValueError("Epidemic harus bernilai 0 atau 1.")

if (df["Discount"] < 0).any():
    raise ValueError("Discount tidak boleh negatif.")

if (df["Demand"] < 0).any():
    raise ValueError("Demand tidak boleh negatif.")

# Pastikan setiap series memiliki maksimal satu baris per tanggal.
duplicates = df.duplicated(
    subset=["Date", "Store ID", "Product ID"],
    keep=False,
)

if duplicates.any():
    raise ValueError(
        "Ditemukan tanggal duplikat untuk kombinasi Store ID dan Product ID."
    )

# Pastikan Seasonality sesuai dengan pola bulan dataset.
expected_seasonality = df["Date"].dt.month.map(
    seasonality_from_month
)

mismatch = df["Seasonality"] != expected_seasonality

if mismatch.any():
    raise ValueError(
        f"Ditemukan {int(mismatch.sum())} baris Seasonality "
        "yang tidak sesuai dengan pola bulan."
    )

# Pastikan cuaca konsisten per tanggal dan Region.
weather_variation = (
    df.groupby(["Date", "Region"])["Weather Condition"]
    .nunique()
)

if (weather_variation > 1).any():
    raise ValueError(
        "Weather Condition tidak konsisten per tanggal dan Region."
    )

# Pastikan Epidemic konsisten per tanggal.
epidemic_variation = df.groupby("Date")["Epidemic"].nunique()

if (epidemic_variation > 1).any():
    raise ValueError(
        "Epidemic tidak konsisten pada tanggal yang sama."
    )

print("Jumlah baris valid:", len(df))
print("Periode:", df["Date"].min(), "sampai", df["Date"].max())


# --------------------------------------------------
# 3. Membuat fitur kalender
# --------------------------------------------------

df = df.sort_values(
    ["Store ID", "Product ID", "Date"]
).reset_index(drop=True)

df["dayofweek"] = df["Date"].dt.dayofweek
df["month"] = df["Date"].dt.month
df["day"] = df["Date"].dt.day
df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
df["is_payday"] = df["day"].isin(PAYDAY_DAYS).astype(int)


# --------------------------------------------------
# 4. Membuat fitur lag dan rolling Demand
# --------------------------------------------------

grouped = df.groupby(["Store ID", "Product ID"])[TARGET]

for k in (1, 7, 14, 28):
    df[f"lag_{k}"] = grouped.shift(k)

df["rm_7"] = grouped.transform(
    lambda x: x.shift(1).rolling(7, min_periods=1).mean()
)

df["rm_28"] = grouped.transform(
    lambda x: x.shift(1).rolling(28, min_periods=1).mean()
)

df["std_7"] = grouped.transform(
    lambda x: x.shift(1).rolling(7, min_periods=2).std()
).fillna(0)


# --------------------------------------------------
# 5. Menyiapkan data model
# --------------------------------------------------

df_model = df.dropna(
    subset=["lag_1", "lag_7", "lag_14", "lag_28"]
).copy()

if df_model.empty:
    raise ValueError("Data training tidak cukup untuk membuat fitur lag.")

dates = np.sort(df_model["Date"].unique())

if len(dates) < 10:
    raise ValueError("Jumlah tanggal unik terlalu sedikit untuk split.")

d70 = dates[int(len(dates) * 0.70)]
d85 = dates[int(len(dates) * 0.85)]

train_df = df_model[df_model["Date"] <= d70].copy()

val_df = df_model[
    (df_model["Date"] > d70)
    & (df_model["Date"] <= d85)
].copy()

test_df = df_model[df_model["Date"] > d85].copy()

if train_df.empty or val_df.empty or test_df.empty:
    raise ValueError("Train, validation, atau test kosong.")

print(
    f"2. Split temporal: "
    f"train={len(train_df)}, "
    f"validation={len(val_df)}, "
    f"test={len(test_df)}"
)


# --------------------------------------------------
# 6. Encoding kategori dari train saja
# --------------------------------------------------

classes = {
    col: sorted(train_df[col].astype(str).unique().tolist())
    for col in CAT_COLS
}

encoders = Encoders(classes)

for part in (train_df, val_df, test_df):
    for col in CAT_COLS:
        part[f"{col}_code"] = (
            part[col]
            .astype(str)
            .map(lambda value, c=col: encoders.code(c, value))
            .astype(int)
        )


X_train = train_df[FEATURES]
y_train = train_df[TARGET]

X_val = val_df[FEATURES]
y_val = val_df[TARGET]

X_test = test_df[FEATURES]
y_test = test_df[TARGET]


# --------------------------------------------------
# 7. Training XGBoost
# --------------------------------------------------

print("3. Training XGBoost V2...")

model = xgb.XGBRegressor(
    n_estimators=1000,
    learning_rate=0.03,
    max_depth=4,
    min_child_weight=20,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=5.0,
    early_stopping_rounds=40,
    random_state=42,
    objective="reg:squarederror",
    eval_metric="mae",
    n_jobs=-1,
)

model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    verbose=False,
)


# --------------------------------------------------
# 8. Evaluasi test
# --------------------------------------------------

pred_model = np.maximum(0, model.predict(X_test))

baseline_predictions = {
    "lag_1": test_df["lag_1"].to_numpy(),
    "lag_7": test_df["lag_7"].to_numpy(),
    "moving_average_7": test_df["rm_7"].to_numpy(),
    "moving_average_28": test_df["rm_28"].to_numpy(),
}

baseline_metrics = {
    name: metrics(y_test, pred)
    for name, pred in baseline_predictions.items()
}

model_metrics = metrics(y_test, pred_model)

best_baseline_name = min(
    baseline_metrics,
    key=lambda name: baseline_metrics[name]["wmape"],
)

best_baseline = baseline_metrics[best_baseline_name]["wmape"]

beats_baseline = model_metrics["wmape"] < best_baseline

print("\n4. EVALUASI TEST V2")
print("XGBoost V2:", model_metrics)

for name, result in baseline_metrics.items():
    print(f"{name}: {result}")

print("Baseline terbaik:", best_baseline_name)
print("Mengalahkan baseline:", beats_baseline)


# --------------------------------------------------
# 9. Residual validation
# --------------------------------------------------

pred_val = np.maximum(0, model.predict(X_val))

val_residuals = y_val.to_numpy() - pred_val

residual_std_global = float(
    np.std(val_residuals, ddof=1)
)

residual_df = val_df[["Store ID", "Product ID"]].copy()
residual_df["resid"] = val_residuals

per_series = (
    residual_df.groupby(["Store ID", "Product ID"])["resid"]
    .std()
    .fillna(0)
)

residual_std = {
    f"{store}|{product}": round(float(value), 3)
    for (store, product), value in per_series.items()
}


# --------------------------------------------------
# 10. Feature importance
# --------------------------------------------------

feature_importance = {
    name: round(float(value), 6)
    for name, value in zip(
        FEATURES,
        model.feature_importances_,
    )
}


# --------------------------------------------------
# 11. Menyimpan artifacts V2
# --------------------------------------------------

print("\n5. Menyimpan artifacts V2...")

model.save_model(
    ARTIFACTS_DIR / "xgboost_stock_model_v2.json"
)

(ARTIFACTS_DIR / "encoders_v2.json").write_text(
    json.dumps(classes, indent=2),
    encoding="utf-8",
)

spec = {
    "model_type": "XGBRegressor",
    "model_version": "v2",
    "dataset": DATASET_PATH.name,
    "trained_at": datetime.now(timezone.utc).isoformat(
        timespec="seconds"
    ),
    "xgboost_version": xgb.__version__,
    "features": FEATURES,
    "target": TARGET,
    "split": {
        "train_end": str(pd.Timestamp(d70).date()),
        "validation_end": str(pd.Timestamp(d85).date()),
    },
    "best_iteration": int(model.best_iteration),
    "metrics_test": {
        "xgboost": model_metrics,
        "baselines": baseline_metrics,
    },
    "beats_baseline": bool(beats_baseline),
    "serving_method": (
        "xgboost" if beats_baseline else best_baseline_name
    ),
    "best_baseline": best_baseline_name,
    "best_baseline_wmape": round(float(best_baseline), 4),
    "residual_std_global": round(residual_std_global, 3),
    "residual_std": residual_std,
    "feature_importance": feature_importance,
}

(ARTIFACTS_DIR / "model_spec_v2.json").write_text(
    json.dumps(spec, indent=2),
    encoding="utf-8",
)

print("\n6. Training selesai.")
print("Folder:", ARTIFACTS_DIR)
print("Serving method:", spec["serving_method"])
print("WMAPE XGBoost V2:", round(model_metrics["wmape"], 4))
print("WMAPE baseline terbaik:", round(best_baseline, 4))
print("Mengalahkan baseline:", beats_baseline)
