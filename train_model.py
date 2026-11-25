
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from backend.features import CAT_COLS, FEATURES, TARGET, Encoders, PAYDAY_DAYS

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "dataset" / "sales_data.csv"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)


def wmape(y, p):
    y = np.asarray(y)
    p = np.asarray(p)
    return float(np.abs(y - p).sum() / max(1e-9, np.abs(y).sum()) * 100)


def metrics(y, p):
    y = np.asarray(y)
    p = np.asarray(p)
    return {
        "mae": float(np.mean(np.abs(y - p))),
        "rmse": float(np.sqrt(np.mean((y - p) ** 2))),
        "wmape": wmape(y, p),
    }


print("1. Membaca dataset:", DATASET_PATH)
df = pd.read_csv(DATASET_PATH)

required = [
    "Date", "Store ID", "Product ID", "Category", "Region", "Demand"
]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Kolom wajib tidak ditemukan: {missing}")

df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")

df = df.dropna(subset=["Date", TARGET]).copy()
df = df.sort_values(["Store ID", "Product ID", "Date"]).reset_index(drop=True)

# Fitur kalender
df["dayofweek"] = df["Date"].dt.dayofweek
df["month"] = df["Date"].dt.month
df["day"] = df["Date"].dt.day
df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
df["is_payday"] = df["day"].isin(PAYDAY_DAYS).astype(int)

# Lag dan rolling: hanya menggunakan Demand dari hari sebelumnya
g = df.groupby(["Store ID", "Product ID"])[TARGET]

for k in (1, 7, 14, 28):
    df[f"lag_{k}"] = g.shift(k)

df["rm_7"] = g.transform(
    lambda x: x.shift(1).rolling(7, min_periods=1).mean()
)
df["rm_28"] = g.transform(
    lambda x: x.shift(1).rolling(28, min_periods=1).mean()
)
df["std_7"] = g.transform(
    lambda x: x.shift(1).rolling(7, min_periods=2).std()
).fillna(0)

# Encoding kategori akan dipelajari dari train saja
df_model = df.dropna(
    subset=["lag_1", "lag_7", "lag_14", "lag_28"]
).copy()

# Split temporal 70 / 15 / 15
dates = np.sort(df_model["Date"].unique())
d70 = dates[int(len(dates) * 0.70)]
d85 = dates[int(len(dates) * 0.85)]

train_df = df_model[df_model["Date"] <= d70].copy()
val_df = df_model[
    (df_model["Date"] > d70) & (df_model["Date"] <= d85)
].copy()
test_df = df_model[df_model["Date"] > d85].copy()

print(
    f"2. Split data: train={len(train_df)}, "
    f"val={len(val_df)}, test={len(test_df)}"
)

# Encoder hanya dipelajari dari data training
classes = {
    c: sorted(train_df[c].astype(str).unique().tolist())
    for c in CAT_COLS
}
enc = Encoders(classes)

for part in (train_df, val_df, test_df):
    for c in CAT_COLS:
        part[f"{c}_code"] = (
            part[c].astype(str)
            .map(lambda v, c=c: enc.code(c, v))
            .astype(int)
        )

X_tr, y_tr = train_df[FEATURES], train_df[TARGET]
X_va, y_va = val_df[FEATURES], val_df[TARGET]
X_te, y_te = test_df[FEATURES], test_df[TARGET]

print("3. Training XGBoost...")
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
)

model.fit(
    X_tr,
    y_tr,
    eval_set=[(X_va, y_va)],
    verbose=False,
)

# Prediksi test
p_model = np.maximum(0, model.predict(X_te))

# Baseline
baseline_predictions = {
    "lag_1": test_df["lag_1"].to_numpy(),
    "lag_7": test_df["lag_7"].to_numpy(),
    "moving_average_7": test_df["rm_7"].to_numpy(),
    "moving_average_28": test_df["rm_28"].to_numpy(),
}

baseline_metrics = {
    name: metrics(y_te, pred)
    for name, pred in baseline_predictions.items()
}

model_metrics = metrics(y_te, p_model)

best_baseline_name = min(
    baseline_metrics,
    key=lambda name: baseline_metrics[name]["wmape"]
)
best_baseline = baseline_metrics[best_baseline_name]["wmape"]

beats_baseline = model_metrics["wmape"] < best_baseline

print("\n4. Evaluasi TEST:")
print("XGBoost:", model_metrics)

for name, result in baseline_metrics.items():
    print(f"{name}: {result}")

print("Baseline terbaik:", best_baseline_name)
print("Mengalahkan baseline:", beats_baseline)

# Residual validation saja untuk estimasi ketidakpastian
p_val = np.maximum(0, model.predict(X_va))
val_residuals = y_va.to_numpy() - p_val

residual_std_global = float(np.std(val_residuals, ddof=1))

residual_df = val_df[["Store ID", "Product ID"]].copy()
residual_df["resid"] = val_residuals

per_series = (
    residual_df.groupby(["Store ID", "Product ID"])["resid"]
    .std()
    .fillna(0)
)

residual_std = {
    f"{s}|{p}": round(float(v), 3)
    for (s, p), v in per_series.items()
}

# Simpan artifacts
model.save_model(ARTIFACTS_DIR / "xgboost_stock_model.json")

(ARTIFACTS_DIR / "encoders.json").write_text(
    json.dumps(classes, indent=2),
    encoding="utf-8",
)

spec = {
    "model_type": "XGBRegressor",
    "dataset": DATASET_PATH.name,
    "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "xgboost_version": xgb.__version__,
    "features": FEATURES,
    "target": TARGET,
    "split": {
        "train_end": str(pd.Timestamp(d70).date()),
        "val_end": str(pd.Timestamp(d85).date()),
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
}

(ARTIFACTS_DIR / "model_spec.json").write_text(
    json.dumps(spec, indent=2),
    encoding="utf-8",
)

print("\n5. Artifacts tersimpan.")
print("Serving method:", spec["serving_method"])
