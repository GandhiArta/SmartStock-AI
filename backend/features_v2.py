
"""Definisi fitur bersama untuk training dan serving SmartStock V2."""

import numpy as np

# Kolom kategorikal yang akan di-encode.
CAT_COLS = [
    "Product ID",
    "Store ID",
    "Category",
    "Region",
    "Weather Condition",
    "Seasonality",
]

TARGET = "Demand"

# Seluruh fitur yang digunakan model V2.
FEATURES = [
    "Product ID_code",
    "Store ID_code",
    "Category_code",
    "Region_code",
    "Weather Condition_code",
    "Seasonality_code",

    "dayofweek",
    "month",
    "day",
    "is_weekend",
    "is_payday",

    "lag_1",
    "lag_7",
    "lag_14",
    "lag_28",

    "rm_7",
    "rm_28",
    "std_7",

    "Promotion",
    "Discount",
    "Epidemic",
]

PAYDAY_DAYS = frozenset([25, 26, 27, 28, 29, 30, 31, 1])


def seasonality_from_month(month):
    """Menentukan seasonality sesuai pola pada dataset."""

    if month in (12, 1, 2):
        return "Winter"
    if month in (3, 4, 5):
        return "Spring"
    if month in (6, 7, 8):
        return "Summer"
    return "Autumn"


def calendar_features(d):
    """Fitur kalender untuk satu tanggal."""

    return {
        "dayofweek": d.dayofweek,
        "month": d.month,
        "day": d.day,
        "is_weekend": int(d.dayofweek >= 5),
        "is_payday": int(d.day in PAYDAY_DAYS),
        "Seasonality": seasonality_from_month(d.month),
    }


def history_features(history):
    """Fitur histori Demand sampai hari sebelum prediksi."""

    h = [float(v) for v in history]

    if not h:
        h = [0.0]

    l1 = h[-1]
    l7 = h[-7] if len(h) >= 7 else l1
    l14 = h[-14] if len(h) >= 14 else l7
    l28 = h[-28] if len(h) >= 28 else l14

    last7 = h[-7:]
    last28 = h[-28:]

    return {
        "lag_1": l1,
        "lag_7": l7,
        "lag_14": l14,
        "lag_28": l28,
        "rm_7": float(np.mean(last7)),
        "rm_28": float(np.mean(last28)),
        "std_7": (
            float(np.std(last7, ddof=1))
            if len(last7) > 1
            else 0.0
        ),
    }


class Encoders:
    """Encoder kategori. Nilai tidak dikenal menghasilkan -1."""

    def __init__(self, classes):
        self.classes = classes

        self._map = {
            col: {
                str(value): index
                for index, value in enumerate(values)
            }
            for col, values in classes.items()
        }

    def code(self, col, val):
        return self._map.get(col, {}).get(str(val), -1)
