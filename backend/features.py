"""Definisi fitur bersama untuk training dan serving SmartStock.

Target utama: Demand.
Fitur menggunakan identitas series, kalender, dan histori Demand.
Tidak menggunakan data aktual pada tanggal yang sedang diprediksi.
"""

import numpy as np

# Kategori yang digunakan untuk encoding.
CAT_COLS = [
    "Product ID",
    "Store ID",
    "Category",
    "Region",
]

# Target forecasting.
TARGET = "Demand"

# Fitur yang digunakan model.
FEATURES = [
    "Product ID_code",
    "Store ID_code",
    "Category_code",
    "Region_code",

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
]

PAYDAY_DAYS = frozenset([25, 26, 27, 28, 29, 30, 31, 1])


def calendar_features(d):
    """Fitur kalender untuk satu tanggal (pd.Timestamp)."""

    return {
        "dayofweek": d.dayofweek,
        "month": d.month,
        "day": d.day,
        "is_weekend": int(d.dayofweek >= 5),
        "is_payday": int(d.day in PAYDAY_DAYS),
    }


def history_features(history):
    """Fitur lag dan rolling dari riwayat Demand sampai hari sebelumnya."""

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
    """Encoder kategori. Nilai yang tidak dikenal -> -1."""

    def __init__(self, classes):
        self.classes = classes

        self._map = {
            c: {
                v: i
                for i, v in enumerate(vals)
            }
            for c, vals in classes.items()
        }

    def code(self, col, val):
        return self._map.get(col, {}).get(str(val), -1)