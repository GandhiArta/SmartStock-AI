import math
from statistics import NormalDist


def calculate_reorder_policy(
    demand_lead,
    sigma_daily,
    on_hand,
    on_order=0,
    lead_time=3,
    review_period=0,
    service_level=0.95,
):
    """Safety stock & reorder point (periodic/continuous review, demand normal).

    demand_lead : total ramalan permintaan selama (lead_time + review_period) hari.
    sigma_daily : simpangan baku GALAT peramalan harian (dari residual validasi),
                  BUKAN sebaran angka prediksi.
    service_level : probabilitas tidak stockout (0.95 -> z = 1.645).
    """
    if not 0.5 < service_level < 0.9999:
        raise ValueError("service_level harus di antara 0.5 dan 0.9999")
    z = NormalDist().inv_cdf(service_level)
    protection_days = max(1, lead_time + review_period)
    sd_protection = max(0.0, sigma_daily) * math.sqrt(protection_days)

    safety_stock = z * sd_protection
    reorder_point = demand_lead + safety_stock
    position = on_hand + on_order  # inventory position = stok + pesanan dalam perjalanan

    suggested_qty = 0 if position > reorder_point else math.ceil(reorder_point - position)

    return {
        "safety_stock": math.ceil(safety_stock),
        "reorder_point": math.ceil(reorder_point),
        "suggested_order_qty": int(suggested_qty),
        "service_level": service_level,
        "lead_time_days": lead_time,
    }
