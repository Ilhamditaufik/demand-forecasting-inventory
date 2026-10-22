
from pathlib import Path

import numpy as np
import pandas as pd

from darts import TimeSeries
from darts.dataprocessing.transformers.reconciliation import (
    BottomUpReconciliator,
)


BASE_DIR = Path(__file__).resolve().parent
REPORT_DIR = BASE_DIR / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_PATH = REPORT_DIR / "lightgbm_bottom_level_forecasts.csv.gz"

if not INPUT_PATH.exists():
    raise FileNotFoundError(
        f"File prediksi tidak ditemukan: {INPUT_PATH}\n"
        "Jalankan lightgbm_validation.py terlebih dahulu."
    )

print("=" * 60)
print("HIERARCHICAL FORECAST RECONCILIATION - DARTS")
print("=" * 60)

# 1. Baca prediksi pada level produk-toko.
bottom = pd.read_csv(INPUT_PATH)

metadata_cols = [
    "item_id",
    "dept_id",
    "cat_id",
    "store_id",
    "state_id",
]

required = set(metadata_cols)
missing = required - set(bottom.columns)

if missing:
    raise ValueError(f"Kolom metadata tidak ditemukan: {missing}")

forecast_cols = sorted(
    [c for c in bottom.columns if c.startswith("forecast_")],
    key=lambda c: c.replace("forecast_", ""),
)

if len(forecast_cols) != 28:
    raise ValueError(
        f"Diharapkan 28 kolom forecast, ditemukan {len(forecast_cols)}."
    )

if bottom[forecast_cols].isna().any().any():
    raise ValueError("Terdapat nilai prediksi kosong.")

print(f"Seri produk-toko: {len(bottom):,}")
print(f"Hari prediksi    : {len(forecast_cols)}")

# 2. Agregasikan prediksi menjadi total setiap toko.
store_totals = (
    bottom.groupby(
        ["state_id", "store_id"],
        observed=True,
    )[forecast_cols]
    .sum()
    .reset_index()
)

store_totals["state_id"] = store_totals["state_id"].astype(str)
store_totals["store_id"] = store_totals["store_id"].astype(str)

stores = sorted(store_totals["store_id"].unique())
states = sorted(store_totals["state_id"].unique())

store_to_state = (
    store_totals[["store_id", "state_id"]]
    .drop_duplicates()
    .set_index("store_id")["state_id"]
    .to_dict()
)

if store_totals["store_id"].nunique() != len(stores):
    raise ValueError("Ditemukan identitas toko yang tidak konsisten.")

# 3. Bentuk tabel dengan setiap toko sebagai satu komponen.
dates = [
    pd.to_datetime(c.replace("forecast_", ""))
    for c in forecast_cols
]

wide = pd.DataFrame({"date": dates})

for store in stores:
    row = store_totals.loc[
        store_totals["store_id"].eq(store),
        forecast_cols,
    ]

    if len(row) != 1:
        raise ValueError(f"Data toko {store} tidak unik.")

    wide[store] = row.iloc[0].to_numpy(dtype=float)

# Nilai state dan nasional awal hanya untuk membentuk struktur lengkap.
# BottomUpReconciliator akan membangun ulang nilai agregat dari toko.
for state in states:
    state_stores = [
        store for store in stores
        if store_to_state[store] == state
    ]
    wide[state] = wide[state_stores].sum(axis=1)

wide["Total"] = wide[stores].sum(axis=1)

component_cols = stores + states + ["Total"]

# 4. Definisikan hubungan hierarki:
# toko -> state -> nasional.
hierarchy = {}

for store in stores:
    hierarchy[store] = [store_to_state[store]]

for state in states:
    hierarchy[state] = ["Total"]

# 5. Konversi ke Darts TimeSeries.
series = TimeSeries.from_dataframe(
    wide,
    time_col="date",
    value_cols=component_cols,
    freq="D",
    hierarchy=hierarchy,
)

# 6. Rekonsiliasi bottom-up.
reconciler = BottomUpReconciliator()
reconciled = reconciler.transform(series)

result = reconciled.to_dataframe()
result.index.name = "date"
result = result.reset_index()

# 7. Verifikasi bahwa total toko = total state
# dan total seluruh state = nasional.
state_errors = []

for state in states:
    state_stores = [
        store for store in stores
        if store_to_state[store] == state
    ]

    error = (
        result[state] - result[state_stores].sum(axis=1)
    ).abs().max()

    state_errors.append(float(error))

max_state_error = max(state_errors, default=0.0)

max_national_error = float(
    (
        result["Total"] - result[states].sum(axis=1)
    ).abs().max()
)

print("\n--- PEMERIKSAAN KONSISTENSI ---")
print(f"Jumlah toko       : {len(stores)}")
print(f"Jumlah state      : {len(states)}")
print(f"Kesalahan agregasi maksimum per state: {max_state_error:.8f}")
print(f"Kesalahan agregasi nasional          : {max_national_error:.8f}")

tolerance = 1e-4

if max_state_error > tolerance or max_national_error > tolerance:
    raise AssertionError(
        "Rekonsiliasi belum konsisten. Periksa hubungan hierarki."
    )

# 8. Simpan hasil untuk laporan dan dashboard.
wide_path = REPORT_DIR / "hierarchical_reconciled_wide.csv"
result.to_csv(wide_path, index=False)

long_result = result.melt(
    id_vars=["date"],
    value_vars=component_cols,
    var_name="node",
    value_name="forecast_units",
)

level_map = {
    **{store: "store" for store in stores},
    **{state: "state" for state in states},
    "Total": "national",
}

long_result["level"] = long_result["node"].map(level_map)

long_path = REPORT_DIR / "hierarchical_reconciled_forecasts.csv"
long_result.to_csv(long_path, index=False)

print("\n--- FILE HASIL ---")
print(f"Tabel hierarki : {wide_path}")
print(f"Data dashboard: {long_path}")
print("\nHIERARCHICAL RECONCILIATION SELESAI!")
