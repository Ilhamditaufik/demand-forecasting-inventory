
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "raw"
OUTPUT_DIR = BASE_DIR / "outputs"
REPORT_DIR = BASE_DIR / "reports"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

sales_path = DATA_DIR / "sales_train_validation.csv"
calendar_path = DATA_DIR / "calendar.csv"

if not sales_path.exists() or not calendar_path.exists():
    raise FileNotFoundError(
        "Dataset penjualan atau kalender tidak ditemukan di data/raw."
    )

print("=" * 60)
print("BASELINE FORECASTING - SEASONAL NAIVE")
print("=" * 60)

# 1. Identifikasi kolom harian secara berurutan.
columns = pd.read_csv(sales_path, nrows=0).columns.tolist()

day_cols = sorted(
    [col for col in columns if col.startswith("d_")],
    key=lambda col: int(col.split("_")[1])
)

if len(day_cols) <= 28:
    raise ValueError("Data harian tidak cukup untuk pengujian 28 hari.")

# Gunakan uint16 untuk menghemat memori.
dtype_map = {col: "uint16" for col in day_cols}

print("\nMembaca dataset penjualan...")
sales = pd.read_csv(sales_path, dtype=dtype_map)

# 2. Bagi data secara kronologis:
#    28 hari terakhir menjadi data pengujian.
train_days = day_cols[:-28]
test_days = day_cols[-28:]

print(f"Jumlah seri produk-toko : {len(sales):,}")
print(f"Total hari tersedia     : {len(day_cols):,}")
print(f"Hari untuk pelatihan    : {len(train_days):,}")
print(f"Hari untuk pengujian    : {len(test_days):,}")

# 3. Nilai aktual untuk 28 hari pengujian.
actual = sales[test_days].to_numpy(
    dtype=np.int32,
    copy=True
)

# 4. Ambil tujuh hari terakhir dari DATA PELATIHAN.
last_week = sales[train_days[-7:]].to_numpy(
    dtype=np.uint16,
    copy=True
)

# Ulangi pola tujuh hari tersebut sebanyak empat kali.
forecast = np.tile(last_week, (1, 4)).astype(np.int32)

# 5. Hitung metrik evaluasi.
errors = actual - forecast
absolute_errors = np.abs(errors)

total_actual = actual.sum(dtype=np.int64)
total_absolute_error = absolute_errors.sum(dtype=np.int64)

wape = (
    100 * total_absolute_error / total_actual
    if total_actual > 0
    else float("nan")
)

mae = absolute_errors.mean()
rmse = np.sqrt(np.mean(errors.astype(np.float64) ** 2))

metrics = pd.DataFrame([
    {"metric": "WAPE (%)", "value": wape},
    {"metric": "MAE (units)", "value": mae},
    {"metric": "RMSE (units)", "value": rmse},
    {"metric": "Number of series", "value": len(sales)},
    {"metric": "Training days", "value": len(train_days)},
    {"metric": "Test days", "value": len(test_days)},
])

metrics.to_csv(REPORT_DIR / "baseline_metrics.csv", index=False)

print("\n--- HASIL EVALUASI BASELINE ---")
print(f"WAPE : {wape:.2f}%")
print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")

# 6. Hitung metrik terpisah untuk setiap state.
state_rows = []

for state in sorted(sales["state_id"].unique()):
    mask = sales["state_id"].eq(state).to_numpy()

    state_actual = actual[mask]
    state_forecast = forecast[mask]

    state_errors = state_actual - state_forecast
    state_abs_errors = np.abs(state_errors)

    state_total_actual = state_actual.sum(dtype=np.int64)
    state_total_abs_error = state_abs_errors.sum(dtype=np.int64)

    state_wape = (
        100 * state_total_abs_error / state_total_actual
        if state_total_actual > 0
        else float("nan")
    )

    state_rows.append({
        "state_id": state,
        "series_count": int(mask.sum()),
        "WAPE_percent": state_wape,
        "MAE_units": state_abs_errors.mean(),
        "RMSE_units": np.sqrt(
            np.mean(state_errors.astype(np.float64) ** 2)
        ),
    })

pd.DataFrame(state_rows).to_csv(
    REPORT_DIR / "baseline_metrics_by_state.csv",
    index=False
)

# 7. Bandingkan total prediksi dan aktual untuk setiap hari.
calendar = pd.read_csv(calendar_path)

calendar_lookup = calendar.set_index("d")["date"]

test_dates = pd.to_datetime(
    calendar_lookup.loc[test_days].to_numpy()
)

daily_comparison = pd.DataFrame({
    "date": test_dates,
    "actual_units": actual.sum(axis=0, dtype=np.int64),
    "forecast_units": forecast.sum(axis=0, dtype=np.int64),
})

daily_comparison.to_csv(
    REPORT_DIR / "baseline_daily_comparison.csv",
    index=False
)

# 8. Simpan grafik perbandingan.
plt.figure(figsize=(13, 6))

plt.plot(
    daily_comparison["date"],
    daily_comparison["actual_units"],
    label="Actual Sales",
    linewidth=2
)

plt.plot(
    daily_comparison["date"],
    daily_comparison["forecast_units"],
    label="Seasonal Naive Forecast",
    linewidth=2
)

plt.title("M5 Walmart - 28-Day Baseline Validation")
plt.xlabel("Date")
plt.ylabel("Total Units Sold")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

chart_path = OUTPUT_DIR / "baseline_28day_validation.png"
plt.savefig(chart_path, dpi=150)
plt.close()

print("\n--- FILE HASIL ---")
print(f"Metrik keseluruhan : {REPORT_DIR / 'baseline_metrics.csv'}")
print(f"Metrik per state   : {REPORT_DIR / 'baseline_metrics_by_state.csv'}")
print(f"Perbandingan harian: {REPORT_DIR / 'baseline_daily_comparison.csv'}")
print(f"Grafik             : {chart_path}")

print("\nBASELINE VALIDATION SELESAI!")
