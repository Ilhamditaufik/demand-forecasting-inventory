
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
REPORT_DIR = BASE_DIR / "reports"
OUTPUT_DIR = BASE_DIR / "outputs"

REPORT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

source = REPORT_DIR / "prophet_daily_forecast.csv"

if not source.exists():
    raise FileNotFoundError(
        f"File prediksi tidak ditemukan: {source}"
    )

df = pd.read_csv(source, parse_dates=["date"])

required = [
    "date",
    "actual_units",
    "prophet_forecast",
    "yhat_lower",
    "yhat_upper",
]

missing = [col for col in required if col not in df.columns]

if missing:
    raise ValueError(f"Kolom yang diperlukan tidak tersedia: {missing}")

df = df.sort_values("date").reset_index(drop=True)

actual = df["actual_units"].to_numpy(dtype=float)
lower_raw = df["yhat_lower"].to_numpy(dtype=float)
upper_raw = df["yhat_upper"].to_numpy(dtype=float)

if not (
    np.isfinite(actual).all()
    and np.isfinite(lower_raw).all()
    and np.isfinite(upper_raw).all()
):
    raise ValueError("Ada nilai kosong atau tidak valid dalam data interval.")

# Permintaan tidak mungkin negatif.
lower = np.maximum(lower_raw, 0)
upper = np.maximum(upper_raw, lower)

# Apakah penjualan aktual berada di dalam interval?
covered = (actual >= lower) & (actual <= upper)

width = upper - lower

coverage = 100 * covered.mean()
mean_width = width.mean()
median_width = np.median(width)

total_actual = actual.sum()
normalized_width = (
    100 * width.mean() / actual.mean()
    if actual.mean() > 0
    else np.nan
)

metrics = pd.DataFrame([
    {
        "metric": "Nominal interval coverage (%)",
        "value": 80.0,
    },
    {
        "metric": "Empirical coverage (%)",
        "value": coverage,
    },
    {
        "metric": "Days covered",
        "value": int(covered.sum()),
    },
    {
        "metric": "Total test days",
        "value": len(df),
    },
    {
        "metric": "Mean interval width (units)",
        "value": mean_width,
    },
    {
        "metric": "Median interval width (units)",
        "value": median_width,
    },
    {
        "metric": "Mean width / mean actual (%)",
        "value": normalized_width,
    },
])

metrics.to_csv(
    REPORT_DIR / "prophet_interval_metrics.csv",
    index=False,
)

df["interval_lower"] = lower
df["interval_upper"] = upper
df["interval_width"] = width
df["covered"] = covered

df.to_csv(
    REPORT_DIR / "prophet_interval_evaluation.csv",
    index=False,
)

# Grafik aktual, prediksi titik, serta interval.
plt.figure(figsize=(13, 6))

plt.fill_between(
    df["date"],
    lower,
    upper,
    alpha=0.2,
    label="80% Prediction Interval",
)

plt.plot(
    df["date"],
    actual,
    label="Actual Sales",
    linewidth=2,
)

plt.plot(
    df["date"],
    df["prophet_forecast"],
    label="Prophet Forecast",
    linestyle="--",
    linewidth=1.8,
)

plt.title("Prophet - Prediction Interval Evaluation")
plt.xlabel("Date")
plt.ylabel("National Daily Units Sold")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

chart = OUTPUT_DIR / "prophet_prediction_intervals.png"
plt.savefig(chart, dpi=150)
plt.close()

print("=" * 55)
print("PROPHET PREDICTION INTERVAL EVALUATION")
print("=" * 55)
print(metrics.to_string(index=False))
print(f"\nHasil evaluasi: {REPORT_DIR / 'prophet_interval_metrics.csv'}")
print(f"Data interval : {REPORT_DIR / 'prophet_interval_evaluation.csv'}")
print(f"Grafik        : {chart}")
print("\nINTERVAL EVALUATION SELESAI!")
