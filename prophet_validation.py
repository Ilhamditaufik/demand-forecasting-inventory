
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet

BASE_DIR = Path(__file__).resolve().parent
REPORT_DIR = BASE_DIR / "reports"
OUTPUT_DIR = BASE_DIR / "outputs"

REPORT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sales_path = REPORT_DIR / "daily_sales_summary.csv"
baseline_path = REPORT_DIR / "baseline_daily_comparison.csv"
lightgbm_path = REPORT_DIR / "lightgbm_daily_comparison.csv"
calendar_path = BASE_DIR / "data" / "raw" / "calendar.csv"

for path in [sales_path, baseline_path, lightgbm_path, calendar_path]:
    if not path.exists():
        raise FileNotFoundError(f"File tidak ditemukan: {path}")

print("=" * 60)
print("PROPHET - NATIONAL AGGREGATE VALIDATION")
print("=" * 60)

# 1. Ambil total penjualan nasional harian.
daily = pd.read_csv(sales_path, parse_dates=["date"])
daily = daily.sort_values("date").reset_index(drop=True)

if len(daily) <= 28:
    raise ValueError("Data tidak cukup untuk pengujian 28 hari.")

train = daily.iloc[:-28].copy()
test = daily.iloc[-28:].copy()

print(f"Data pelatihan : {len(train)} hari")
print(f"Data pengujian : {len(test)} hari")
print(f"Tanggal awal tes: {test['date'].min().date()}")
print(f"Tanggal akhir tes: {test['date'].max().date()}")

# 2. Siapkan informasi hari khusus dari kalender M5.
calendar = pd.read_csv(calendar_path, parse_dates=["date"])

event_frames = []

for col in ["event_name_1", "event_name_2"]:
    events = calendar.loc[
        calendar[col].notna() &
        calendar["date"].between(train["date"].min(), test["date"].max()),
        ["date", col],
    ].copy()

    events = events.rename(columns={
        "date": "ds",
        col: "holiday",
    })

    event_frames.append(events)

holidays = pd.concat(event_frames, ignore_index=True)
holidays = holidays.drop_duplicates(
    subset=["ds", "holiday"]
).reset_index(drop=True)

# Prophet memerlukan target bernama y dan tanggal bernama ds.
prophet_train = train.rename(columns={
    "date": "ds",
    "total_units_sold": "y",
})[["ds", "y"]]

# 3. Latih Prophet pada data sebelum periode pengujian.
print("\nMelatih model Prophet...")

model = Prophet(
    holidays=holidays if not holidays.empty else None,
    weekly_seasonality=True,
    yearly_seasonality=True,
    daily_seasonality=False,
    interval_width=0.80,
    uncertainty_samples=300,
)

model.fit(prophet_train)

# 4. Prediksi hanya pada 28 tanggal pengujian.
future = pd.DataFrame({"ds": test["date"]})
predictions = model.predict(future)

prophet_forecast = np.clip(
    predictions["yhat"].to_numpy(dtype=float),
    0,
    None,
)

actual = test["total_units_sold"].to_numpy(dtype=float)

# 5. Fungsi evaluasi pada tingkat agregat nasional.
def calculate_metrics(model_name, actual_values, predicted_values):
    errors = actual_values - predicted_values
    absolute_errors = np.abs(errors)

    total_actual = actual_values.sum()
    wape = (
        100 * absolute_errors.sum() / total_actual
        if total_actual > 0
        else float("nan")
    )

    mae = absolute_errors.mean()
    rmse = np.sqrt(np.mean(errors ** 2))

    return {
        "model": model_name,
        "WAPE_percent": wape,
        "MAE_units": mae,
        "RMSE_units": rmse,
    }


# 6. Ambil prediksi dari dua eksperimen yang sudah selesai.
comparison_data = pd.DataFrame({
    "date": test["date"],
    "actual_units": actual,
    "prophet_forecast": prophet_forecast,
})

baseline = pd.read_csv(
    baseline_path,
    parse_dates=["date"],
)

baseline = baseline[["date", "forecast_units"]].rename(
    columns={"forecast_units": "seasonal_naive_forecast"}
)

lightgbm = pd.read_csv(
    lightgbm_path,
    parse_dates=["date"],
)

lightgbm = lightgbm[["date", "lightgbm_forecast"]]

comparison_data = comparison_data.merge(
    baseline,
    on="date",
    how="left",
    validate="one_to_one",
)

comparison_data = comparison_data.merge(
    lightgbm,
    on="date",
    how="left",
    validate="one_to_one",
)

prediction_cols = [
    "seasonal_naive_forecast",
    "lightgbm_forecast",
    "prophet_forecast",
]

if comparison_data[prediction_cols].isna().any().any():
    raise ValueError(
        "Ada prediksi yang tidak cocok berdasarkan tanggal. "
        "Periksa file baseline dan LightGBM."
    )

# 7. Hitung metrik semua model pada tanggal dan tingkat agregasi sama.
rows = []

for col, model_name in [
    ("seasonal_naive_forecast", "Seasonal Naive"),
    ("lightgbm_forecast", "LightGBM"),
    ("prophet_forecast", "Prophet"),
]:
    rows.append(
        calculate_metrics(
            model_name,
            comparison_data["actual_units"].to_numpy(dtype=float),
            comparison_data[col].to_numpy(dtype=float),
        )
    )

metrics = pd.DataFrame(rows).sort_values(
    "WAPE_percent"
).reset_index(drop=True)

metrics.to_csv(
    REPORT_DIR / "national_aggregate_model_comparison.csv",
    index=False,
)

# 8. Simpan prediksi Prophet dan interval awalnya.
forecast_output = comparison_data.merge(
    predictions[
        ["ds", "yhat_lower", "yhat_upper"]
    ].rename(columns={"ds": "date"}),
    on="date",
    how="left",
    validate="one_to_one",
)

forecast_output.to_csv(
    REPORT_DIR / "prophet_daily_forecast.csv",
    index=False,
)

# 9. Buat grafik perbandingan.
plt.figure(figsize=(13, 6))

plt.plot(
    comparison_data["date"],
    comparison_data["actual_units"],
    label="Actual Sales",
    linewidth=2,
)

plt.plot(
    comparison_data["date"],
    comparison_data["seasonal_naive_forecast"],
    label="Seasonal Naive",
    linestyle="--",
)

plt.plot(
    comparison_data["date"],
    comparison_data["lightgbm_forecast"],
    label="LightGBM",
    linewidth=1.8,
)

plt.plot(
    comparison_data["date"],
    comparison_data["prophet_forecast"],
    label="Prophet",
    linewidth=1.8,
)

plt.title("National Daily Sales: Model Comparison")
plt.xlabel("Date")
plt.ylabel("Total Units Sold")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

chart_path = OUTPUT_DIR / "national_model_comparison.png"
plt.savefig(chart_path, dpi=150)
plt.close()

print("\n--- HASIL PERBANDINGAN NASIONAL ---")
print(metrics.round(4).to_string(index=False))

print("\n--- FILE HASIL ---")
print(f"Metrik perbandingan : {REPORT_DIR / 'national_aggregate_model_comparison.csv'}")
print(f"Prediksi Prophet    : {REPORT_DIR / 'prophet_daily_forecast.csv'}")
print(f"Grafik              : {chart_path}")

print("\nPROPHET VALIDATION SELESAI!")
