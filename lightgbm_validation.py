
from pathlib import Path
import gc
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lightgbm import LGBMRegressor


# ============================================================
# 1. KONFIGURASI
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "raw"
OUTPUT_DIR = BASE_DIR / "outputs"
REPORT_DIR = BASE_DIR / "reports"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

SALES_PATH = DATA_DIR / "sales_train_validation.csv"
CALENDAR_PATH = DATA_DIR / "calendar.csv"
PRICE_PATH = DATA_DIR / "sell_prices.csv"
BASELINE_PATH = REPORT_DIR / "baseline_metrics.csv"
BASELINE_DAILY_PATH = REPORT_DIR / "baseline_daily_comparison.csv"

for path in [SALES_PATH, CALENDAR_PATH, PRICE_PATH]:
    if not path.exists():
        raise FileNotFoundError(f"File tidak ditemukan: {path}")


# ============================================================
# 2. MEMBACA DATA PENJUALAN
# ============================================================

print("=" * 65)
print("LIGHTGBM - DEMAND FORECASTING VALIDATION")
print("=" * 65)

all_columns = pd.read_csv(SALES_PATH, nrows=0).columns.tolist()

day_cols = sorted(
    [c for c in all_columns if c.startswith("d_")],
    key=lambda c: int(c.split("_")[1])
)

# 28 hari terakhir dipisahkan sebagai data pengujian.
cutoff = len(day_cols) - 28

if cutoff < 400:
    raise ValueError("Riwayat data terlalu pendek untuk eksperimen ini.")

metadata_cols = [
    "item_id", "dept_id", "cat_id", "store_id", "state_id"
]

sales_dtype = {col: "uint16" for col in day_cols}

print("\nMembaca data penjualan...")
sales = pd.read_csv(SALES_PATH, dtype=sales_dtype)

metadata = sales[metadata_cols].copy()

for col in metadata_cols:
    metadata[col] = metadata[col].astype("category")

# Matriks penjualan: baris = seri produk-toko, kolom = hari.
y = sales[day_cols].to_numpy(dtype=np.uint16, copy=True)

print(f"Jumlah seri produk-toko: {len(sales):,}")
print(f"Jumlah hari            : {len(day_cols):,}")
print(f"Hari pelatihan         : {cutoff:,}")
print(f"Hari pengujian         : {len(day_cols) - cutoff}")


# ============================================================
# 3. MEMBACA KALENDER
# ============================================================

calendar = pd.read_csv(CALENDAR_PATH, parse_dates=["date"])
calendar_by_day = calendar.set_index("d")

event_columns = [
    "event_name_1",
    "event_type_1",
    "event_name_2",
    "event_type_2",
]

event_categories = {}

for col in event_columns:
    values = calendar[col].dropna().astype(str).unique().tolist()
    event_categories[col] = sorted(set(values + ["None"]))

# Kalender di dataset M5 menggunakan state CA, TX, dan WI.
states = sales["state_id"].astype(str).to_numpy()


# ============================================================
# 4. TENTUKAN TITIK WAKTU UNTUK MELATIH MODEL
# ============================================================

# Mengambil titik waktu dari sepanjang riwayat pelatihan.
# Target setiap titik waktu tetap berada sebelum periode pengujian.
origin_indices = np.unique(
    np.linspace(365, cutoff - 1, 30, dtype=int)
)

forecast_indices = list(range(cutoff, len(day_cols)))
all_target_indices = list(origin_indices) + forecast_indices

print(f"\nJumlah titik waktu pelatihan: {len(origin_indices)}")


# ============================================================
# 5. MEMBACA HARGA DENGAN PENGGUNAAN MEMORI EFISIEN
# ============================================================

# Kita hanya perlu minggu harga yang digunakan untuk fitur model.
needed_weeks = set()

for idx in all_target_indices:
    for day_index in (idx, idx - 7):
        day_name = f"d_{day_index + 1}"
        needed_weeks.add(
            int(calendar_by_day.loc[day_name, "wm_yr_wk"])
        )

print("Membaca data harga produk...")

prices = pd.read_csv(
    PRICE_PATH,
    usecols=["store_id", "item_id", "wm_yr_wk", "sell_price"],
    dtype={
        "store_id": "category",
        "item_id": "category",
        "wm_yr_wk": "int32",
        "sell_price": "float32",
    },
)

prices = prices.loc[
    prices["wm_yr_wk"].isin(needed_weeks)
].copy()

# Kunci teks untuk mencocokkan harga dengan setiap seri produk-toko.
metadata_keys = sales[["store_id", "item_id"]].astype(str)

# Cache menyimpan vektor harga dalam urutan baris dataset penjualan.
price_cache = {}


def get_price_vector(week):
    """Mengembalikan harga untuk setiap seri produk-toko pada suatu minggu."""
    week = int(week)

    if week in price_cache:
        return price_cache[week]

    week_prices = prices.loc[
        prices["wm_yr_wk"].eq(week),
        ["store_id", "item_id", "sell_price"],
    ].copy()

    week_prices["store_id"] = week_prices["store_id"].astype(str)
    week_prices["item_id"] = week_prices["item_id"].astype(str)

    merged = metadata_keys.merge(
        week_prices,
        on=["store_id", "item_id"],
        how="left",
        sort=False,
        validate="one_to_one",
    )

    vector = merged["sell_price"].to_numpy(
        dtype=np.float32,
        na_value=np.nan,
    )

    price_cache[week] = vector
    return vector


# ============================================================
# 6. FEATURE ENGINEERING
# ============================================================

def build_features(target_idx, history):
    """
    Membuat fitur untuk memprediksi satu hari.
    history hanya boleh berisi data yang tersedia sebelum target_idx.
    """

    n_series = history.shape[0]
    X = metadata.copy()

    # Lag: penjualan pada hari sebelumnya.
    X["lag_1"] = history[:, target_idx - 1].astype(np.float32)
    X["lag_7"] = history[:, target_idx - 7].astype(np.float32)
    X["lag_14"] = history[:, target_idx - 14].astype(np.float32)
    X["lag_28"] = history[:, target_idx - 28].astype(np.float32)

    # Statistik penjualan historis; tidak menggunakan target hari ini.
    last_7 = history[:, target_idx - 7:target_idx].astype(
        np.float32, copy=False
    )
    last_28 = history[:, target_idx - 28:target_idx].astype(
        np.float32, copy=False
    )

    X["rolling_mean_7"] = last_7.mean(axis=1, dtype=np.float32)
    X["rolling_mean_28"] = last_28.mean(axis=1, dtype=np.float32)
    X["rolling_std_28"] = last_28.std(axis=1, dtype=np.float32)
    X["zero_ratio_28"] = (
        last_28 == 0
    ).mean(axis=1, dtype=np.float32)

    same_weekday = np.stack(
        [
            history[:, target_idx - 7],
            history[:, target_idx - 14],
            history[:, target_idx - 21],
            history[:, target_idx - 28],
        ],
        axis=1,
    ).astype(np.float32)

    X["same_weekday_mean_4w"] = same_weekday.mean(
        axis=1, dtype=np.float32
    )

    # Informasi kalender untuk hari yang sedang diprediksi.
    day_info = calendar_by_day.loc[f"d_{target_idx + 1}"]
    date_value = pd.Timestamp(day_info["date"])

    X["wday"] = pd.Categorical(
        np.full(n_series, int(day_info["wday"])),
        categories=list(range(1, 8)),
    )

    X["month"] = pd.Categorical(
        np.full(n_series, int(day_info["month"])),
        categories=list(range(1, 13)),
    )

    X["year"] = np.full(
        n_series, int(day_info["year"]), dtype=np.int16
    )
    X["day_of_month"] = date_value.day
    X["day_of_year"] = date_value.dayofyear

    is_weekend = int(
        day_info["weekday"] in ("Saturday", "Sunday")
    )
    X["is_weekend"] = np.full(
        n_series, is_weekend, dtype=np.int8
    )

    for col in event_columns:
        value = (
            str(day_info[col])
            if pd.notna(day_info[col])
            else "None"
        )

        X[col] = pd.Categorical(
            np.repeat(value, n_series),
            categories=event_categories[col],
        )

    # Fitur SNAP disesuaikan dengan state setiap seri.
    snap = np.zeros(n_series, dtype=np.int8)

    for state in ("CA", "TX", "WI"):
        state_mask = states == state
        snap[state_mask] = int(day_info[f"snap_{state}"])

    X["snap"] = snap

    # Harga pada minggu target dan harga tujuh hari sebelumnya.
    target_week = int(day_info["wm_yr_wk"])
    previous_idx = target_idx - 7

    previous_day_info = calendar_by_day.loc[
        f"d_{previous_idx + 1}"
    ]
    previous_week = int(previous_day_info["wm_yr_wk"])

    current_price = get_price_vector(target_week)
    previous_price = get_price_vector(previous_week)

    X["sell_price"] = current_price
    X["price_missing"] = (
        ~np.isfinite(current_price)
    ).astype(np.int8)

    # Perubahan harga mingguan. Jika salah satu harga tidak ada,
    # perubahan ditetapkan nol dan harga_missing tetap tersedia.
    price_change = np.zeros(n_series, dtype=np.float32)

    valid_price = (
        np.isfinite(current_price)
        & np.isfinite(previous_price)
        & (previous_price > 0)
    )

    price_change[valid_price] = (
        current_price[valid_price] / previous_price[valid_price]
    ) - 1.0

    X["price_change_pct"] = price_change
    X["discount_pct"] = np.maximum(-price_change, 0.0)

    return X


# ============================================================
# 7. SIAPKAN DATA LATIH
# ============================================================

X_parts = []
y_parts = []

print("\nMembuat fitur pelatihan...")

for count, target_idx in enumerate(origin_indices, start=1):
    X_day = build_features(target_idx, y)
    target_day = y[:, target_idx].astype(np.float32)

    X_parts.append(X_day)
    y_parts.append(target_day)

    if count % 5 == 0 or count == len(origin_indices):
        print(
            f"Fitur selesai: {count}/{len(origin_indices)} titik waktu"
        )

X_train = pd.concat(X_parts, ignore_index=True)
y_train = np.concatenate(y_parts)

del X_parts, y_parts
gc.collect()

print(f"\nJumlah baris data latih: {len(X_train):,}")
print(f"Jumlah fitur           : {X_train.shape[1]}")


# ============================================================
# 8. LATIH MODEL LIGHTGBM
# ============================================================

print("\nMelatih model LightGBM...")

model = LGBMRegressor(
    objective="regression",
    n_estimators=400,
    learning_rate=0.05,
    num_leaves=31,
    max_depth=-1,
    min_child_samples=100,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=max(1, min(8, os.cpu_count() or 4)),
    verbosity=-1,
)

model.fit(
    X_train,
    y_train,
    categorical_feature="auto",
)

# Simpan model agar dapat digunakan kembali.
model_path = OUTPUT_DIR / "lightgbm_demand_model.txt"
model.booster_.save_model(str(model_path))

# Simpan informasi fitur yang paling banyak digunakan model.
importance = pd.DataFrame({
    "feature": X_train.columns,
    "importance_gain": model.booster_.feature_importance(
        importance_type="gain"
    ),
})

importance = importance.sort_values(
    "importance_gain", ascending=False
)

importance.to_csv(
    REPORT_DIR / "lightgbm_feature_importance.csv",
    index=False,
)

print("Pelatihan selesai.")


# ============================================================
# 9. PREDIKSI 28 HARI SECARA BERURUTAN
# ============================================================

print("\nMemprediksi 28 hari pengujian...")

# History awal hanya berisi data sebelum periode pengujian.
# Kolom masa depan baru diisi setelah masing-masing prediksi dibuat.
history = np.empty(
    (len(sales), len(day_cols)),
    dtype=np.float32,
)

history[:, :cutoff] = y[:, :cutoff]

forecast_matrix = np.zeros(
    (len(sales), len(forecast_indices)),
    dtype=np.float32,
)

for step, target_idx in enumerate(forecast_indices):
    X_day = build_features(target_idx, history)

    predictions = model.predict(X_day)
    predictions = np.clip(predictions, 0, None)

    forecast_matrix[:, step] = predictions

    # Prediksi hari ini menjadi bagian dari riwayat untuk hari berikutnya.
    history[:, target_idx] = predictions

    if (step + 1) % 7 == 0 or step + 1 == len(forecast_indices):
        print(
            f"Prediksi selesai: {step + 1}/{len(forecast_indices)} hari"
        )


# ============================================================
# 10. EVALUASI MODEL
# ============================================================

actual = y[:, forecast_indices].astype(np.float32)
errors = actual - forecast_matrix
absolute_errors = np.abs(errors)

total_actual = actual.sum(dtype=np.float64)
total_abs_error = absolute_errors.sum(dtype=np.float64)

wape = (
    100 * total_abs_error / total_actual
    if total_actual > 0
    else float("nan")
)

mae = absolute_errors.mean()
rmse = np.sqrt(np.mean(errors.astype(np.float64) ** 2))

print("\n" + "=" * 55)
print("HASIL LIGHTGBM - VALIDASI 28 HARI")
print("=" * 55)
print(f"WAPE : {wape:.2f}%")
print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")


metrics = pd.DataFrame([
    {"metric": "WAPE (%)", "value": wape},
    {"metric": "MAE (units)", "value": mae},
    {"metric": "RMSE (units)", "value": rmse},
    {"metric": "Number of series", "value": len(sales)},
    {"metric": "Training days", "value": cutoff},
    {"metric": "Test days", "value": len(forecast_indices)},
])

metrics.to_csv(
    REPORT_DIR / "lightgbm_metrics.csv",
    index=False,
)


# ============================================================
# 11. EVALUASI PER STATE
# ============================================================

state_rows = []

for state in sorted(np.unique(states)):
    mask = states == state

    state_actual = actual[mask]
    state_pred = forecast_matrix[mask]
    state_error = state_actual - state_pred

    state_abs_error = np.abs(state_error)
    state_total_actual = state_actual.sum(dtype=np.float64)

    state_wape = (
        100 * state_abs_error.sum(dtype=np.float64) / state_total_actual
        if state_total_actual > 0
        else float("nan")
    )

    state_rows.append({
        "state_id": state,
        "series_count": int(mask.sum()),
        "WAPE_percent": state_wape,
        "MAE_units": state_abs_error.mean(),
        "RMSE_units": np.sqrt(
            np.mean(state_error.astype(np.float64) ** 2)
        ),
    })

pd.DataFrame(state_rows).to_csv(
    REPORT_DIR / "lightgbm_metrics_by_state.csv",
    index=False,
)


# ============================================================
# 12. SIMPAN HASIL HARIAN DAN GRAFIK
# ============================================================

test_dates = pd.to_datetime([
    calendar_by_day.loc[f"d_{idx + 1}", "date"]
    for idx in forecast_indices
])


# Simpan prediksi setiap seri produk-toko untuk proses hierarki.
forecast_date_labels = [
    pd.Timestamp(date).strftime("%Y-%m-%d")
    for date in test_dates
]

bottom_forecasts = sales[metadata_cols].copy()

for i, date_label in enumerate(forecast_date_labels):
    bottom_forecasts[f"forecast_{date_label}"] = (
        forecast_matrix[:, i]
    )

bottom_forecast_path = (
    REPORT_DIR / "lightgbm_bottom_level_forecasts.csv.gz"
)

bottom_forecasts.to_csv(
    bottom_forecast_path,
    index=False,
    compression="gzip",
)

print(f"Prediksi produk-toko tersimpan: {bottom_forecast_path}")


daily = pd.DataFrame({
    "date": test_dates,
    "actual_units": actual.sum(axis=0, dtype=np.float64),
    "lightgbm_forecast": forecast_matrix.sum(
        axis=0, dtype=np.float64
    ),
})

daily.to_csv(
    REPORT_DIR / "lightgbm_daily_comparison.csv",
    index=False,
)

plt.figure(figsize=(13, 6))

plt.plot(
    daily["date"],
    daily["actual_units"],
    label="Actual Sales",
    linewidth=2,
)

plt.plot(
    daily["date"],
    daily["lightgbm_forecast"],
    label="LightGBM Forecast",
    linewidth=2,
)

# Tambahkan baseline lama ke grafik bila tersedia.
if BASELINE_DAILY_PATH.exists():
    baseline_daily = pd.read_csv(BASELINE_DAILY_PATH)
    baseline_daily["date"] = pd.to_datetime(baseline_daily["date"])

    baseline_daily = baseline_daily.rename(
        columns={"forecast_units": "seasonal_naive_forecast"}
    )

    daily = daily.merge(
        baseline_daily[["date", "seasonal_naive_forecast"]],
        on="date",
        how="left",
    )

    plt.plot(
        daily["date"],
        daily["seasonal_naive_forecast"],
        label="Seasonal Naive",
        linestyle="--",
        linewidth=1.5,
    )

plt.title("M5 Walmart - Actual vs Forecast")
plt.xlabel("Date")
plt.ylabel("Total Units Sold")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

chart_path = OUTPUT_DIR / "lightgbm_28day_validation.png"
plt.savefig(chart_path, dpi=150)
plt.close()


# ============================================================
# 13. BANDINGKAN DENGAN BASELINE SEBELUMNYA
# ============================================================

comparison_rows = []

if BASELINE_PATH.exists():
    baseline = pd.read_csv(BASELINE_PATH)

    def baseline_value(metric_name):
        found = baseline.loc[
            baseline["metric"].eq(metric_name), "value"
        ]
        return float(found.iloc[0]) if not found.empty else np.nan

    comparison_rows.append({
        "model": "Seasonal Naive",
        "WAPE_percent": baseline_value("WAPE (%)"),
        "MAE_units": baseline_value("MAE (units)"),
        "RMSE_units": baseline_value("RMSE (units)"),
    })

comparison_rows.append({
    "model": "LightGBM",
    "WAPE_percent": wape,
    "MAE_units": mae,
    "RMSE_units": rmse,
})

comparison = pd.DataFrame(comparison_rows)

comparison.to_csv(
    REPORT_DIR / "model_comparison.csv",
    index=False,
)

print("\n--- PERBANDINGAN MODEL ---")
print(comparison.to_string(index=False))

print("\n--- FILE HASIL ---")
print(f"Metrik LightGBM    : {REPORT_DIR / 'lightgbm_metrics.csv'}")
print(f"Metrik per state   : {REPORT_DIR / 'lightgbm_metrics_by_state.csv'}")
print(f"Perbandingan model : {REPORT_DIR / 'model_comparison.csv'}")
print(f"Perbandingan harian: {REPORT_DIR / 'lightgbm_daily_comparison.csv'}")
print(f"Feature importance : {REPORT_DIR / 'lightgbm_feature_importance.csv'}")
print(f"Model tersimpan    : {model_path}")
print(f"Grafik             : {chart_path}")

print("\nLIGHTGBM VALIDATION SELESAI!")
