
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import lightning.pytorch as pl

from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_forecasting import (
    TimeSeriesDataSet,
    TemporalFusionTransformer,
)
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss


# ============================================================
# 1. KONFIGURASI
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "raw"
REPORT_DIR = BASE_DIR / "reports"
OUTPUT_DIR = BASE_DIR / "outputs"

REPORT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SALES_PATH = DATA_DIR / "sales_train_validation.csv"
CALENDAR_PATH = DATA_DIR / "calendar.csv"
LIGHTGBM_PATH = REPORT_DIR / "lightgbm_bottom_level_forecasts.csv.gz"

# Eksperimen terbatas agar masih realistis untuk laptop.
MAX_SERIES_PER_STORE_CATEGORY = 2
ENCODER_LENGTH = 56
HORIZON = 28
MAX_EPOCHS = 10
BATCH_SIZE = 64
MAX_TRAIN_BATCHES = 100

pl.seed_everything(42, workers=True)

for path in [SALES_PATH, CALENDAR_PATH, LIGHTGBM_PATH]:
    if not path.exists():
        raise FileNotFoundError(f"File tidak ditemukan: {path}")

print("=" * 65)
print("TEMPORAL FUSION TRANSFORMER - M5 VALIDATION")
print("=" * 65)


# ============================================================
# 2. MEMBACA DATA DAN MEMILIH SERI PRODUK-TOKO
# ============================================================

print("\n[1/8] Membaca dataset penjualan...")

all_columns = pd.read_csv(SALES_PATH, nrows=0).columns.tolist()

day_cols = sorted(
    [c for c in all_columns if c.startswith("d_")],
    key=lambda c: int(c.split("_")[1]),
)

metadata_cols = [
    "item_id", "dept_id", "cat_id", "store_id", "state_id"
]

cutoff = len(day_cols) - HORIZON
validation_start = cutoff - HORIZON

if validation_start <= ENCODER_LENGTH + HORIZON:
    raise ValueError("Data tidak cukup untuk pelatihan dan validasi.")

sales_dtype = {c: "uint16" for c in day_cols}

sales = pd.read_csv(
    SALES_PATH,
    dtype=sales_dtype,
)

for col in metadata_cols:
    sales[col] = sales[col].astype(str)

train_days = day_cols[:cutoff]
test_days = day_cols[cutoff:]

# Seleksi berdasarkan penjualan historis sebelum periode pengujian.
selector = sales[metadata_cols].copy()
selector["_train_total"] = sales[train_days].sum(axis=1)

# Ambil maksimal dua seri terlaris per kombinasi toko-kategori.
selected_indices = (
    selector.sort_values("_train_total", ascending=False)
    .groupby(["store_id", "cat_id"], sort=False)
    .head(MAX_SERIES_PER_STORE_CATEGORY)
    .index
)

selected = sales.loc[
    selected_indices,
    metadata_cols + day_cols,
].copy()

selected["series_id"] = (
    selected["item_id"].astype(str)
    + "::"
    + selected["store_id"].astype(str)
)

if selected["series_id"].nunique() != len(selected):
    raise ValueError("ID seri produk-toko tidak unik.")

print(f"Seri yang dipilih      : {len(selected):,}")
print(f"Toko yang terwakili    : {selected['store_id'].nunique()}")
print(f"Kategori yang terwakili: {selected['cat_id'].nunique()}")
print(f"Rentang pelatihan      : {validation_start} hari pertama")
print(f"Rentang validasi       : {HORIZON} hari")
print(f"Rentang pengujian      : {HORIZON} hari")


# ============================================================
# 3. MEMBENTUK DATA PANEL DAN FITUR KALENDER
# ============================================================

print("\n[2/8] Membuat panel time series...")

calendar = pd.read_csv(
    CALENDAR_PATH,
    parse_dates=["date"],
)

calendar_cols = [
    "d",
    "date",
    "weekday",
    "month",
    "year",
    "event_name_1",
    "event_type_1",
    "event_name_2",
    "event_type_2",
    "snap_CA",
    "snap_TX",
    "snap_WI",
]

panel = selected.melt(
    id_vars=metadata_cols + ["series_id"],
    value_vars=day_cols,
    var_name="d",
    value_name="sales",
)

panel["time_idx"] = (
    panel["d"].str.replace("d_", "", regex=False).astype("int32") - 1
)

panel["sales"] = panel["sales"].astype("float32")

panel = panel.merge(
    calendar[calendar_cols],
    on="d",
    how="left",
    validate="many_to_one",
)

if panel["date"].isna().any():
    raise ValueError("Ada tanggal penjualan yang tidak cocok dengan kalender.")

# Fitur kalender harus diketahui pada tanggal yang diprediksi.
panel["day_of_year"] = panel["date"].dt.dayofyear.astype("float32")
panel["day_of_month"] = panel["date"].dt.day.astype("float32")

panel["year"] = panel["year"].astype("float32")

for col in ["snap_CA", "snap_TX", "snap_WI"]:
    panel[col] = panel[col].astype("float32")

categorical_cols = [
    *metadata_cols,
    "series_id",
    "weekday",
    "month",
    "event_name_1",
    "event_type_1",
    "event_name_2",
    "event_type_2",
]

for col in [
    "event_name_1",
    "event_type_1",
    "event_name_2",
    "event_type_2",
]:
    panel[col] = panel[col].fillna("NoEvent")

# Membentuk kategori berdasarkan seluruh nilai fitur kalender yang diketahui.
for col in categorical_cols:
    panel[col] = panel[col].astype(str).astype("category")

panel = panel.sort_values(
    ["series_id", "time_idx"]
).reset_index(drop=True)

print(f"Jumlah baris panel: {len(panel):,}")


# ============================================================
# 4. MENYIAPKAN TRAINING DAN VALIDATION
# ============================================================

print("\n[3/8] Menyiapkan TimeSeriesDataSet...")

static_categoricals = [
    "series_id",
    "item_id",
    "dept_id",
    "cat_id",
    "store_id",
    "state_id",
]

known_categoricals = [
    "weekday",
    "month",
    "event_name_1",
    "event_type_1",
    "event_name_2",
    "event_type_2",
]

known_reals = [
    "time_idx",
    "year",
    "day_of_year",
    "day_of_month",
    "snap_CA",
    "snap_TX",
    "snap_WI",
]

# Data latih berakhir sebelum data validasi.
train_data = panel.loc[
    panel["time_idx"] < validation_start
].copy()

training = TimeSeriesDataSet(
    train_data,
    time_idx="time_idx",
    target="sales",
    group_ids=["series_id"],

    min_encoder_length=ENCODER_LENGTH,
    max_encoder_length=ENCODER_LENGTH,
    min_prediction_length=HORIZON,
    max_prediction_length=HORIZON,

    static_categoricals=static_categoricals,

    time_varying_known_categoricals=known_categoricals,
    time_varying_known_reals=known_reals,

    time_varying_unknown_reals=["sales"],

    target_normalizer=GroupNormalizer(
        groups=["series_id"],
        transformation="softplus",
    ),

    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,
    allow_missing_timesteps=False,
)

# Validasi pada 28 hari yang langsung mendahului periode pengujian.
validation_source = panel.loc[
    panel["time_idx"] < cutoff
].copy()

validation = TimeSeriesDataSet.from_dataset(
    training,
    validation_source,
    predict=True,
    stop_randomization=True,
)

train_loader = training.to_dataloader(
    train=True,
    batch_size=BATCH_SIZE,
    num_workers=0,
)

validation_loader = validation.to_dataloader(
    train=False,
    batch_size=BATCH_SIZE,
    num_workers=0,
)

print(f"Window pelatihan : {len(training):,}")
print(f"Window validasi  : {len(validation):,}")


# ============================================================
# 5. MEMBANGUN DAN MELATIH TFT
# ============================================================

print("\n[4/8] Menyiapkan Temporal Fusion Transformer...")

tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=0.03,
    hidden_size=16,
    attention_head_size=2,
    dropout=0.1,
    hidden_continuous_size=8,
    output_size=7,
    loss=QuantileLoss(),
    log_interval=-1,
    reduce_on_plateau_patience=2,
)

print(f"Parameter model: {tft.size():,}")

checkpoint_dir = OUTPUT_DIR / "tft_checkpoints"
checkpoint_dir.mkdir(parents=True, exist_ok=True)

checkpoint_callback = ModelCheckpoint(
    dirpath=str(checkpoint_dir),
    filename="tft-m5-{epoch:02d}-{val_loss:.3f}",
    monitor="val_loss",
    mode="min",
    save_top_k=1,
)

early_stopping = EarlyStopping(
    monitor="val_loss",
    min_delta=0.0001,
    patience=3,
    mode="min",
)

trainer = pl.Trainer(
    max_epochs=MAX_EPOCHS,
    accelerator="cpu",
    devices=1,
    limit_train_batches=MAX_TRAIN_BATCHES,
    limit_val_batches=20,
    gradient_clip_val=0.1,
    callbacks=[checkpoint_callback, early_stopping],
    enable_checkpointing=True,
    enable_model_summary=False,
    logger=False,
    log_every_n_steps=25,
    num_sanity_val_steps=0,
)

print("\nMemulai pelatihan TFT...")
print("Ini merupakan eksperimen terbatas, bukan tuning final.")

trainer.fit(
    tft,
    train_dataloaders=train_loader,
    val_dataloaders=validation_loader,
)

if not checkpoint_callback.best_model_path:
    raise RuntimeError(
        "Checkpoint terbaik tidak terbentuk. Periksa log pelatihan."
    )

best_model_path = Path(checkpoint_callback.best_model_path)

stable_model_path = OUTPUT_DIR / "tft_m5_top_series.ckpt"
shutil.copy2(best_model_path, stable_model_path)

best_tft = TemporalFusionTransformer.load_from_checkpoint(
    str(best_model_path)
)

print(f"\nCheckpoint model tersimpan: {stable_model_path}")


# ============================================================
# 6. MEMPREDIKSI PERIODE PENGUJIAN
# ============================================================

print("\n[5/8] Membuat prediksi 28 hari...")

test_dataset = TimeSeriesDataSet.from_dataset(
    training,
    panel,
    predict=True,
    stop_randomization=True,
)

test_loader = test_dataset.to_dataloader(
    train=False,
    batch_size=BATCH_SIZE,
    num_workers=0,
)

prediction_result = best_tft.predict(
    test_loader,
    mode="prediction",
)

if hasattr(prediction_result, "output"):
    prediction_result = prediction_result.output

if isinstance(prediction_result, (tuple, list)):
    prediction_result = prediction_result[0]

if torch.is_tensor(prediction_result):
    tft_predictions = prediction_result.detach().cpu().numpy()
else:
    tft_predictions = np.asarray(prediction_result)

# Prediction mode biasanya menghasilkan nilai median.
# Penanganan berikut juga meliputi output yang masih tiga dimensi.
if tft_predictions.ndim == 3:
    if tft_predictions.shape[-1] == 7:
        tft_predictions = tft_predictions[:, :, 3]
    elif tft_predictions.shape[-1] == 1:
        tft_predictions = tft_predictions[:, :, 0]
    else:
        raise ValueError(
            f"Bentuk output TFT tidak dikenali: {tft_predictions.shape}"
        )

if tft_predictions.ndim != 2:
    raise ValueError(
        f"Prediksi seharusnya matriks dua dimensi: {tft_predictions.shape}"
    )

tft_predictions = np.clip(tft_predictions, 0, None)

# Urutan sampel prediksi berasal dari index dataset, bukan tebakan urutan CSV.
prediction_index = test_dataset.decoded_index.reset_index(drop=True)

if "series_id" not in prediction_index.columns:
    raise ValueError(
        "series_id tidak ditemukan pada decoded_index. "
        f"Kolom yang ditemukan: {prediction_index.columns.tolist()}"
    )

series_order = prediction_index["series_id"].astype(str).tolist()

if len(series_order) != tft_predictions.shape[0]:
    raise ValueError(
        "Jumlah seri pada output prediksi berbeda dengan index dataset."
    )

selected_by_id = selected.set_index("series_id")

missing_series = set(series_order) - set(selected_by_id.index)

if missing_series:
    raise ValueError(
        "Ada ID seri hasil decoding yang tidak cocok dengan data sumber."
    )

if tft_predictions.shape[1] != HORIZON:
    raise ValueError(
        f"Horizon output {tft_predictions.shape[1]}, "
        f"seharusnya {HORIZON}."
    )


# ============================================================
# 7. PERBANDINGAN TIGA MODEL PADA SAMPEL YANG SAMA
# ============================================================

print("\n[6/8] Membandingkan TFT, LightGBM, dan Seasonal Naive...")

test_dates = pd.to_datetime(
    calendar.set_index("d").loc[test_days, "date"]
).reset_index(drop=True)

actual = selected_by_id.loc[
    series_order, test_days
].to_numpy(dtype=float)

# Baseline mengulang penjualan tujuh hari terakhir sebelum test.
last_week = selected_by_id.loc[
    series_order, train_days[-7:]
].to_numpy(dtype=float)

seasonal_naive = np.tile(last_week, (1, 4))

# Ambil prediksi LightGBM untuk seri yang sama.
lgbm = pd.read_csv(LIGHTGBM_PATH)

lgbm["item_id"] = lgbm["item_id"].astype(str)
lgbm["store_id"] = lgbm["store_id"].astype(str)

lgbm["series_id"] = (
    lgbm["item_id"] + "::" + lgbm["store_id"]
)

lgbm_forecast_cols = sorted(
    [c for c in lgbm.columns if c.startswith("forecast_")],
    key=lambda c: pd.Timestamp(c.replace("forecast_", "")),
)

if len(lgbm_forecast_cols) != HORIZON:
    raise ValueError(
        "Jumlah kolom prediksi LightGBM berbeda dari horizon TFT."
    )

lgbm_by_id = lgbm.drop_duplicates(
    "series_id"
).set_index("series_id")

missing_lgbm = set(series_order) - set(lgbm_by_id.index)

if missing_lgbm:
    raise ValueError(
        "Ada seri TFT yang tidak ditemukan di prediksi LightGBM."
    )

lightgbm_predictions = lgbm_by_id.loc[
    series_order, lgbm_forecast_cols
].to_numpy(dtype=float)


def calculate_metrics(name, actual_values, predicted_values):
    residuals = actual_values - predicted_values
    absolute_errors = np.abs(residuals)

    denominator = actual_values.sum()
    wape = (
        100 * absolute_errors.sum() / denominator
        if denominator > 0
        else float("nan")
    )

    return {
        "model": name,
        "WAPE_percent": wape,
        "MAE_units": absolute_errors.mean(),
        "RMSE_units": np.sqrt(np.mean(residuals ** 2)),
        "number_of_series": actual_values.shape[0],
        "test_days": actual_values.shape[1],
    }


metrics = pd.DataFrame([
    calculate_metrics("TFT", actual, tft_predictions),
    calculate_metrics("LightGBM", actual, lightgbm_predictions),
    calculate_metrics("Seasonal Naive", actual, seasonal_naive),
]).sort_values("WAPE_percent").reset_index(drop=True)

metrics_path = REPORT_DIR / "tft_sample_metrics.csv"
metrics.to_csv(metrics_path, index=False)

# Susun prediksi dalam format panjang untuk analisis selanjutnya.
records = []

for i, series_id in enumerate(series_order):
    meta = selected_by_id.loc[series_id]

    for j, forecast_date in enumerate(test_dates):
        records.append({
            "date": forecast_date,
            "series_id": series_id,
            "item_id": meta["item_id"],
            "store_id": meta["store_id"],
            "state_id": meta["state_id"],
            "cat_id": meta["cat_id"],
            "actual_units": actual[i, j],
            "tft_forecast": tft_predictions[i, j],
            "lightgbm_forecast": lightgbm_predictions[i, j],
            "seasonal_naive_forecast": seasonal_naive[i, j],
        })

prediction_df = pd.DataFrame(records)

predictions_path = REPORT_DIR / "tft_sample_predictions.csv"
prediction_df.to_csv(predictions_path, index=False)

# Grafik total harian pada sampel 60 seri.
daily = (
    prediction_df.groupby("date", as_index=False)
    .agg(
        actual_units=("actual_units", "sum"),
        tft_forecast=("tft_forecast", "sum"),
        lightgbm_forecast=("lightgbm_forecast", "sum"),
        seasonal_naive_forecast=("seasonal_naive_forecast", "sum"),
    )
)

plt.figure(figsize=(13, 6))

plt.plot(
    daily["date"],
    daily["actual_units"],
    label="Actual",
    linewidth=2,
)

plt.plot(
    daily["date"],
    daily["tft_forecast"],
    label="Temporal Fusion Transformer",
    linewidth=1.8,
)

plt.plot(
    daily["date"],
    daily["lightgbm_forecast"],
    label="LightGBM",
    linewidth=1.8,
)

plt.plot(
    daily["date"],
    daily["seasonal_naive_forecast"],
    label="Seasonal Naive",
    linestyle="--",
)

plt.title("TFT Validation - Selected Product-Store Series")
plt.xlabel("Date")
plt.ylabel("Total Units Sold in Sample")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

chart_path = OUTPUT_DIR / "tft_sample_validation.png"
plt.savefig(chart_path, dpi=150)
plt.close()


# ============================================================
# 8. RINGKASAN HASIL
# ============================================================

print("\n" + "=" * 65)
print("TFT VALIDATION SUMMARY")
print("=" * 65)

print(metrics.round(4).to_string(index=False))

print("\n--- FILE HASIL ---")
print(f"Metrik tiga model : {metrics_path}")
print(f"Prediksi detail   : {predictions_path}")
print(f"Checkpoint TFT    : {stable_model_path}")
print(f"Grafik            : {chart_path}")

print("\nTFT VALIDATION SELESAI!")
