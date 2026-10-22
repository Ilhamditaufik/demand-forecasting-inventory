
from pathlib import Path
import sys

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "raw"
REPORT_DIR = BASE_DIR / "reports"
OUTPUT_DIR = BASE_DIR / "outputs"

passed = 0
failed = 0
warnings = 0


def check(condition, message):
    global passed, failed

    if condition:
        passed += 1
        print(f"[PASS] {message}")
    else:
        failed += 1
        print(f"[FAIL] {message}")


def warn(message):
    global warnings
    warnings += 1
    print(f"[WARN] {message}")


def read_csv(path, **kwargs):
    if not path.exists():
        raise FileNotFoundError(f"File tidak ditemukan: {path}")
    return pd.read_csv(path, **kwargs)


print("=" * 68)
print("PROJECT QUALITY ASSURANCE")
print("Demand Forecasting & Inventory Optimization")
print("=" * 68)

# 1. Pemeriksaan file utama
print("\n[1] FILE DAN ARTIFAK")

required_files = [
    DATA_DIR / "sales_train_validation.csv",
    DATA_DIR / "calendar.csv",
    DATA_DIR / "sell_prices.csv",
    REPORT_DIR / "baseline_metrics.csv",
    REPORT_DIR / "lightgbm_metrics.csv",
    REPORT_DIR / "national_aggregate_model_comparison.csv",
    REPORT_DIR / "prophet_interval_metrics.csv",
    REPORT_DIR / "hierarchical_reconciled_wide.csv",
    REPORT_DIR / "inventory_optimization_recommendations.csv",
    REPORT_DIR / "inventory_optimization_summary.csv",
    REPORT_DIR / "inventory_summary_by_store.csv",
    REPORT_DIR / "tft_sample_metrics.csv",
    REPORT_DIR / "tft_sample_predictions.csv",
    BASE_DIR / "app.py",
]

for path in required_files:
    check(path.exists(), f"Artefak tersedia: {path.relative_to(BASE_DIR)}")

if failed:
    print("\nAda artefak yang belum ditemukan. Lengkapi dahulu sebelum melanjutkan.")

# 2. Perbandingan model nasional
print("\n[2] EVALUASI FORECASTING")

try:
    models = read_csv(
        REPORT_DIR / "national_aggregate_model_comparison.csv"
    )

    expected_models = {"Seasonal Naive", "LightGBM", "Prophet"}

    check(
        expected_models.issubset(set(models["model"])),
        "Perbandingan memuat tiga model nasional",
    )

    metrics = ["WAPE_percent", "MAE_units", "RMSE_units"]

    check(
        models[metrics].notna().all().all()
        and np.isfinite(models[metrics].to_numpy()).all(),
        "Metrik model nasional tidak kosong atau tak terhingga",
    )

    check(
        (models[metrics] >= 0).all().all(),
        "Semua metrik kesalahan bernilai nonnegatif",
    )

    best_wape = models.loc[
        models["WAPE_percent"].idxmin()
    ]

    print(
        f"Model dengan WAPE terbaik pada evaluasi nasional: "
        f"{best_wape['model']} ({best_wape['WAPE_percent']:.2f}%)"
    )

except Exception as exc:
    check(False, f"Evaluasi model nasional: {exc}")

# 3. Evaluasi TFT
print("\n[3] TFT")

try:
    tft_metrics = read_csv(REPORT_DIR / "tft_sample_metrics.csv")
    tft_predictions = read_csv(
        REPORT_DIR / "tft_sample_predictions.csv"
    )

    check(
        {"TFT", "LightGBM", "Seasonal Naive"}.issubset(
            set(tft_metrics["model"])
        ),
        "TFT dibandingkan dengan dua baseline pada sampel",
    )

    check(
        tft_predictions["series_id"].nunique() == 60,
        "Dataset pengujian TFT mencakup 60 seri",
    )

    check(
        len(tft_predictions) == 60 * 28,
        "Dataset pengujian TFT mencakup 28 hari untuk setiap seri",
    )

    forecast_columns = [
        "actual_units",
        "tft_forecast",
        "lightgbm_forecast",
        "seasonal_naive_forecast",
    ]

    check(
        tft_predictions[forecast_columns].notna().all().all(),
        "Prediksi TFT dan pembanding tidak memiliki nilai kosong",
    )

    check(
        (tft_predictions[forecast_columns] >= 0).all().all(),
        "Aktual dan prediksi sampel tidak bernilai negatif",
    )

    warn(
        "Metrik TFT hanya berlaku untuk 60 seri sampel; "
        "jangan bandingkan langsung dengan WAPE nasional."
    )

except Exception as exc:
    check(False, f"Validasi TFT: {exc}")

# 4. Konsistensi rekonsiliasi hierarki
print("\n[4] HIERARCHICAL RECONCILIATION")

try:
    hierarchy = read_csv(
        REPORT_DIR / "hierarchical_reconciled_wide.csv"
    )

    required_nodes = {"CA", "TX", "WI", "Total"}

    check(
        required_nodes.issubset(set(hierarchy.columns)),
        "Total CA, TX, WI, dan nasional tersedia",
    )

    state_max_error = 0.0

    for state in ["CA", "TX", "WI"]:
        store_cols = [
            col for col in hierarchy.columns
            if col.startswith(state + "_")
            and col.rsplit("_", 1)[-1].isdigit()
        ]

        if not store_cols:
            check(False, f"Toko tersedia untuk state {state}")
            continue

        error = (
            hierarchy[state] - hierarchy[store_cols].sum(axis=1)
        ).abs().max()

        state_max_error = max(state_max_error, float(error))

    national_error = (
        hierarchy["Total"]
        - hierarchy[["CA", "TX", "WI"]].sum(axis=1)
    ).abs().max()

    check(
        state_max_error <= 1e-3,
        "Total toko konsisten dengan total state",
    )

    check(
        float(national_error) <= 1e-3,
        "Total state konsisten dengan total nasional",
    )

except Exception as exc:
    check(False, f"Validasi hierarki: {exc}")

# 5. Konsistensi inventaris
print("\n[5] INVENTORY OPTIMIZATION")

try:
    inv = read_csv(
        REPORT_DIR / "inventory_optimization_recommendations.csv"
    )

    summary = read_csv(
        REPORT_DIR / "inventory_optimization_summary.csv",
        dtype=str,
    )

    inv_numeric = [
        "planning_demand_7d",
        "assumed_initial_stock",
        "recommended_order_units",
        "expected_fulfilled_units",
        "expected_lost_sales_units",
        "expected_ending_inventory_units",
    ]

    check(
        inv[inv_numeric].notna().all().all(),
        "Kolom penting inventaris terisi",
    )

    check(
        (inv[inv_numeric] >= 0).all().all(),
        "Jumlah permintaan, pesanan, dan stok tidak negatif",
    )

    demand_error = (
        inv["planning_demand_7d"]
        - inv["expected_fulfilled_units"]
        - inv["expected_lost_sales_units"]
    ).abs().max()

    stock_error = (
        inv["assumed_initial_stock"]
        + inv["recommended_order_units"]
        - inv["expected_fulfilled_units"]
        - inv["expected_ending_inventory_units"]
    ).abs().max()

    check(
        float(demand_error) <= 1e-6,
        "Permintaan = terpenuhi + potensi lost sales",
    )

    check(
        float(stock_error) <= 1e-6,
        "Stok awal + pesanan = terpenuhi + stok akhir",
    )

    # Bandingkan agregat rekomendasi dengan ringkasan solver.
    inv_summary = dict(zip(summary["metric"], summary["value"]))

    for metric, col in [
        ("forecast_demand_units", "planning_demand_7d"),
        ("recommended_order_units", "recommended_order_units"),
        ("fulfilled_units", "expected_fulfilled_units"),
        ("potential_lost_sales_units", "expected_lost_sales_units"),
        ("ending_inventory_units", "expected_ending_inventory_units"),
    ]:
        recorded = float(inv_summary[metric])
        computed = float(inv[col].sum())

        check(
            abs(recorded - computed) <= 1e-6,
            f"Agregat inventaris konsisten: {metric}",
        )

    budget_limit = float(inv_summary["purchase_budget_limit"])
    purchase_cost = float(inv_summary["estimated_purchase_cost"])
    storage_limit = float(inv_summary["storage_capacity_units"])

    check(
        purchase_cost <= budget_limit + 0.01,
        "Estimasi biaya pembelian tidak melebihi anggaran",
    )

    check(
        inv["recommended_order_units"].sum()
        <= storage_limit + 1e-6,
        "Jumlah pesanan tidak melebihi kapasitas",
    )

    warn(
        "Biaya, stok awal, anggaran, penalti, dan kapasitas "
        "merupakan asumsi simulasi, bukan data operasional Walmart."
    )

except Exception as exc:
    check(False, f"Validasi inventaris: {exc}")

# 6. Pemeriksaan interval prediksi
print("\n[6] PREDICTION INTERVALS")

try:
    interval = read_csv(
        REPORT_DIR / "prophet_interval_evaluation.csv"
    )

    interval_cols = [
        "actual_units",
        "interval_lower",
        "interval_upper",
    ]

    check(
        interval[interval_cols].notna().all().all(),
        "Data interval tidak memiliki nilai kosong",
    )

    check(
        (
            interval["interval_lower"]
            <= interval["interval_upper"]
        ).all(),
        "Batas bawah interval tidak melebihi batas atas",
    )

    coverage = (
        interval["covered"].astype(str).str.lower()
        .isin(["true", "1"])
        .mean()
        * 100
    )

    print(f"Cakupan empiris interval: {coverage:.2f}%")

    if coverage < 80:
        warn(
            "Cakupan interval di bawah target nominal 80%; "
            "kalibrasi dan validasi lebih panjang masih diperlukan."
        )

except Exception as exc:
    check(False, f"Validasi interval: {exc}")

# 7. Ringkasan
print("\n" + "=" * 68)
print("QUALITY ASSURANCE SUMMARY")
print("=" * 68)
print(f"PASS     : {passed}")
print(f"FAIL     : {failed}")
print(f"WARNING  : {warnings}")

if failed:
    print("\nPROJECT QUALITY CHECK: PERLU PERBAIKAN")
    sys.exit(1)

print("\nPROJECT QUALITY CHECK: SEMUA PEMERIKSAAN WAJIB LULUS")
print("Catatan warning tetap harus ditangani dan didokumentasikan.")
