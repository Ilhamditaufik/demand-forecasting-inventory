
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ortools.sat.python import cp_model


# =========================================================
# 1. KONFIGURASI PROYEK DAN ASUMSI SIMULASI
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "raw"
REPORT_DIR = BASE_DIR / "reports"
OUTPUT_DIR = BASE_DIR / "outputs"

REPORT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FORECAST_PATH = REPORT_DIR / "lightgbm_bottom_level_forecasts.csv.gz"
PRICE_PATH = DATA_DIR / "sell_prices.csv"
CALENDAR_PATH = DATA_DIR / "calendar.csv"

# Periode perencanaan: tujuh hari pertama dari hasil backtest.
PLANNING_HORIZON_DAYS = 7

# Asumsi demo, bukan nilai aktual dari Walmart.
ASSUMED_INITIAL_STOCK_RATIO = 0.20
ASSUMED_PURCHASE_COST_RATIO = 0.70
ASSUMED_HOLDING_COST_RATIO = 0.02
ASSUMED_STOCKOUT_FIXED_PENALTY = 0.25

PURCHASE_BUDGET_SHARE = 0.70
STORAGE_CAPACITY_SHARE = 0.75

MAX_SOLVER_SECONDS = 90
NUM_WORKERS = 8

for path in [FORECAST_PATH, PRICE_PATH, CALENDAR_PATH]:
    if not path.exists():
        raise FileNotFoundError(f"File tidak ditemukan: {path}")

print("=" * 68)
print("INVENTORY OPTIMIZATION - OR-TOOLS CP-SAT")
print("=" * 68)


# =========================================================
# 2. BACA PREDIKSI PRODUK-TOKO
# =========================================================

print("\n[1/7] Membaca prediksi LightGBM...")

df = pd.read_csv(FORECAST_PATH)

metadata_cols = [
    "item_id",
    "dept_id",
    "cat_id",
    "store_id",
    "state_id",
]

missing_metadata = set(metadata_cols) - set(df.columns)

if missing_metadata:
    raise ValueError(
        f"Kolom metadata tidak ditemukan: {missing_metadata}"
    )

forecast_cols = sorted(
    [c for c in df.columns if c.startswith("forecast_")],
    key=lambda c: pd.Timestamp(c.replace("forecast_", "")),
)

if len(forecast_cols) < PLANNING_HORIZON_DAYS:
    raise ValueError(
        "Data forecast kurang dari tujuh hari."
    )

planning_cols = forecast_cols[:PLANNING_HORIZON_DAYS]

forecast_dates = pd.to_datetime([
    c.replace("forecast_", "") for c in planning_cols
])

if df[planning_cols].isna().any().any():
    raise ValueError("Prediksi memuat nilai kosong.")

if (df[planning_cols] < 0).any().any():
    raise ValueError(
        "Prediksi negatif ditemukan. Periksa kembali model forecasting."
    )

# Total permintaan tujuh hari dibulatkan ke unit barang.
demand_float = df[planning_cols].sum(axis=1).to_numpy(dtype=float)
demand = np.floor(demand_float + 0.5).astype(np.int64)

df["planning_demand_7d"] = demand

print(f"Seri produk-toko : {len(df):,}")
print(f"Tanggal mulai    : {forecast_dates.min().date()}")
print(f"Tanggal selesai  : {forecast_dates.max().date()}")
print(f"Total permintaan : {demand.sum():,} unit")


# =========================================================
# 3. CARI HARGA JUAL TERBARU YANG TERSEDIA
# =========================================================

print("\n[2/7] Menyiapkan harga produk...")

calendar = pd.read_csv(
    CALENDAR_PATH,
    parse_dates=["date"],
)

planning_end = forecast_dates.max()

eligible_calendar = (
    calendar.loc[
        calendar["date"] <= planning_end,
        ["date", "wm_yr_wk"],
    ]
    .sort_values("date")
)

# Ambil empat minggu harga terakhir yang tersedia sampai
# tanggal akhir periode perencanaan.
recent_weeks = (
    eligible_calendar
    .drop_duplicates("wm_yr_wk", keep="last")
    .tail(4)["wm_yr_wk"]
    .astype(int)
    .tolist()
)

if not recent_weeks:
    raise ValueError("Minggu harga tidak ditemukan di kalender.")

print(f"Minggu harga yang diperiksa: {recent_weeks}")
print("Membaca harga per potongan untuk menghemat memori...")

price_parts = []

for chunk in pd.read_csv(
    PRICE_PATH,
    usecols=["store_id", "item_id", "wm_yr_wk", "sell_price"],
    dtype={
        "store_id": "string",
        "item_id": "string",
        "wm_yr_wk": "int32",
        "sell_price": "float32",
    },
    chunksize=500_000,
):
    matched = chunk.loc[
        chunk["wm_yr_wk"].isin(recent_weeks)
    ]

    if not matched.empty:
        price_parts.append(matched)

if price_parts:
    prices = pd.concat(price_parts, ignore_index=True)

    prices = (
        prices.sort_values("wm_yr_wk")
        .drop_duplicates(
            subset=["store_id", "item_id"],
            keep="last",
        )
    )
else:
    prices = pd.DataFrame(
        columns=["store_id", "item_id", "wm_yr_wk", "sell_price"]
    )

prices = prices[
    ["store_id", "item_id", "sell_price"]
].copy()

# Pastikan tipe kunci join konsisten.
df["store_id"] = df["store_id"].astype("string")
df["item_id"] = df["item_id"].astype("string")
df["cat_id"] = df["cat_id"].astype("string")

df = df.merge(
    prices,
    on=["store_id", "item_id"],
    how="left",
    validate="one_to_one",
)

missing_price = df["sell_price"].isna()
missing_price_count = int(missing_price.sum())

# Untuk produk tanpa harga pada minggu tersebut, gunakan median
# kategori, kemudian median semua harga yang tersedia.
category_medians = df.groupby(
    "cat_id", dropna=False
)["sell_price"].median()

df["sell_price"] = df["sell_price"].fillna(
    df["cat_id"].map(category_medians)
)

overall_median = df["sell_price"].median()

if pd.isna(overall_median):
    overall_median = 1.0

df["sell_price"] = (
    df["sell_price"]
    .fillna(overall_median)
    .clip(lower=0.01)
)

df["price_was_missing"] = missing_price.astype(int)

print(f"Produk-toko tanpa harga langsung: {missing_price_count:,}")
print("Harga pengganti menggunakan median kategori/global.")
print("Harga jual dipakai sebagai proxy nilai barang, bukan biaya pemasok aktual.")


# =========================================================
# 4. BANGUN SKENARIO INVENTARIS DAN BATASAN
# =========================================================

print("\n[3/7] Menyiapkan variabel optimasi...")

# Semua nilai uang diubah menjadi integer sen agar kompatibel
# dengan CP-SAT.
retail_price_cents = np.maximum(
    1,
    np.rint(df["sell_price"].to_numpy(dtype=float) * 100),
).astype(np.int64)

unit_purchase_cost = np.maximum(
    1,
    np.rint(
        retail_price_cents * ASSUMED_PURCHASE_COST_RATIO
    ),
).astype(np.int64)

holding_cost = np.maximum(
    1,
    np.rint(
        retail_price_cents * ASSUMED_HOLDING_COST_RATIO
    ),
).astype(np.int64)

stockout_penalty = np.maximum(
    unit_purchase_cost + 1,
    retail_price_cents
    + int(round(ASSUMED_STOCKOUT_FIXED_PENALTY * 100)),
).astype(np.int64)

# Stok awal disimulasikan sebesar 20% dari permintaan tujuh hari.
# Ini bukan stok fisik Walmart.
initial_stock = np.floor(
    demand * ASSUMED_INITIAL_STOCK_RATIO
).astype(np.int64)

# Jumlah unit yang dibutuhkan jika seluruh permintaan dapat dipenuhi.
required_order = np.maximum(demand - initial_stock, 0)

# Memungkinkan pemesanan hingga 125% permintaan untuk
# merepresentasikan kemungkinan persediaan berlebih.
max_order = np.maximum(
    np.ceil(demand * 1.25).astype(np.int64) - initial_stock,
    0,
)

full_replenishment_cost = int(
    np.dot(required_order, unit_purchase_cost)
)

budget_limit = int(
    full_replenishment_cost * PURCHASE_BUDGET_SHARE
)

storage_capacity = int(
    required_order.sum() * STORAGE_CAPACITY_SHARE
)

print(f"Permintaan             : {demand.sum():,} unit")
print(f"Stok awal asumsi       : {initial_stock.sum():,} unit")
print(f"Kebutuhan pemesanan    : {required_order.sum():,} unit")
print(f"Anggaran simulasi      : ${budget_limit / 100:,.2f}")
print(f"Kapasitas pemesanan    : {storage_capacity:,} unit")


# =========================================================
# 5. DEFINISIKAN MODEL CP-SAT
# =========================================================

print("\n[4/7] Membuat model optimasi OR-Tools...")

model = cp_model.CpModel()

order_vars = []
served_vars = []
lost_vars = []
ending_vars = []

n = len(df)

for i in range(n):
    q = model.NewIntVar(
        0,
        int(max_order[i]),
        f"order_{i}",
    )

    served = model.NewIntVar(
        0,
        int(demand[i]),
        f"served_{i}",
    )

    lost = model.NewIntVar(
        0,
        int(demand[i]),
        f"lost_{i}",
    )

    ending = model.NewIntVar(
        0,
        int(initial_stock[i] + max_order[i]),
        f"ending_{i}",
    )

    # Semua permintaan harus tercatat sebagai terpenuhi atau
    # sebagai potensi kehilangan penjualan.
    model.Add(served + lost == int(demand[i]))

    # Neraca persediaan.
    model.Add(
        int(initial_stock[i]) + q == served + ending
    )

    order_vars.append(q)
    served_vars.append(served)
    lost_vars.append(lost)
    ending_vars.append(ending)

# Batasan kapasitas pemesanan.
model.Add(sum(order_vars) <= storage_capacity)

# Batasan anggaran pengadaan.
model.Add(
    sum(
        order_vars[i] * int(unit_purchase_cost[i])
        for i in range(n)
    )
    <= budget_limit
)

# Tujuan: meminimalkan biaya pembelian, biaya penyimpanan,
# dan penalti potensi kehilangan penjualan.
objective_terms = []

for i in range(n):
    objective_terms.append(
        order_vars[i] * int(unit_purchase_cost[i])
        + ending_vars[i] * int(holding_cost[i])
        + lost_vars[i] * int(stockout_penalty[i])
    )

model.Minimize(sum(objective_terms))


# =========================================================
# 6. JALANKAN SOLVER
# =========================================================

print("\n[5/7] Menjalankan solver...")
print(
    f"Batas waktu solver: {MAX_SOLVER_SECONDS} detik. "
    "Tunggu sampai hasil ditampilkan."
)

solver = cp_model.CpSolver()
solver.parameters.max_time_in_seconds = MAX_SOLVER_SECONDS
solver.parameters.num_search_workers = NUM_WORKERS

status = solver.Solve(model)

if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
    raise RuntimeError(
        "Solver tidak menemukan solusi yang layak. "
        f"Status: {solver.StatusName(status)}"
    )

print(f"Status solver: {solver.StatusName(status)}")

if status == cp_model.FEASIBLE:
    print(
        "Catatan: solusi layak ditemukan, tetapi belum terbukti "
        "sebagai solusi optimal dalam batas waktu."
    )


# =========================================================
# 7. EKSTRAK HASIL DAN SIMPAN LAPORAN
# =========================================================

print("\n[6/7] Menyimpan rekomendasi inventaris...")

orders = np.array(
    [solver.Value(v) for v in order_vars],
    dtype=np.int64,
)

served = np.array(
    [solver.Value(v) for v in served_vars],
    dtype=np.int64,
)

lost = np.array(
    [solver.Value(v) for v in lost_vars],
    dtype=np.int64,
)

ending = np.array(
    [solver.Value(v) for v in ending_vars],
    dtype=np.int64,
)

df["assumed_initial_stock"] = initial_stock
df["recommended_order_units"] = orders
df["expected_fulfilled_units"] = served
df["expected_lost_sales_units"] = lost
df["expected_ending_inventory_units"] = ending

df["estimated_unit_purchase_cost"] = unit_purchase_cost / 100
df["estimated_holding_cost_per_unit"] = holding_cost / 100
df["estimated_stockout_penalty_per_unit"] = (
    stockout_penalty / 100
)

df["estimated_purchase_cost"] = (
    orders * unit_purchase_cost / 100
)

df["estimated_holding_cost"] = ending * holding_cost / 100

df["estimated_stockout_penalty"] = (
    lost * stockout_penalty / 100
)

recommendation_cols = [
    "item_id",
    "dept_id",
    "cat_id",
    "store_id",
    "state_id",
    "sell_price",
    "price_was_missing",
    "planning_demand_7d",
    "assumed_initial_stock",
    "recommended_order_units",
    "expected_fulfilled_units",
    "expected_lost_sales_units",
    "expected_ending_inventory_units",
    "estimated_unit_purchase_cost",
    "estimated_purchase_cost",
    "estimated_holding_cost",
    "estimated_stockout_penalty",
]

recommendations = df[recommendation_cols].copy()

recommendation_path = (
    REPORT_DIR / "inventory_optimization_recommendations.csv"
)

recommendations.to_csv(recommendation_path, index=False)

# Ringkasan per toko dan state untuk dashboard.
store_summary = (
    df.groupby(
        ["state_id", "store_id"],
        as_index=False,
        observed=True,
    )
    .agg(
        forecast_demand_units=("planning_demand_7d", "sum"),
        assumed_initial_stock=("assumed_initial_stock", "sum"),
        recommended_order_units=("recommended_order_units", "sum"),
        fulfilled_units=("expected_fulfilled_units", "sum"),
        lost_sales_units=("expected_lost_sales_units", "sum"),
        ending_inventory_units=("expected_ending_inventory_units", "sum"),
        estimated_purchase_cost=("estimated_purchase_cost", "sum"),
        estimated_holding_cost=("estimated_holding_cost", "sum"),
        estimated_stockout_penalty=("estimated_stockout_penalty", "sum"),
    )
)

store_summary.to_csv(
    REPORT_DIR / "inventory_summary_by_store.csv",
    index=False,
)

# Ringkasan keseluruhan.
total_demand = int(demand.sum())
total_fulfilled = int(served.sum())
total_lost = int(lost.sum())
total_order = int(orders.sum())
total_ending = int(ending.sum())

actual_purchase_cost = float(
    np.dot(orders, unit_purchase_cost) / 100
)

actual_holding_cost = float(
    np.dot(ending, holding_cost) / 100
)

actual_stockout_penalty = float(
    np.dot(lost, stockout_penalty) / 100
)

lost_rate = (
    100 * total_lost / total_demand
    if total_demand > 0
    else 0.0
)

summary_rows = [
    ("solver_status", solver.StatusName(status)),
    ("planning_horizon_days", PLANNING_HORIZON_DAYS),
    ("planning_start_date", forecast_dates.min().date().isoformat()),
    ("planning_end_date", forecast_dates.max().date().isoformat()),
    ("number_of_product_store_series", n),
    ("forecast_demand_units", total_demand),
    ("assumed_initial_stock_units", int(initial_stock.sum())),
    ("recommended_order_units", total_order),
    ("fulfilled_units", total_fulfilled),
    ("potential_lost_sales_units", total_lost),
    ("ending_inventory_units", total_ending),
    ("potential_lost_sales_rate_pct", lost_rate),
    ("purchase_budget_limit", budget_limit / 100),
    ("storage_capacity_units", storage_capacity),
    ("estimated_purchase_cost", actual_purchase_cost),
    ("estimated_holding_cost", actual_holding_cost),
    ("estimated_stockout_penalty", actual_stockout_penalty),
    ("objective_cost", solver.ObjectiveValue() / 100),
]

summary = pd.DataFrame(summary_rows, columns=["metric", "value"])

summary.to_csv(
    REPORT_DIR / "inventory_optimization_summary.csv",
    index=False,
)

# Grafik 15 produk-toko dengan permintaan prediksi tertinggi.
top = (
    recommendations.sort_values(
        "planning_demand_7d",
        ascending=False,
    )
    .head(15)
    .copy()
)

top["series_label"] = (
    top["item_id"].astype(str)
    + " / "
    + top["store_id"].astype(str)
)

top = top.sort_values("planning_demand_7d")

plt.figure(figsize=(12, 7))

positions = np.arange(len(top))

plt.barh(
    positions,
    top["planning_demand_7d"],
    label="Forecast Demand",
)

plt.barh(
    positions,
    top["recommended_order_units"],
    label="Recommended Order",
)

plt.yticks(positions, top["series_label"])
plt.xlabel("Units over 7-day planning horizon")
plt.title("Top 15 Demand Series: Forecast vs Order Recommendation")
plt.legend()
plt.grid(axis="x", alpha=0.25)
plt.tight_layout()

chart_path = OUTPUT_DIR / "inventory_optimization_top15.png"
plt.savefig(chart_path, dpi=150)
plt.close()


# =========================================================
# HASIL AKHIR
# =========================================================

print("\n" + "=" * 68)
print("INVENTORY OPTIMIZATION SUMMARY")
print("=" * 68)

print(f"Permintaan prediksi    : {total_demand:,} unit")
print(f"Stok awal asumsi       : {initial_stock.sum():,} unit")
print(f"Rekomendasi pemesanan  : {total_order:,} unit")
print(f"Permintaan terpenuhi   : {total_fulfilled:,} unit")
print(f"Potensi lost sales     : {total_lost:,} unit")
print(f"Potensi lost-sales rate: {lost_rate:.2f}%")
print(f"Persediaan akhir       : {total_ending:,} unit")
print(f"Estimasi biaya beli    : ${actual_purchase_cost:,.2f}")
print(f"Estimasi holding cost : ${actual_holding_cost:,.2f}")
print(f"Estimasi penalti       : ${actual_stockout_penalty:,.2f}")

print("\n--- FILE HASIL ---")
print(f"Rekomendasi per produk : {recommendation_path}")
print(f"Ringkasan per toko     : {REPORT_DIR / 'inventory_summary_by_store.csv'}")
print(f"Ringkasan keseluruhan  : {REPORT_DIR / 'inventory_optimization_summary.csv'}")
print(f"Grafik                 : {chart_path}")

print("\nINVENTORY OPTIMIZATION SELESAI!")
