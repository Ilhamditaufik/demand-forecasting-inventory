
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

# Lokasi proyek dan dataset
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "raw"
OUTPUT_DIR = BASE_DIR / "outputs"
REPORT_DIR = BASE_DIR / "reports"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

sales_path = DATA_DIR / "sales_train_validation.csv"
calendar_path = DATA_DIR / "calendar.csv"

# 1. Pastikan file tersedia
for path in [sales_path, calendar_path]:
    if not path.exists():
        raise FileNotFoundError(f"File tidak ditemukan: {path}")

print("=" * 55)
print("EDA - NATIONWIDE DEMAND FORECASTING")
print("=" * 55)

# 2. Baca nama kolom terlebih dahulu
columns = pd.read_csv(sales_path, nrows=0).columns.tolist()

# Kolom harian penjualan: d_1, d_2, ..., d_1913
day_cols = [col for col in columns if col.startswith("d_")]
metadata_cols = [col for col in columns if not col.startswith("d_")]

print(f"\nJumlah kolom metadata : {len(metadata_cols)}")
print(f"Jumlah kolom harian  : {len(day_cols)}")

# Gunakan tipe integer yang lebih hemat memori untuk penjualan
sales_dtype = {col: "uint16" for col in day_cols}

print("\nMembaca data penjualan. Mohon tunggu sampai proses selesai...")

sales = pd.read_csv(sales_path, dtype=sales_dtype)

# 3. Ringkasan struktur data
print("\n--- STRUKTUR DATA ---")
print(f"Jumlah baris              : {len(sales):,}")
print(f"Jumlah kolom keseluruhan  : {len(sales.columns):,}")
print(f"Jumlah produk unik        : {sales['item_id'].nunique():,}")
print(f"Jumlah toko unik          : {sales['store_id'].nunique():,}")
print(f"Jumlah kategori           : {sales['cat_id'].nunique():,}")
print(f"Jumlah departemen         : {sales['dept_id'].nunique():,}")
print(f"Jumlah state              : {sales['state_id'].nunique():,}")

# 4. Ringkasan data penjualan
daily_totals = sales[day_cols].sum(axis=0).astype("int64")

print("\n--- RINGKASAN PENJUALAN ---")
print(f"Total unit terjual sepanjang periode: {daily_totals.sum():,}")
print(f"Rata-rata unit terjual per hari     : {daily_totals.mean():,.2f}")
print(f"Penjualan harian tertinggi          : {daily_totals.max():,}")
print(f"Penjualan harian terendah           : {daily_totals.min():,}")

# 5. Hubungkan kolom penjualan dengan tanggal kalender
calendar = pd.read_csv(calendar_path, parse_dates=["date"])
calendar_lookup = calendar.set_index("d")

missing_days = set(day_cols) - set(calendar_lookup.index)

if missing_days:
    raise ValueError(
        f"Ada kolom hari yang tidak ditemukan dalam kalender: "
        f"{list(missing_days)[:5]}"
    )

daily_summary = pd.DataFrame({
    "date": calendar_lookup.loc[day_cols, "date"].to_numpy(),
    "total_units_sold": daily_totals.to_numpy(),
})

daily_summary = daily_summary.sort_values("date")

# Simpan ringkasan penjualan harian
daily_summary.to_csv(
    REPORT_DIR / "daily_sales_summary.csv",
    index=False,
)

# 6. Agregasi penjualan per bulan
monthly_summary = (
    daily_summary.set_index("date")["total_units_sold"]
    .resample("MS")
    .sum()
    .reset_index()
)

monthly_summary.to_csv(
    REPORT_DIR / "monthly_sales_summary.csv",
    index=False,
)

# 7. Visualisasi tren penjualan
plt.figure(figsize=(13, 6))
plt.plot(
    monthly_summary["date"],
    monthly_summary["total_units_sold"],
    linewidth=1.8,
)

plt.title("M5 Walmart - Monthly Total Sales")
plt.xlabel("Month")
plt.ylabel("Units Sold")
plt.grid(True, alpha=0.3)
plt.tight_layout()

chart_path = OUTPUT_DIR / "eda_monthly_sales.png"
plt.savefig(chart_path, dpi=150)
plt.close()

# 8. Tampilkan hasil singkat
print("\n--- CONTOH RINGKASAN HARIAN ---")
print(daily_summary.head(5).to_string(index=False))

print("\n--- CONTOH RINGKASAN BULANAN ---")
print(monthly_summary.head(5).to_string(index=False))

print("\n--- FILE HASIL ---")
print(f"Ringkasan harian : {REPORT_DIR / 'daily_sales_summary.csv'}")
print(f"Ringkasan bulanan: {REPORT_DIR / 'monthly_sales_summary.csv'}")
print(f"Grafik penjualan : {chart_path}")

print("\nEDA SELESAI!")
