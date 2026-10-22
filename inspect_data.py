
from pathlib import Path
import pandas as pd

DATA_DIR = Path("data/raw")

required_files = [
    "sales_train_validation.csv",
    "calendar.csv",
    "sell_prices.csv",
]

print("=== PEMERIKSAAN DATASET M5 ===\n")

for filename in required_files:
    filepath = DATA_DIR / filename

    if not filepath.exists():
        print(f"[TIDAK DITEMUKAN] {filepath}")
        continue

    try:
        # Membaca lima baris pertama saja agar pemeriksaan awal ringan.
        df = pd.read_csv(filepath, nrows=5)

        print(f"\nFile: {filename}")
        print(f"Jumlah kolom: {len(df.columns)}")
        print(f"Ukuran file: {filepath.stat().st_size / (1024**2):.2f} MB")
        print(f"Kolom: {df.columns[:10].tolist()}")
        print("Contoh data:")
        print(df.head(3).to_string(index=False))

    except Exception as error:
        print(f"[ERROR] {filename}: {error}")

print("\n=== PEMERIKSAAN SELESAI ===")
