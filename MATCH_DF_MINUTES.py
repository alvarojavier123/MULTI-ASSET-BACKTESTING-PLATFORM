import pandas as pd
import os
from datetime import datetime

# === CONFIG ===
folders = ["ASSETS/STOCKS-MINUTES", "ASSETS/METALS-MINUTES", "ASSETS/INDEX-MINUTES", "ASSETS/CRYPTO-MINUTES",
           "ASSETS/FOREX-MINUTES"]

start_date = "2020-01-01 00:00:00"
end_date = datetime.today().strftime("%Y-%m-%d %H:%M:00")

# === Static minute-level index ===
date_index = pd.date_range(start=start_date, end=end_date, freq='T')
master_df = pd.DataFrame(index=date_index)

# === Load and process each asset ===
for folder in folders:
    for file in os.listdir(folder):
        if file.endswith(".csv"):
            asset_name = file.replace(".csv", "")
            path = os.path.join(folder, file)

            try:
                df = pd.read_csv(path)

                # Verifica que la columna 'timestamp' exista
                if 'timestamp' not in df.columns:
                    print(f"⛔ ERROR: 'timestamp' column not found in {path}")
                    continue

                # Parse timestamp and set index
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
                df = df.dropna(subset=['timestamp'])  # Drop rows where timestamp couldn't be parsed
                df.set_index('timestamp', inplace=True)

                # Drop duplicated timestamps (keep the first)
                if df.index.duplicated().any():
                    print(f"⚠️ WARNING: Duplicate timestamps found in {file}. Dropping duplicates.")
                    df = df[~df.index.duplicated(keep='first')]

                # Keep only OHLC columns
                if not all(col in df.columns for col in ['open', 'high', 'low', 'close']):
                    print(f"⛔ ERROR: Missing OHLC columns in {path}")
                    continue

                ohlc = df[['open', 'high', 'low', 'close']].copy()
                ohlc.columns = [f"{col}_{asset_name}" for col in ohlc.columns]

                # Add dummy signal column
                ohlc[f"signal_{asset_name}"] = 0

                # Reindex to full minute range and fill missing values with 0
                ohlc = ohlc.reindex(date_index).fillna(0)

                # Merge into master DataFrame
                master_df = master_df.join(ohlc)

            except Exception as e:
                print(f"❌ Failed to process {path}: {e}")

# === Resultado final ===
print(master_df.head())

# Opcional: guardar en CSV
master_df.to_csv("ALL_ASSETS_MINUTES.csv")
print("✅ Archivo guardado como ALL_ASSETS_MINUTES.csv")
