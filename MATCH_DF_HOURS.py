import pandas as pd
import os
from datetime import datetime

# === CONFIG ===
folders = ["ASSETS/STOCKS-HOURS", "ASSETS/METALS-HOURS", "ASSETS/INDEX-HOURS", "ASSETS/CRYPTO-HOURS",
           "ASSETS/FOREX-HOURS"]
start_date = "2020-01-01 00:00:00"
end_date = datetime.today().strftime("%Y-%m-%d %H:00:00")

# === Static hourly index ===
date_index = pd.date_range(start=start_date, end=end_date, freq='H')
master_df = pd.DataFrame(index=date_index)

# === Load and process each asset ===
for folder in folders:
    for file in os.listdir(folder):
        if file.endswith(".csv"):
            asset_name = file.replace(".csv", "")
            df = pd.read_csv(os.path.join(folder, file))

            # Parse and set datetime index
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)

            # Keep only OHLC
            ohlc = df[['open', 'high', 'low', 'close']].copy()

            # Rename columns
            ohlc.columns = [f"{col}_{asset_name}" for col in ohlc.columns]

            # Add dummy signal column
            ohlc[f"signal_{asset_name}"] = 0

            # Reindex to full hourly range and fill missing with 0
            ohlc = ohlc.reindex(date_index).fillna(0)

            # Merge into master DataFrame
            master_df = master_df.join(ohlc)

# === Result ===
print(master_df.head())
# Optional: Save to CSV
master_df.to_csv("ALL_ASSETS_HOURLY.csv")
