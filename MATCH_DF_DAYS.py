import pandas as pd
import os
from datetime import datetime

# === CONFIG ===
folders = ["ASSETS/STOCKS-DAILY", "ASSETS/METALS-DAILY", "ASSETS/INDEX-DAILY", "ASSETS/CRYPTO-DAILY",
            "ASSETS/FOREX-DAILY"]

start_date = "2020-01-01"
end_date = datetime.today().strftime("%Y-%m-%d")

# === Static date index ===
date_index = pd.date_range(start=start_date, end=end_date, freq='D')
master_df = pd.DataFrame(index=date_index)

# === Load and process each asset ===
for folder in folders:
    for file in os.listdir(folder):
        if file.endswith(".csv"):
            asset_name = file.replace(".csv", "")  # e.g., AMZN
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

            # Reindex to full date range, fill missing with 0
            ohlc = ohlc.reindex(date_index).fillna(0)

            # Merge into master dataframe
            master_df = master_df.join(ohlc)

# === Result ===
print(master_df.head())
# Save if you want
master_df.to_csv("ALL_ASSETS_DAILY.csv")
