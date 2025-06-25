import pandas as pd
pd.set_option("display.max_rows", None)
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import warnings
from tqdm import tqdm
warnings.filterwarnings("ignore")

# === Load and clean data ===
data = pd.read_csv('ALL_ASSETS_DAILY.csv', parse_dates=['Unnamed: 0'])
data = data.set_index('Unnamed: 0')
data = data.apply(pd.to_numeric, errors='coerce')
data.replace(0.0, np.nan, inplace=True)
data = data.loc[data.index < '2024-01-01']  # IN-SAMPLE

stocks = ["NFLX", "AAPL", "MSFT", "NVDA", "GOOGL", "META",  "AMZN", "WFC", "ORCL", "JPM", "INTC", "BAC"]
tech_stocks = ["NFLX", "AAPL", "MSFT", "NVDA", "GOOGL", "META",  "AMZN", "ORCL", "INTC"]
bank_stocks = ["WFC", "JPM", "BAC"]
metals = ["XAU"]
index = ["US500"]
crypto = ["BTCUSD"]

take_profit_pct = 0.3  
stop_loss_pct = 0.08 
holding_days = 40
slippage = 0.001
fees = 0.002
cap = 4 # NUMBER OF ASSET TO DIVERSIFY

asset_trading_hours = {
    "NFLX": 6.5,
    "AAPL": 6.5,
    "MSFT": 6.5,
    "NVDA": 6.5,
    "GOOGL": 6.5,
    "META": 6.5,
    "AMZN": 6.5,
    "WFC": 6.5,
    "ORCL": 6.5,
    "JPM": 6.5,
    "INTC": 6.5,
    "BAC": 6.5,
    "XAU": 23,
    "US500": 23,
    "BTCUSD": 24
}


#exclude = index + metals + bank_stocks
#close_cols = [col for col in data.columns if col.startswith('close_') and col.split('_')[1] not in exclude]
close_cols = [col for col in data.columns if col.startswith('close_')]
df_close = data[close_cols].copy()
print(df_close.head(300))

lookback = 30
strategy = pd.DataFrame(0, index=df_close.index, columns=df_close.columns)

for asset_col in df_close.columns:
    print("asset_col = ", asset_col)
    for i in df_close.index:
        if i < df_close.index[0] + pd.Timedelta(days=lookback):
            continue

        df = df_close.loc[i - pd.Timedelta(days=lookback):i, asset_col]
        #print(df)
        df = df.dropna()
        #print(df)

        price_direction = df.iloc[-1] - df.iloc[0]
        #print("price direction = ", price_direction)
        diff = df.diff()
        #print("diff = ", diff)
        mean = diff.mean()
        #print("mean = ", mean)
        std_dev = diff.std()
        #print("std = ", std_dev)
        threshold = diff.mean() + 3 * diff.std()
        #print("second std = ", threshold)

        best_line = np.polyfit(df[:-1], df[1:], 1)
        slope = best_line[0]

        if slope > 0.5 and price_direction > threshold:
            #print("SIGNAL!")
            strategy.at[i, asset_col] = slope
           
        
        #print("---------------------------------------")


#print(strategy.head(100))
strategy = strategy.replace(0, np.nan)


valid_rows = strategy.notna().sum(axis=1) >= cap
print("valid rows = ", valid_rows.loc['2020-07-10'])
print("valid rows = ", valid_rows.loc['2020-07-11'])

filtered_strategy = strategy[valid_rows]
print(filtered_strategy.loc['2020-07-11'])



ranks = filtered_strategy.rank(axis=1, ascending=False)


strategy = (ranks <= cap).astype(int)