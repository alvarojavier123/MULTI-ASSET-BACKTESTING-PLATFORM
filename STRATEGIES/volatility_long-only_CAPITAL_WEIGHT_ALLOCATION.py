import pandas as pd
pd.set_option("display.max_rows", None)
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import math
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

FTMO = ["AAPL", "AMZN", "BAC", "META", "GOOGL", "MSFT", "NFLX", "NVDA", "XAU", "BTC", "US500"] # I ALREADY HAVE
FTMO = ["all forex", "BABA", "TSLA", "PFE", "V", "ZM", "WMT" , "T" , "RACE" , "BAYGn", "ALVG"] # MISSING

take_profit_pct = 0.3  
stop_loss_pct = 0.03 
holding_days = 40
slippage = 0.001
fees = 0.002
cap = 2 # NUMBER OF ASSET TO DIVERSIFY
weight_allocation = [0.8, 0.2]  # 🔁 <-- Change this easily in future

if len(weight_allocation) != cap:
    raise ValueError(f"Length of weight_allocation ({len(weight_allocation)}) must match cap ({cap})")

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


#exclude = index + metals  + crypto 
#close_cols = [col for col in data.columns if col.startswith('close_') and col.split('_')[1] not in exclude]
close_cols = [col for col in data.columns if col.startswith('close_')]
df_close = data[close_cols].copy()
df_close = df_close.dropna()
print(df_close.head(100))

def apply_volatility_to_df(df_close, window, factor=np.sqrt(252)):
    volatility = pd.DataFrame(index=df_close.index)

    for col in df_close.columns:
        prices = df_close[col]
        clean_prices = prices.dropna()
        clean_returns = clean_prices.pct_change()
        clean_vol = clean_returns.rolling(window=window, min_periods=window).std() * factor
        full_vol = pd.Series(index=prices.index, dtype='float64')
        full_vol.loc[clean_vol.index] = clean_vol

        volatility[f'{col}'] = full_vol

    return volatility


volatility = apply_volatility_to_df(df_close, window=100)

volatility_clean = volatility.dropna(how='any')  # Filtra filas con al menos un NaN
print(volatility_clean.head(100))

vol_ranks = volatility_clean.rank(axis=1, ascending=True) # FALSE:MOST VOLATILE, TRUE:LESS VOLATILE
print(vol_ranks.head(100))

weights_df = pd.DataFrame(index=vol_ranks.index, columns=vol_ranks.columns, dtype='float64')

for date in vol_ranks.index:
    ranks_today = vol_ranks.loc[date]
    selected = ranks_today[ranks_today <= cap].sort_values()

    for i, (col, rank) in enumerate(selected.items()):
        weight = weight_allocation[i]
        weights_df.at[date, col] = weight

print(weights_df.head(100))

long_signal = (vol_ranks <= cap).astype(int)
print(long_signal.head(100))

strategy = long_signal


min_df = pd.read_csv("ALL_ASSETS_MINUTES.csv", parse_dates=[0], index_col=0)
min_df.replace(0.0, np.nan, inplace=True)
min_df.sort_index(inplace=True)
min_df = min_df.loc[min_df.index < '2024-01-01']  # IN-SAMPLE


assets = [col.replace("close_", "") for col in strategy.columns]

for asset in assets:
    signal_col = f'signal_{asset}'
    if signal_col not in min_df.columns:
        min_df[signal_col] = np.nan

for asset in assets:
    weight_col = f'weight_{asset}'
    if weight_col not in min_df.columns:
        min_df[weight_col] = np.nan


for date in strategy.index:
    timestamp = pd.Timestamp(f"{date.date()} 23:59:00")

    if timestamp in min_df.index:
        for asset in assets:
            signal_col = f'signal_{asset}'
            weight_col = f'weight_{asset}'

            signal = strategy.at[date, f'close_{asset}']
            weight = weights_df.at[date, f'close_{asset}']

            min_df.at[timestamp, signal_col] = signal

            if signal == 1 and not pd.isna(weight):
                min_df.at[timestamp, weight_col] = weight