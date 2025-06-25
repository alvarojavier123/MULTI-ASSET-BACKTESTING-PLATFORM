import pandas as pd
pd.set_option("display.max_rows", None)
pd.set_option("display.float_format", lambda x: f"{x:.6f}")
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

take_profit_pct = 0.1 
stop_loss_pct = 0.01 
holding_days = 10
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


strategy = pd.DataFrame(np.nan, index=df_close.index, columns=df_close.columns)  # Start with NaN, not 0.0


for asset_col in tqdm(df_close.columns):
    price_series = df_close[asset_col].dropna()

    for current_date in price_series.index:
        hist = price_series.loc[:current_date].dropna()
        if len(hist) < 101:
            continue

        hist_returns = hist.pct_change().dropna()
        if len(hist_returns) < 100:
            continue

        ma_long = hist_returns.iloc[-100:].mean()
        ma_short = hist_returns.iloc[-20:].mean()
        signal = ma_short - ma_long

        if signal > 0:
            strategy.at[current_date, asset_col] = signal

print(strategy)
print(strategy.loc['2020-06-01'])
print(len(strategy))
valid_rows = strategy.notna().sum(axis=1) >= cap
filtered_strategy = strategy[valid_rows]
ranks = filtered_strategy.rank(axis=1, ascending=False)
print(ranks.loc['2020-06-01'])

strategy = (ranks <= cap).astype(int)
print(len(strategy))
print(strategy.loc['2020-06-01'])