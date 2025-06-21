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

cap = 3

exclude = index + metals + bank_stocks

close_cols = [col for col in data.columns if col.startswith('close_') and col.split('_')[1] not in exclude]
df_close = data[close_cols].copy()

print(df_close.head(300))


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

vol_ranks = volatility_clean.rank(axis=1, ascending=False) # FALSE:MOST VOLATILE, TRUE:LESS VOLATILE

long_signal = (vol_ranks <= cap).astype(int)
