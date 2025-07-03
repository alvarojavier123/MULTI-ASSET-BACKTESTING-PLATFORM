import pandas as pd
pd.set_option("display.max_rows", None)
pd.set_option('display.float_format', lambda x: '%.10f' % x)

import cvxpy as cp
import numpy as np
np.set_printoptions(suppress=True, precision=10)
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

stocks = [
      "NFLX", "AAPL", "MSFT", "NVDA", "GOOGL", "META",  "AMZN", "WFC", "ORCL", 
      "JPM", "INTC", "BAC", "TSLA", "BABA", "V", "T", "WMT", "PFE"]

tech_stocks = ["NFLX", "AAPL", "MSFT", "NVDA", "GOOGL", "META",  "AMZN", "ORCL", "INTC", "BABA"]
bank_stocks = ["WFC", "JPM", "BAC"]
metals = ["XAU"]
index = ["US500"]
crypto = ["BTCUSD"]


take_profit_pct = 0.2  
stop_loss_pct = 0.05
holding_days = 20
slippage = 0.001
fees = 0.002
cap = 2 # NUMBER OF ASSET TO DIVERSIFY
weight_allocation = [0.5, 0.5]  

if len(weight_allocation) != cap:
    raise ValueError(f"Length of weight_allocation ({len(weight_allocation)}) must match cap ({cap})")

asset_trading_hours = {
    # === Stocks (US Market: 6.5h/día) ===
    "NFLX": 6.5, "AAPL": 6.5, "MSFT": 6.5, "NVDA": 6.5, "GOOGL": 6.5, "META": 6.5,
    "AMZN": 6.5, "WFC": 6.5, "ORCL": 6.5, "JPM": 6.5, "INTC": 6.5, "BAC": 6.5,
    "TSLA": 6.5, "BABA": 6.5, "V": 6.5, "T": 6.5, "WMT": 6.5, "PFE": 6.5,

    # === Metals & Index ===
    "XAU": 23,       # Gold: 23h (cierra 1h diaria por mantenimiento)
    "US500": 23,     # S&P500 CFD: 23h/día típicamente (depende del bróker)

    # === Crypto ===
    "BTCUSD": 24,    # 24h/7d

    # === Forex (24h de lunes a viernes, promedio diario ≈ 24h) ===
    "AUDJPY": 24,
    "EURUSD": 24,
    "GBPUSD": 24,
    "USDCAD": 24,
    "USDCHF": 24,
    "USDJPY": 24
}


#exclude = index + metals  + crypto 
#close_cols = [col for col in data.columns if col.startswith('close_') and col.split('_')[1] not in exclude]
close_cols = [col for col in data.columns if col.startswith('close_')]
df_close = data[close_cols].copy()
date_range = df_close.loc['2022-04-02':'2022-07-13'] #  WEIRD DATE RANGE MANY ROWS DROPPED...
all_nan_cols_range = date_range.columns[date_range.isna().all()]

for col in all_nan_cols_range:
    print(col)

df_close.drop(columns=all_nan_cols_range, inplace=True)
df_close = df_close.dropna()

df_change = df_close.pct_change()
df_change.dropna(inplace=True)

df_cumulative_returns = (df_change+1).cumprod()*100



no_of_assets = df_change.shape[1]
print("no of stocks = ", no_of_assets)

weights = cp.Variable(no_of_assets)

print(np.array(df_change))
print("weights.value = ", weights.value)

# Save the portfolio returns in a variable
portfolio_returns = (np.array(df_change) @ weights)
print(portfolio_returns)

final_portfolio_value = cp.sum(cp.log(1+portfolio_returns))
print(final_portfolio_value.value)

objective = cp.Maximize(final_portfolio_value)
print(objective)

constraints = [0.0 <= weights, cp.sum(weights) == 1]
print(constraints)

problem = cp.Problem(objective, constraints)
print("problem = ", problem)

problem.solve()

print(weights.value)

kelly_portfolio_returns = ((df_change)*(weights.value)).sum(axis=1)

kelly_portfolio_value = (1+(kelly_portfolio_returns)).cumprod()

kelly_annualised_returns = ((kelly_portfolio_value[-1])**(252/len(df_change)))-1

equal_weight_portfolio = (1+((df_change).mean(axis=1))).cumprod()


plt.figure(figsize=(10, 7))
plt.plot(kelly_portfolio_value*100, label='Kelly Portfolio Performance')
plt.plot(equal_weight_portfolio*100,
         label='Equal Weight Portfolio Performance')

plt.xlabel('Date', fontsize=14)
plt.ylabel('Portfolio Returns', fontsize=14)
plt.legend(loc='best', fontsize=12)
plt.title('Portfolio Optimisation based on Kelly Criterion', fontsize=16)
plt.legend(loc='best', fontsize=12)
plt.grid()
plt.show()
