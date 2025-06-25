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
data = pd.read_csv('../ALL_ASSETS_DAILY.csv', parse_dates=['Unnamed: 0'])
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
stop_loss_pct = 0.03 
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


exclude = index + metals + bank_stocks + crypto 
close_cols = [col for col in data.columns if col.startswith('close_') and col.split('_')[1] not in exclude]
#close_cols = [col for col in data.columns if col.startswith('close_')]
df_close = data[close_cols].copy()
df_close = df_close.dropna()
print(df_close.head(300))

# Calculate annualised returns
annual_returns = ((((df_close.iloc[-1]-df_close.iloc[0]) /
                    df_close.iloc[0])+1)**(252/len(df_close)) - 1)

# Print the annualised returns of the stocks
print("The annualised returns of Microsoft is: ",
      str(round(annual_returns['close_MSFT']*100, 2))+"%")

print("The annualised returns of Google is: ",
      str(round(annual_returns['close_GOOGL']*100, 2))+"%")


price_returns = df_close.pct_change().dropna()

annual_std_dev = price_returns.std()*math.sqrt(252)
print("The annualised standard deviation of Microsoft is: ",
      str(round(annual_std_dev['close_MSFT']*100, 2))+"%")

print("The annualised standard deviation of Google is: ",
      str(round(annual_std_dev['close_GOOGL']*100, 2))+"%")

a = 0.5
b = 0.5

portolio_returns = a*annual_returns['close_MSFT'] + b*annual_returns['close_GOOGL']
print("The portfolio returns is: ", str(round(portolio_returns*100, 2))+"%")

# Calculate the covariance of the stocks and multiply it by 252 to get annualised covariance
cov_msft_googl = np.cov(
    price_returns['close_MSFT'], price_returns['close_GOOGL'], bias=True) * 252

print("The covariance of MSFT and GOOGL is: ", round(cov_msft_googl[0, 1], 4))

correlation = price_returns['close_MSFT'].corr(price_returns['close_GOOGL'])
print("The correlation between MSFT and GOOGL is:", round(correlation, 4))

# Calculate portfolio standard deviation
portolio_std_dev = math.sqrt((a**2)*(annual_std_dev['close_MSFT']**2) + (b**2)*(annual_std_dev['close_GOOGL']**2)
                             + 2*a*b*cov_msft_googl[0, 1])

# Print the portfolio standard deviation
print("The portfolio standard deviation is: ",
      str(round(portolio_std_dev*100, 2))+"%")
