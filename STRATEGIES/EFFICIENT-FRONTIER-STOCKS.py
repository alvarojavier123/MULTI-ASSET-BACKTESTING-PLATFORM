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


exclude = index + metals  + crypto 
close_cols = [col for col in data.columns if col.startswith('close_') and col.split('_')[1] not in exclude]
#close_cols = [col for col in data.columns if col.startswith('close_')]
df_close = data[close_cols].copy()
df_close = df_close.dropna()
print(df_close.head(300))

# === SELECT YOUR ASSETS HERE ===
my_assets = ["GOOGL", "MSFT"]  # EDIT THIS LIST TO SELECT ASSETS

# Generate close column names based on selected assets
close_cols = [f"close_{ticker}" for ticker in my_assets]

# Filter and clean the data
df_close = data[close_cols].copy()
df_close = df_close.dropna()
print(df_close.head(300))

# Calculate annualised returns
annual_returns = ((((df_close.iloc[-1]-df_close.iloc[0]) /
                    df_close.iloc[0])+1)**(252/len(df_close)) - 1)

# Print the annualised returns
for col in close_cols:
    ticker = col.split('_')[1]
    print(f"The annualised returns of {ticker} is: ", str(round(annual_returns[col]*100, 2)) + "%")

# Calculate daily returns and annualised standard deviation
price_returns = df_close.pct_change().dropna()
annual_std_dev = price_returns.std()*math.sqrt(252)
for col in close_cols:
    ticker = col.split('_')[1]
    print(f"The annualised standard deviation of {ticker} is: ", str(round(annual_std_dev[col]*100, 2)) + "%")

# Set equal weights
a = 0.5
b = 0.5

# Calculate portfolio returns
portfolio_returns = a*annual_returns[close_cols[0]] + b*annual_returns[close_cols[1]]
print("The portfolio returns is: ", str(round(portfolio_returns*100, 2)) + "%")

# Calculate covariance and correlation
cov_matrix = np.cov(price_returns[close_cols[0]], price_returns[close_cols[1]], bias=True) * 252
print(f"The covariance of {my_assets[0]} and {my_assets[1]} is: ", round(cov_matrix[0, 1], 4))

# Calculate portfolio standard deviation
portfolio_std_dev = math.sqrt((a**2)*(annual_std_dev[close_cols[0]]**2) + (b**2)*(annual_std_dev[close_cols[1]]**2)
                               + 2*a*b*cov_matrix[0, 1])
print("The portfolio standard deviation is: ", str(round(portfolio_std_dev*100, 2)) + "%")

print("The (Portfolio returns/portfolio standard deviation) is: ",
      round(portfolio_returns/portfolio_std_dev, 2))


# Portfolio optimization - Efficient Frontier
portfolio = pd.DataFrame()
num_of_portfolios = 3000
for i in range(num_of_portfolios):
   
    w1 = np.random.uniform(0.1, 0.9)
    w2 = 1 - w1

    portfolio.loc[i, f'{my_assets[0]}_weight'] = w1
    portfolio.loc[i, f'{my_assets[1]}_weight'] = w2

    portfolio.loc[i, 'returns'] = w1 * annual_returns[close_cols[0]] + w2 * annual_returns[close_cols[1]]
    portfolio.loc[i, 'std_dev'] = math.sqrt((w1**2)*(annual_std_dev[close_cols[0]]**2) +
                                            (w2**2)*(annual_std_dev[close_cols[1]]**2)
                                            + 2*w1*w2*cov_matrix[0, 1])
    portfolio.loc[i, 'returns/std_dev'] = portfolio.loc[i, 'returns'] / portfolio.loc[i, 'std_dev']

max_ret_by_std_dev = portfolio.iloc[portfolio['returns/std_dev'].idxmax()]
min_std_dev = portfolio.iloc[portfolio['std_dev'].idxmin()]
print(portfolio)

# Plot the portfolios
plt.figure(figsize=(10, 7))
plt.grid()
plt.xlabel('Portfolio Standard Deviation', fontsize=14)
plt.xticks(fontsize=12)
plt.ylabel('Portfolio Returns', fontsize=14)
plt.yticks(fontsize=12)
plt.title('Portfolio Optimization based on Efficient Frontier', fontsize=20)
plt.scatter(portfolio.std_dev, portfolio.returns, label='Random Portfolios')
plt.legend(loc='best', fontsize=14)
plt.show()

# Highlight the optimal portfolios
plt.figure(figsize=(10, 7))
plt.grid()
plt.scatter(portfolio.std_dev, portfolio.returns, label='Random Portfolios')
plt.scatter(max_ret_by_std_dev.std_dev, max_ret_by_std_dev.returns,
            marker='*', s=200, color='r', label='Maximum Ret/Risk Portfolio')
plt.scatter(min_std_dev.std_dev, min_std_dev.returns,
            marker='*', s=200, color='darkorange', label='Minimum Risk Portfolio')
plt.xlabel('Portfolio Standard Deviation', fontsize=14)
plt.xticks(fontsize=12)
plt.ylabel('Portfolio Returns', fontsize=14)
plt.yticks(fontsize=12)
plt.legend(loc='best', fontsize=14)
plt.title('Portfolio Optimization based on Efficient Frontier', fontsize=20)
plt.show()

# Print optimal weights
print("Optimal weights (max return/std_dev):")
print(max_ret_by_std_dev[[f'{my_assets[0]}_weight', f'{my_assets[1]}_weight']])

