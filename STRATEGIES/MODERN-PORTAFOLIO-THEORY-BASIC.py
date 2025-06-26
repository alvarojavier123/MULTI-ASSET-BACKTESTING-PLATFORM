import pandas as pd
pd.set_option("display.max_rows", None)
pd.set_option('display.float_format', lambda x: '%.10f' % x)


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
df_close = df_close.dropna()
#print(df_close.head(100))

lookback = 180

strategy = pd.DataFrame(index=df_close.index, columns=df_close.columns)
weights_df = pd.DataFrame(index=strategy.index, columns=strategy.columns, dtype='float64')

for current_date in tqdm(df_close.index):
    
    prices = df_close.loc[:current_date].tail(lookback)
    #print(prices)
    window_returns = prices.pct_change().dropna()
    #print(window_returns)

    if len(prices) < lookback:
        continue

    #print(current_date)
    #print("prices = ", prices['close_TSLA'])

    annual_returns = ((((prices.iloc[-1]-prices.iloc[0]) / prices.iloc[0]) + 1)**(252/len(prices)) - 1)
    #print("ANNUAL RETURNS = ", annual_returns)

    annual_std_dev = window_returns.std() * math.sqrt(252)
    #print(annual_std_dev)

    cov_matrix = window_returns.cov() * 252
    #print(cov_matrix)
    

    cov_pairs = []

    for i in range(len(cov_matrix.columns)):
        for j in range(i):
            asset1 = cov_matrix.columns[i]
            asset2 = cov_matrix.columns[j]
            cov_value = cov_matrix.iloc[i, j]
            cov_pairs.append((asset1, asset2, cov_value))


    negative_cov_pairs = [pair for pair in cov_pairs if pair[2] < 0]
    sorted_pairs = sorted(negative_cov_pairs, key=lambda x: x[2])  # from most negative up
    #print(sorted_pairs)
    selected_assets = set()

    for asset1, asset2, cov_val in sorted_pairs:
        if len(selected_assets) >= cap:
            break

        assets_in_pair = {asset1, asset2}
        #print(assets_in_pair)

        if selected_assets.isdisjoint(assets_in_pair):
            ret1 = annual_returns.get(asset1, np.nan)
            #print(ret1)
            ret2 = annual_returns.get(asset2, np.nan)
            #print(ret2)

            # Filtro: ambos retornos deben ser positivos
            if not np.isnan(ret1) and not np.isnan(ret2):
                if ret1 > 0.2 and ret2 > 0.2:
                    selected_assets.update(assets_in_pair)

                    portfolio = pd.DataFrame()
                    num_of_portfolios = 100
                    for i in range(num_of_portfolios):
                    
                        w1 = np.random.uniform(0.1, 0.9)
                        w2 = 1 - w1

                        portfolio.loc[i, f'{asset1}_weight'] = w1
                        portfolio.loc[i, f'{asset2}_weight'] = w2

                        portfolio.loc[i, 'returns'] = w1 * ret1 + w2 * ret2
                        portfolio.loc[i, 'std_dev'] = math.sqrt((w1**2)*(annual_std_dev[asset1]**2) +
                                                                (w2**2)*(annual_std_dev[asset2]**2)
                                                                + 2*w1*w2*cov_val)
                        portfolio.loc[i, 'returns/std_dev'] = portfolio.loc[i, 'returns'] / portfolio.loc[i, 'std_dev']

                    max_ret_by_std_dev = portfolio.iloc[portfolio['returns/std_dev'].idxmax()]
                    #print(max_ret_by_std_dev)
                    min_std_dev = portfolio.iloc[portfolio['std_dev'].idxmin()]
                    #print(portfolio)

                    # Print optimal weights
                    #print("Optimal weights (max return/std_dev):")
                    #print(max_ret_by_std_dev[[f'{asset1}_weight', f'{asset2}_weight']])
                    weights_df.at[current_date, asset1] = max_ret_by_std_dev[f'{asset1}_weight']
                    weights_df.at[current_date, asset2] = max_ret_by_std_dev[f'{asset2}_weight']
                    strategy.at[current_date, asset1] = 1
                    strategy.at[current_date, asset2] = 1
                    #print(strategy.loc[current_date])

                    """
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
                    """

                 
                
"""
*** MANUAL CAPITAL ALLOCATION ***
weights_df = pd.DataFrame(index=strategy.index, columns=strategy.columns, dtype='float64')

for date in strategy.index:

    today = strategy.loc[date]
    signals = today.dropna()

    selected_assets = list(signals.items())

    if len(selected_assets) > 0:

        for i , asset in enumerate(selected_assets):

            weight = weight_allocation[i]

            weights_df.at[date, asset[0]] = weight
"""