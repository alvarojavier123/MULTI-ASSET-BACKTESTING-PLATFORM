import pandas as pd
pd.set_option("display.max_rows", None)
pd.set_option('display.float_format', lambda x: '%.10f' % x)
import datetime
import time
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
holding_days = 30
slippage = 0.001
fees = 0.002
cap = 2 # NUMBER OF ASSET TO DIVERSIFY


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

lookback = 60

strategy = pd.DataFrame(index=df_close.index, columns=df_close.columns)
weights_df = pd.DataFrame(index=strategy.index, columns=strategy.columns, dtype='float64')
diversification = {}


for current_date in tqdm(df_close.index):

    prices = df_close.loc[:current_date].tail(lookback)
    window_returns = prices.pct_change().dropna()
 
    if len(prices) < lookback:
        continue
    
    #print('current date = ', current_date)
    #print(prices)
    #print(window_returns)

    df_cumulative_returns = (window_returns+1).cumprod()*100
    #print(df_cumulative_returns)

    no_of_assets = df_cumulative_returns.shape[1]
    weights = cp.Variable(no_of_assets)

    portfolio_returns = (np.array(window_returns) @ weights)
    final_portfolio_value = cp.sum(cp.log(1 + portfolio_returns))


    objective = cp.Maximize(final_portfolio_value)


    constraints = [0.0 <= weights, weights <= 0.5, cp.sum(weights) == 1] # max 30% per asset
 

    problem = cp.Problem(objective, constraints)
  
    problem.solve(solver=cp.SCS)


    weights_val = weights.value
    weights_val[weights_val < 1e-3] = 0.0  # Set anything < 0.001 to 0
    weights_val = weights_val / weights_val.sum()  # Re-normalize to make sure sum is 1
    #print("Cleaned weights:", weights_val)

    selected_assets = df_cumulative_returns.columns[weights_val > 0]
    #print(selected_assets)
    
    
    for asset, weight in zip(selected_assets, weights_val[weights_val > 0]):
        #print(f"{asset}: {weight:.4f}")
        weights_df.at[current_date, asset] = weight
        strategy.at[current_date, asset] = 1
        if current_date not in diversification:
            diversification[current_date] = []

        diversification[current_date].append(asset)

        """
        res = input("Continue ?")
        if res == "":
                    print("--------------------------------------------------------------")
                    continue
        """
        
    
    #print(strategy.loc[current_date])
    #print(weights_df.loc[current_date])
    #print(diversification)


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
            
            elif signal == -1 and not pd.isna(weight):
                min_df.at[timestamp, weight_col] = weight


results = []

for asset in tqdm(assets, desc="Backtesting Progress"):

    hours_per_day = asset_trading_hours[asset]
    holding_period_minutes = int(holding_days * hours_per_day * 60)

    signal_col = f'signal_{asset}'
    ohlc_cols = [f"{prefix}_{asset}" for prefix in ["open", "high", "low", "close"]]

    columns = ohlc_cols + [signal_col]
    signals = min_df[(min_df[signal_col].notna()) & (min_df[signal_col] != 0)][columns]

    valid_data = min_df[min_df[f"open_{asset}"].notna()][ohlc_cols].copy()
    valid_data = valid_data.sort_index()

    for signal_time in signals.index:
        position = signals.loc[signal_time, signal_col]
        future_data = valid_data.loc[signal_time:, ohlc_cols]

        if future_data.empty:
            continue

        future_data['cum_minutes'] = range(1, len(future_data) + 1)
        entry_time = future_data.iloc[0].name
        entry_price = future_data.iloc[0][f"open_{asset}"] * (1 + slippage)

        """
        if position == 1:
            tp_price = entry_price * (1 + take_profit_pct)
            sl_price = entry_price * (1 - stop_loss_pct)

            future_data["hit_tp"] = future_data[f"high_{asset}"] >= tp_price
            future_data["hit_sl"] = future_data[f"low_{asset}"] <= sl_price

        elif position == -1:
            tp_price = entry_price * (1 - take_profit_pct)
            sl_price = entry_price * (1 + stop_loss_pct)

            future_data["hit_tp"] = future_data[f"low_{asset}"] <= tp_price
            future_data["hit_sl"] = future_data[f"high_{asset}"] >= sl_price
        """

        holding_limit_idx = future_data[future_data['cum_minutes'] <= holding_period_minutes].index.max()
        if pd.isna(holding_limit_idx):
            continue
        future_data = future_data.loc[:holding_limit_idx]

        """
        tp_idx = future_data[future_data["hit_tp"]].index.min()
        sl_idx = future_data[future_data["hit_sl"]].index.min()

        exit_time = None
        exit_reason = None
        exit_price = None

        if pd.notna(tp_idx) and pd.notna(sl_idx):
            if tp_idx < sl_idx:
                exit_time = tp_idx
                exit_reason = "TP"
                exit_price = tp_price * (1 - fees)
            else:
                exit_time = sl_idx
                exit_reason = "SL"
                exit_price = sl_price * (1 - fees)

        elif pd.notna(tp_idx):
            exit_time = tp_idx
            exit_reason = "TP"
            exit_price = tp_price * (1 - fees)

        elif pd.notna(sl_idx):
            exit_time = sl_idx
            exit_reason = "SL"
            exit_price = sl_price * (1 - fees)

        else:
        """
        exit_time = holding_limit_idx
        exit_reason = "Hold"
        exit_price = future_data.loc[holding_limit_idx][f"close_{asset}"] * (1 - fees)

        if pd.isna(exit_time) or pd.isna(exit_price):
            continue

        ret = (exit_price - entry_price) / entry_price if position == 1 else (entry_price - exit_price) / entry_price

        weight_col = f"weight_{asset}"
        weight = min_df.at[signal_time, weight_col] if weight_col in min_df.columns else np.nan

        results.append({
            "asset": asset,
            "entry_time": entry_time,
            "exit_time": exit_time,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "return": ret,
            "exit_reason": exit_reason,
            "hold_period_day": holding_limit_idx,
            "position": position,
            "weight": weight,
            "signal time": signal_time
        })



results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by='entry_time').reset_index(drop=True)
print(results_df.head(30))

results_df.to_csv("results_df_IN_SAMPLE.csv")


results_df = pd.read_csv("results_df_IN_SAMPLE.csv")
print(results_df.head(30))


filtered_trades = []
open_trades = []
max_time = ()

for i, row in results_df.iterrows():
    print("open trades = ", open_trades)
    #print("i = ", i)
    #print("row = ", row)
    entry_time = pd.to_datetime(row['entry_time'])
    print("entry time = ", entry_time)
    hold_end_time = pd.to_datetime(row['hold_period_day'])
    #print("hold end time = ", hold_end_time)
    print("max time = ", max_time)
    
    if len(filtered_trades) == 0:
        date = entry_time.normalize()
        cap = len(diversification[date])
        print("cap = ", cap)

    #print("------------------------------")


    if len(open_trades) < cap:
        print("open trades = ", open_trades)
        print("i = ", i)
        print("row = ", row)
        print("entry time = ", entry_time)
        print("hold end time = ", hold_end_time)
        print("max time = ", max_time)

        if (len(open_trades) > 0):
            if row['asset'] in [t[0] for t in open_trades]:
                continue
        
        row['diversification'] = cap
        filtered_trades.append(row)
        open_trades.append((row['asset'], hold_end_time))

        max_time = max(t[1] for t in open_trades)
        print("max time = ", max_time)


        continue
    
    elif (len(open_trades) == cap) and (entry_time >= max_time):
        open_trades = []
        print("open trades = ", open_trades)
        print(row)
        signal_time = pd.to_datetime(row['signal time'])
        signal_time = signal_time.normalize()


        if signal_time in diversification:
            
            cap = len(diversification[signal_time])
            print(diversification[signal_time])
            print("cap = ", cap)
            row['diversification'] = cap
            filtered_trades.append(row)
            open_trades.append((row['asset'], hold_end_time))
            print("open trades = ", open_trades)

            max_time = max(t[1] for t in open_trades)
            print(f"{signal_time} IN dictionary")
            """
            res = input("Continue ?")
            if res == "":
                    print("--------------------------------------------------------------")
                    continue
            """
        else:
            print(f"{signal_time} not in dictionary")

        continue
    

filtered_df = pd.DataFrame(filtered_trades)
filtered_df =  filtered_df.drop('Unnamed: 0', axis = 1)

col = 'hold_period_day'
filtered_df = filtered_df[[c for c in filtered_df.columns if c != col] + [col]]
print(filtered_df.head(2000))


import plotly.graph_objects as go


# --- Agrupar y calcular retorno real por bloques ---
returns_summary = []
portfolio_returns = []
return_dates = []

for signal_time, group in filtered_df.groupby('signal time'):
    group = group.copy()
    cap = group['diversification'].iloc[0]

    if len(group) == cap:
        total_return = np.sum(group['return'] * group['weight'])

        assets = ', '.join(group['asset'])
        entry_times = ', '.join(group['entry_time'].astype(str))
        exit_times = ', '.join(group['exit_time'].astype(str))
        exit_reasons = ', '.join(group['exit_reason'])
        asset_returns = ', '.join(group['return'].round(10).astype(str))
        asset_weights = ', '.join(group['weight'].round(10).astype(str))
        hold_date = pd.to_datetime(group['hold_period_day'].max())

        portfolio_returns.append(total_return)
        return_dates.append(hold_date)

        returns_summary.append({
            'assets': assets,
            'entry_times': entry_times,
            'exit_times': exit_times,
            'exit_reasons': exit_reasons,
            'returns': asset_returns,
            'weights': asset_weights,
            'hold_period_day': hold_date,
            'portfolio_return': total_return
        })

returns_df = pd.DataFrame(returns_summary)
returns_df.set_index('hold_period_day', inplace=True)
returns_df = returns_df.sort_index()
returns_series = returns_df['portfolio_return']

# --- Mostrar todas las columnas de returns_df ---
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

print(returns_df)


# --- Métricas ---
capital = (1 + returns_series).cumprod()
cumulative_compounded = capital - 1
cumulative_simple = returns_series.cumsum()

rolling_max = capital.cummax()
drawdown = capital / rolling_max - 1
drawdown = drawdown.fillna(0)
max_drawdown = drawdown.min()

mean_return = returns_series.mean()
std_return = returns_series.std()
days_per_period = returns_series.index.to_series().diff().dt.days.median()
sharpe_ratio = (mean_return / std_return) * np.sqrt(252 / days_per_period) if std_return != 0 else 0

wins = returns_series[returns_series > 0]
losses = returns_series[returns_series < 0]
win_rate = len(wins) / len(returns_series) if len(returns_series) > 0 else 0
profit_factor = wins.sum() / -losses.sum() if losses.sum() != 0 else np.nan

# --- Plot ---
fig = go.Figure()
fig.add_trace(go.Scatter(x=returns_series.index, y=cumulative_compounded, mode='lines', name='Compounded Returns (%)', line=dict(color='blue')))
fig.add_trace(go.Scatter(x=returns_series.index, y=cumulative_simple, mode='lines', name='Simple Returns (Cumsum)', line=dict(color='green', dash='dash')))
fig.add_trace(go.Scatter(x=returns_series.index, y=drawdown, mode='lines', fill='tozeroy', name='Drawdown', line=dict(color='red'), fillcolor='rgba(255, 0, 0, 0.3)', yaxis='y2'))

fig.update_layout(
    title=f"Strategy Performance from {returns_series.index.min().date()} to {returns_series.index.max().date()}",
    xaxis_title='Date',
    yaxis=dict(title='Returns', side='left', tickformat='.0%'),
    yaxis2=dict(title='Drawdown', overlaying='y', side='right', showgrid=False, tickformat=".0%", range=[drawdown.min() * 1.1, 0]),
    legend=dict(x=0.01, y=0.99),
    hovermode='x unified',
    margin=dict(l=60, r=60, t=60, b=60),
    height=600
)

metrics_text = (
    f"Compounded Returns: {cumulative_compounded.iloc[-1]:.2%}\n"
    f"Simple Returns: {cumulative_simple.iloc[-1]:.2%}\n"
    f"Max Drawdown: {max_drawdown:.2%}\n"
    f"Sharpe Ratio: {sharpe_ratio:.2f}\n"
    f"Win Rate: {win_rate:.2%}\n"
    f"Profit Factor: {profit_factor:.2f}"
)

print("====== Strategy Performance Metrics ======")
print(metrics_text)
print("==========================================")

fig.add_annotation(
    xref='paper', yref='paper',
    x=0.5, y=0.98,
    xanchor='center',
    showarrow=False,
    align='left',
    bgcolor='white',
    bordercolor='black',
    borderwidth=1,
    borderpad=4,
    text=metrics_text.replace('\n', '<br>'),
    font=dict(size=12, color='black', family='Arial Black')
)

import plotly.io as pio
pio.renderers.default = "browser"
fig.show()



