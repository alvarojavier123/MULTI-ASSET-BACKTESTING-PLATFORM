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
print(long_signal.head(100))





"""
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



min_df = pd.read_csv("ALL_ASSETS_MINUTES.csv", parse_dates=[0], index_col=0)
min_df.replace(0.0, np.nan, inplace=True)
min_df.sort_index(inplace=True)
min_df = min_df.loc[min_df.index < '2024-01-01']  # IN-SAMPLE


assets = [col.replace("close_", "") for col in strategy.columns]

for asset in assets:
    signal_col = f'signal_{asset}'
    if signal_col not in min_df.columns:
        min_df[signal_col] = np.nan

for date in strategy.index:
    timestamp = pd.Timestamp(f"{date.date()} 23:59:00")

    if timestamp in min_df.index:
        for asset in assets:
            signal_col = f'signal_{asset}'
            signal = strategy.at[date, f'close_{asset}']
            min_df.at[timestamp, signal_col] = signal



results = []

for asset in tqdm(assets, desc="Backtesting Progress"):

    print(asset)
    hours_per_day = asset_trading_hours[asset]
    holding_period_minutes = int(holding_days * hours_per_day * 60)

    signal_col = f'signal_{asset}'
    ohlc_cols = [f"{prefix}_{asset}" for prefix in ["open", "high", "low", "close"]]

    columns = ohlc_cols + [signal_col]
    signals = min_df[(min_df[signal_col].notna()) & (min_df[signal_col] != 0)][columns]
    print(f"{asset} - N señales: {len(signals)}")

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

        else:
            continue

        holding_limit_idx = future_data[future_data['cum_minutes'] <= holding_period_minutes].index.max()
        if pd.isna(holding_limit_idx):
            continue
        future_data = future_data.loc[:holding_limit_idx]

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
            exit_time = holding_limit_idx
            exit_reason = "Hold"
            exit_price = future_data.loc[holding_limit_idx][f"close_{asset}"] * (1 - fees)

        if pd.isna(exit_time) or pd.isna(exit_price):
            continue

        ret = (exit_price - entry_price) / entry_price if position == 1 else (entry_price - exit_price) / entry_price

        results.append({
            "asset": asset,
            "entry_time": entry_time,
            "exit_time": exit_time,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "return": ret,
            "exit_reason": exit_reason,
            "hold_period_day": holding_limit_idx,
            "position": position
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
    #print("i = ", i)
    #print("row = ", row)
    entry_time = pd.to_datetime(row['entry_time'])
    #print("entry time = ", entry_time)
    hold_end_time = pd.to_datetime(row['hold_period_day'])
    #print("hold end time = ", hold_end_time)
    #print("max time = ", max_time)
    #print("------------------------------")

    if len(open_trades) < cap:

        if (len(open_trades) > 0):
            if row['asset'] in [t[0] for t in open_trades]:
                continue

        filtered_trades.append(row)
        open_trades.append((row['asset'], hold_end_time))
        print(open_trades)

        max_time = max(t[1] for t in open_trades)
        #print("max time = ", max_time)


        continue
    
    elif (len(open_trades) == cap) and (entry_time >= max_time):
        open_trades = []
        #print(open_trades)

        filtered_trades.append(row)
        open_trades.append((row['asset'], hold_end_time))
        #print(open_trades)

        max_time = max(t[1] for t in open_trades)
        #print("max time = ", max_time)


        continue

    

filtered_df = pd.DataFrame(filtered_trades)
print(filtered_df.head(50))

chunks = [filtered_df.iloc[i:i+cap] for i in range(0, len(filtered_df), cap)]
returns_summary = []
portfolio_returns = []
return_dates = []

for chunk in chunks:
    if len(chunk) == cap:
        # FIX: Correct portfolio return for equal capital allocation (not compounded)
        total_return = chunk['return'].mean()  # <== THIS IS THE FIX

        assets = ', '.join(chunk['asset'])
        entry_times = ', '.join(chunk['entry_time'].astype(str))
        exit_times = ', '.join(chunk['exit_time'].astype(str))
        exit_reasons = ', '.join(chunk['exit_reason'])

        portfolio_returns.append(total_return)
        return_dates.append(pd.to_datetime(chunk['exit_time'].max()))

        returns_summary.append({
            'assets': assets,
            'entry_times': entry_times,
            'exit_times': exit_times,
            'exit_reasons': exit_reasons,
            'portfolio_return': total_return
        })

summary_df = pd.DataFrame(returns_summary)
print(summary_df.head())



# Step 3: Compute performance metrics and plot
import numpy as np
import plotly.graph_objects as go

returns_series = pd.Series(portfolio_returns, index=return_dates).sort_index()
capital = (1 + returns_series).cumprod()
cumulative_compounded = capital - 1  # Start at 0%
cumulative_simple = returns_series.cumsum()

# Drawdown
rolling_max = capital.cummax()
drawdown = capital / rolling_max - 1
drawdown = drawdown.fillna(0)
max_drawdown = drawdown.min()

# Sharpe ratio
mean_return = returns_series.mean()
std_return = returns_series.std()
days_per_period = returns_series.index.to_series().diff().dt.days.median()
sharpe_ratio = (mean_return / std_return) * np.sqrt(252 / days_per_period) if std_return != 0 else 0

# Win rate and profit factor
wins = returns_series[returns_series > 0]
losses = returns_series[returns_series < 0]
win_rate = len(wins) / len(returns_series) if len(returns_series) > 0 else 0
profit_factor = wins.sum() / -losses.sum() if losses.sum() != 0 else np.nan

# --- Interactive Plot ---
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=returns_series.index,
    y=cumulative_compounded,
    mode='lines',
    name='Compounded Returns (%)',
    line=dict(color='blue')
))

fig.add_trace(go.Scatter(
    x=returns_series.index,
    y=cumulative_simple,
    mode='lines',
    name='Simple Returns (Cumsum)',
    line=dict(color='green', dash='dash')
))

fig.add_trace(go.Scatter(
    x=returns_series.index,
    y=drawdown,
    mode='lines',
    fill='tozeroy',
    name='Drawdown',
    line=dict(color='red'),
    fillcolor='rgba(255, 0, 0, 0.3)',
    yaxis='y2'
))

fig.update_layout(
    title=f"Strategy Performance from {returns_series.index.min().date()} to {returns_series.index.max().date()}",
    xaxis_title='Date',
    yaxis=dict(title='Returns', side='left', tickformat='.0%'),
    yaxis2=dict(
        title='Drawdown',
        overlaying='y',
        side='right',
        showgrid=False,
        tickformat=".0%",
        range=[drawdown.min() * 1.1, 0]
    ),
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

# Print metrics
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

"""



"""
res = input("Continue ?")
        if res == "":
                    print("--------------------------------------------------------------")
                    continue
        
"""
