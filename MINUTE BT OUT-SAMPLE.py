import pandas as pd
pd.set_option("display.max_rows", None)
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from colorama import init, Fore, Style
init()

# Load data
df = pd.read_csv('BTC_USDT_1h_since_2020 (ORDERFLOW DATA).csv')
df.set_index('timestamp', inplace=True)
df = df.loc[df.index > '2023-11-01']  # OUT-SAMPLE
df = df.dropna()



holding_period = 720 # HOURS
holding_period = holding_period * 60 
stop_loss_pct = 0.01
take_profit_pct = 0.4
trading_cost = 0.001


min_df = pd.read_csv("CRYPTO-MINUTES/BTC.csv")
min_df.index = pd.to_datetime(min_df['timestamp'])
min_df = min_df.drop('timestamp', axis=1)


# === AJUSTE AUTOMÁTICO DE ÍNDICE DE SEÑALES SEGÚN FRECUENCIA ===
shifted_signals = df.copy()
shifted_signals.index = pd.to_datetime(shifted_signals.index)

delta = shifted_signals.index.to_series().diff().median()

if delta >= pd.Timedelta(hours=23):  # DAILY DATA
    # 1) Agrupa min_df por fecha y extrae el último timestamp de cada día
    last_minute_per_day = min_df.index.to_series().groupby(min_df.index.date).max()
    # 2) Para cada día en shifted_signals, mapea al último minuto de ese día
    new_index = [
        last_minute_per_day.get(ts.date(), pd.NaT)
        for ts in shifted_signals.index
    ]
    shifted_signals.index = pd.to_datetime(new_index)

elif delta >= pd.Timedelta(minutes=59):  # HOURLY DATA
    # Simplemente desplaza cada hora a hh:59
    shifted_signals.index = shifted_signals.index + pd.Timedelta(minutes=59)

else:
    raise ValueError("Frecuencia no reconocida: espera datos diarios u horarios.")


min_df['signal'] = shifted_signals['final_signal'].reindex(min_df.index)
min_df = min_df.loc[min_df.index > '2024-01-01']  # OUT-SAMPLE


# Prepare columns for PnL and trade info
min_df['pnl'] = 0.0
min_df['exit_reason'] = np.nan
min_df['entry_time'] = pd.NaT
min_df['exit_time'] = pd.NaT

prices = min_df['close'].values
signals = min_df['signal'].values

last_exit = -1
trade_counter = 0
trades = []

print("\nStarting backtest with progress bar...\n")

for i in tqdm(range(len(min_df) - holding_period), desc="Backtesting Progress"):
    if i <= last_exit:
        continue  # skip if still in a trade

    if signals[i] == 0 or np.isnan(signals[i]):
        continue  # no signal here

    direction = signals[i]

    entry_price = prices[i]
    max_slippage = 0.001
    entry_price *= (1 + direction * max_slippage)

    max_exit_idx = i + holding_period
    
    tp_hit, sl_hit = None, None

    # Walk forward through each minute of the trade window
    for j in range(i + 1, max_exit_idx + 1):
        high = min_df['high'].iloc[j]
        low = min_df['low'].iloc[j]

        if direction == 1:
            if high >= entry_price * (1 + take_profit_pct):
                tp_hit = j
                break
            if low <= entry_price * (1 - stop_loss_pct):
                sl_hit = j
                break
        else:  # direction == -1
            if low <= entry_price * (1 - take_profit_pct):
                tp_hit = j
                break
            if high >= entry_price * (1 + stop_loss_pct):
                sl_hit = j
                break

    # Decide exit
    if tp_hit is not None:
        exit_idx = tp_hit
        reason = 'TP'
        exit_price = entry_price * (1 + direction * take_profit_pct)

    elif sl_hit is not None:
        exit_idx = sl_hit
        reason = 'SL'
        exit_price = entry_price * (1 - direction * stop_loss_pct)

    else:
        exit_idx = max_exit_idx
        reason = 'Hold'
        exit_price = min_df['close'].iloc[exit_idx]

    
    gross_ret = direction * (exit_price - entry_price) / entry_price
    ret = gross_ret - 2 * trading_cost  # fees applied on entry and exit

    min_df.at[min_df.index[exit_idx], 'entry_time'] = min_df.index[i]
    min_df.at[min_df.index[exit_idx], 'exit_time'] = min_df.index[exit_idx]
    min_df.at[min_df.index[exit_idx], 'pnl'] = ret
    min_df.at[min_df.index[exit_idx], 'exit_reason'] = reason


    trades.append([min_df.index[i], min_df.index[exit_idx], direction, entry_price, exit_price, ret, reason])

    trade_counter += 1
    last_exit = exit_idx

    """
    cooldown = 240 * 60
    if reason == 'SL':
        signals[i+1:exit_idx+1+cooldown] = 0

    cooldown = 48
    signals[i+1:exit_idx+1+cooldown] = 0
    """


# Save trade log CSV
trade_log = pd.DataFrame(trades, columns=['entry_time', 'exit_time', 'direction', 'entry_price', 'exit_price', 'return', 'exit_reason'])

pd.set_option("display.max_rows", None)
# Filter only rows where a trade was closed
actual_trades = min_df[min_df['pnl'] != 0]

# Display key trade information
print(actual_trades[['entry_time', 'exit_time', 'signal', 'pnl', 'exit_reason']])

# Calculate cumulative return from actual trades
trade_pnls = actual_trades['pnl']
cumprod = (1 + trade_pnls).cumprod()

# Backtest duration and annualization factor
start_time = actual_trades['entry_time'].iloc[0]
end_time = actual_trades['exit_time'].iloc[-1]
total_days = (end_time - start_time).total_seconds() / (3600 * 24)
annual_factor = np.sqrt(252 / (total_days / len(trade_pnls)))

# Metrics
sharpe = trade_pnls.mean() / trade_pnls.std() * annual_factor
rolling_max = cumprod.cummax()
drawdown = cumprod / rolling_max - 1
max_dd = drawdown.min()

# Print results
print(Fore.YELLOW + f"\nTrade count: {len(actual_trades)}" + Style.RESET_ALL)
print(Fore.CYAN + f"Sharpe Ratio: {sharpe:.2f}")
print(f"Cumulative Return (final): {cumprod.iloc[-1]:.2f}")
print(f"Max Drawdown: {max_dd:.2%}" + Style.RESET_ALL)

# Plot cumulative returns
plt.figure(figsize=(15, 7))
cumprod.plot(title='Strategy Cumulative Returns from Actual Trades')
plt.xlabel('Time')
plt.ylabel('Cumulative Return')
plt.grid(True)
plt.show()

