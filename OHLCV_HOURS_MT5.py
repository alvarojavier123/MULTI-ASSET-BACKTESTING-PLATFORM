import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timezone
import os

# === Initialize MT5 ===
if not mt5.initialize():
    print("❌ Failed to initialize MT5:", mt5.last_error())
    quit()

# === Create output folder ===
output_folder = "STOCKS-HOURS"
#output_folder = "METALS-HOURS"
#output_folder = "INDEX-HOURS"
#output_folder = "CRYPTO-HOURS"
output_folder = "FOREX-HOURS"

os.makedirs(output_folder, exist_ok=True)

FTMO = ["all forex", "BABA", "PFE", "V", "ZM", "WMT" , "T" , "RACE" , "BAYGn", "ALVG"] # MISSING
# === Define stock tickers (exact names must match those in your MT5 Market Watch) ===
#symbols = ["AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN", "NFLX"]
#symbols = ['XAUUSD']
symbols = ["NFLX", "BAC", "WFC", "ORCL", "INTC", "JPM"]
symbols = ["BTCUSD"]
symbols = ["AUDJPY"]

#symbols = ["US500"]

# === Define time range ===
start = datetime(2020, 1, 1, tzinfo=timezone.utc)
end = datetime.now(timezone.utc)

# === Download and save each stock ===
for symbol in symbols:
    print(f"Downloading: {symbol}")
    
    # Make sure symbol is visible
    if not mt5.symbol_select(symbol, True):
        print(f"⚠️ Could not select symbol {symbol}")
        continue

    # Download data
    rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_H1, start, end)

    if rates is None or len(rates) == 0:
        print(f"⚠️ No data for {symbol}")
        continue

    # Format DataFrame
    df = pd.DataFrame(rates)
    df['timestamp'] = pd.to_datetime(df['time'], unit='s')
    df.rename(columns={
        'open': 'open',
        'high': 'high',
        'low': 'low',
        'close': 'close',
        'tick_volume': 'volume'
    }, inplace=True)

    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

    # Save to CSV
    df.to_csv(f"ASSETS/{output_folder}/{symbol}.csv", index=False)
    print(f"✅ Saved: {symbol}.csv")

# === Shutdown MT5 ===
mt5.shutdown()
