import ccxt
import pandas as pd
import time
from datetime import datetime, timedelta

# Setup
exchange = ccxt.binance()
symbol = 'BTC/USDT'
timeframe = '1m'
limit = 1000
since = exchange.parse8601('2020-07-01T00:00:00Z')
now = exchange.milliseconds()
interval_ms = 60000 * limit  # 1-minute candles * 1000

# Estimate number of batches
total_batches = (now - since) // interval_ms
batch_num = 0
start_time = time.time()

data = []

print("📥 Starting fast download with progress tracking...\n")

while since < now:
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since, limit)
        if not ohlcv:
            break
        data.extend(ohlcv)
        since = ohlcv[-1][0] + 1
    except Exception as e:
        print(f"⚠️ Skipping bad batch: {e}")
        since += interval_ms

    batch_num += 1
    elapsed = time.time() - start_time
    avg_batch_time = elapsed / batch_num
    remaining_batches = total_batches - batch_num
    eta = timedelta(seconds=int(avg_batch_time * remaining_batches))

    # Progress display
    percent = (batch_num / total_batches) * 100
    print(f"\r⏳ Batch {batch_num}/{total_batches} | {percent:.2f}% | Elapsed: {timedelta(seconds=int(elapsed))} | ETA: {eta}", end='')

    time.sleep(0.25)

# Save
df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
df.set_index('timestamp', inplace=True)
df.to_csv('BTC_USDT_1min_since_2020_FAST.csv')

print("\n✅ Done. File saved as BTC_USDT_1min_since_2020.csv")
