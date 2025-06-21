import ccxt
import pandas as pd
import time
from datetime import datetime

# Configurar Binance
exchange = ccxt.binance()
symbol = 'BTC/USDT'
timeframe = '1d'  # ← CAMBIADO a diario
limit = 1000  # máximo por llamada

# Fecha inicial: 1 de enero de 2020
start_date = exchange.parse8601('2020-01-01T00:00:00Z')
now = exchange.milliseconds()

# Lista para guardar los datos
all_ohlcv = []

# Iterar por lotes
since = start_date
while since < now:
    print(f'Descargando desde {exchange.iso8601(since)}...')
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
    
    if not ohlcv:
        break

    all_ohlcv += ohlcv

    # Avanza el tiempo usando la última vela
    since = ohlcv[-1][0] + 1

    # Esperar para evitar rate limits
    time.sleep(1)

# Convertir a DataFrame
df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

# Guardar a CSV
df.to_csv('BTC_USDT_1d_since_2020.csv', index=False)

print("✅ Datos descargados y guardados como BTC_USDT_1d_since_2020.csv")

