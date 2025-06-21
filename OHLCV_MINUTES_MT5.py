import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta
import os

# === CONFIGURACIÓN ===
symbols = ["AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN"]
symbols = ['XAUUSD']
symbols = ['US500']
symbols = ["ORCL", "INTC", "JPM"] # MISSING
symbols = ["JPM"]
symbols = ["BTCUSD"]

output_folder = "STOCKS-MINUTES"
#output_folder = "METALS-MINUTES"
#output_folder = "INDEX-MINUTES"
output_folder = "CRYPTO-MINUTES"

start_date = datetime(2020, 1, 1)
end_date = datetime.now()
timeframe = mt5.TIMEFRAME_M1  # 1 minuto

# === Inicializar MT5 ===
if not mt5.initialize():
    print("❌ No se pudo inicializar MT5:", mt5.last_error())
    quit()

# === Crear carpeta de salida ===
os.makedirs(output_folder, exist_ok=True)

# === Descargar datos por símbolo ===
for symbol in symbols:
    print(f"📥 Descargando {symbol}")
    all_data = []
    from_date = start_date

    while from_date < end_date:
        to_date = from_date + timedelta(days=1)

        rates = mt5.copy_rates_range(symbol, timeframe, from_date, to_date)

        if rates is not None and len(rates) > 1:
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df = df[['time', 'open', 'high', 'low', 'close', 'tick_volume']]
            df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            all_data.append(df)
            print(f"✅ {symbol} {from_date.date()} - {len(df)} rows")
        else:
            print(f"⚠️ {symbol} {from_date.date()} - sin datos")

        from_date = to_date

    # === Guardar CSV si hay datos ===
    if all_data:
        final_df = pd.concat(all_data)
        final_df.to_csv(f"{output_folder}/{symbol}.csv", index=False)
        print(f"✅ Guardado: {symbol} ({len(final_df)} filas)\n")
    else:
        print(f"❌ No se encontró data para {symbol}\n")

# === Finalizar conexión ===
mt5.shutdown()
