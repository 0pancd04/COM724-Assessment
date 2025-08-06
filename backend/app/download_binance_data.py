import os
import datetime
import pandas as pd
from binance.client import Client
from .binance_client import binance_client  # This should be your configured client

def download_binance_ohlcv(symbols, days=90, interval='1h', save_dir="data/binance_kline_data"):
    """
    Download OHLCV data for a list of symbols over a specified date range and interval.
    
    Args:
        symbols (list): List of symbols like ['BTC', 'ETH', 'SOL']
        days (int): Number of past days to fetch data for
        interval (str): Kline interval ('1m', '5m', '15m', '1h', '1d', etc.)
        save_dir (str): Directory to save CSV files
    """

    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Define end and start times
    end_time = datetime.datetime.now()
    start_time = end_time - datetime.timedelta(days=days)

    # Binance expects the symbol to be like 'BTCUSDT'
    for coin in symbols:
        # Check if symbol already ends with USDT
        if coin.upper().endswith('USDT'):
            symbol = coin.upper()
        else:
            symbol = f"{coin.upper()}USDT"
        print(f"Downloading {symbol} from {start_time} to {end_time} at {interval} interval")

        try:
            klines = binance_client.get_historical_klines(
                symbol=symbol,
                interval=interval,
                start_str=start_time.strftime("%Y-%m-%d %H:%M:%S"),
                end_str=end_time.strftime("%Y-%m-%d %H:%M:%S")
            )

            df = pd.DataFrame(klines, columns=[
                'Open Time', 'Open', 'High', 'Low', 'Close', 'Volume',
                'Close Time', 'Quote Asset Volume', 'Number of Trades',
                'Taker Buy Base Asset Volume', 'Taker Buy Quote Asset Volume', 'Ignore'
            ])

            # Convert numeric columns
            numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            df[numeric_cols] = df[numeric_cols].astype(float)

            # Convert time columns
            df['Open Time'] = pd.to_datetime(df['Open Time'], unit='ms')
            df['Close Time'] = pd.to_datetime(df['Close Time'], unit='ms')

            # Save to CSV
            filename = os.path.join(save_dir, f"{symbol}_{interval}.csv")
            df.to_csv(filename, index=False)
            print(f"Saved {len(df)} rows to {filename}\n")
            
            return df
        except Exception as e:
            print(f"Error downloading {symbol}: {e}")
