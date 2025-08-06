import datetime
from binance.client import Client
import requests
import yfinance as yf
import pandas as pd
import logging
from dotenv import load_dotenv
import os
import re

# Load environment variables from ../.env file
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))


from .logger import setup_logger
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
logger = setup_logger("data_downloader_log", os.path.join(LOG_DIR, "data_downloader_log.log") )

# If no handlers are present, add a default console handler
if not logger.handlers:
    ch = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

def download_data_yfinance(ticker: str, period: str = "5y", interval: str = "1d"):
    """
    Download historical data for a given crypto ticker from Yahoo Finance.

    Args:
        ticker (str): The crypto ticker (e.g. "BTC-USD").
        period (str): Time period for data retrieval (default "5y").
        interval (str): Data interval (default "1d" for daily).

    Returns:
        pd.DataFrame: DataFrame containing open, high, low, close, volume, etc.
    """
    try:
        ticker = str(ticker).strip()
        if not ticker or ticker.lower() == "nan":
            logger.warning(f"Skipping invalid ticker: {ticker}")
            return pd.DataFrame()

        logger.info(f"Downloading data for ticker: {ticker}, period: {period}, interval: {interval}")
        data = yf.download(ticker, period=period, interval=interval)

        if data.empty:
            logger.warning(f"No data found for ticker: {ticker}")
            return pd.DataFrame()
        else:
            logger.info(f"Successfully downloaded data for ticker: {ticker}, shape: {data.shape}, columns: {list(data.columns)}")
            
            # Handle multi-level columns immediately after download
            if hasattr(data.columns, 'nlevels') and data.columns.nlevels > 1:
                logger.info(f"[download_data_yfinance] Multi-level columns detected, flattening...")
                if data.columns.nlevels == 2:
                    data.columns = data.columns.droplevel(0)  # Remove ticker level
                logger.info(f"[download_data_yfinance] Flattened columns: {list(data.columns)}")

        return data

    except Exception as e:
        logger.error(f"Error downloading data for ticker {ticker}: {e}", exc_info=True)
        return pd.DataFrame()


def get_top_crypto_tickers(url: str = "https://finance.yahoo.com/markets/crypto/all/?start=0&count=50", top_n: int = 30):
    """
    Scrapes the Yahoo Finance crypto page to get a list of top crypto tickers.

    Args:
        url (str): URL to fetch the crypto table.
        top_n (int): Number of top tickers to return (default 30).

    Returns:
        list: List of ticker symbols (strings).
    """
    try:
        logger.info(f"Fetching top crypto tickers from {url}")
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                          "(KHTML, like Gecko) Chrome/90.0.4430.93 Safari/537.36"
        }
        response = requests.get(url, headers=headers)
        response.raise_for_status()

        tables = pd.read_html(response.text)
        df = tables[0]

        if "Symbol" in df.columns:
            raw_tickers = df["Symbol"]
        elif "Ticker" in df.columns:
            raw_tickers = df["Ticker"]
        else:
            msg = "Ticker column not found in the table"
            logger.error(msg)
            raise ValueError(msg)

        # Clean and filter tickers
        tickers = (
            raw_tickers.dropna()
            .astype(str)
            .str.strip()
            .unique()
            .tolist()
        )

        # Filter only properly formatted crypto tickers like BTC-USD
        valid_tickers = [t for t in tickers if re.fullmatch(r"[A-Z0-9]{2,10}-USD", t)]

        logger.info(f"Filtered {len(valid_tickers)} valid tickers, returning top {top_n}")
        return valid_tickers[:top_n]

    except Exception as e:
        logger.error(f"Error fetching tickers from {url}: {e}", exc_info=True)
        return []
    

def flatten_ticker_data(df: pd.DataFrame) -> pd.Series:
    """
    Flatten a DataFrame of daily data (with columns Open, High, Low, Close, Volume)
    into a single row. For each date, the five metrics are concatenated in the order:
    Open, High, Low, Close, Volume.
    
    Args:
        df (pd.DataFrame): DataFrame with a datetime index and at least the columns:
                           "Open", "High", "Low", "Close", "Volume".
                           
    Returns:
        pd.Series: A flattened Series where the index labels are formed as
                   "<date>_<metric>".
    """
    try:
        logger.info("Flattening ticker data")
        # Ensure we only work with the required columns
        cols = ["Open", "High", "Low", "Close", "Volume"]
        df = df[cols].copy()

        # Sort by date to ensure consistent ordering
        df.sort_index(inplace=True)

        # Create new column names by combining date and metric
        new_columns = []
        for date in df.index:
            date_str = pd.to_datetime(date).strftime("%Y-%m-%d")
            for col in cols:
                new_columns.append(f"{date_str}_{col}")

        # Flatten the DataFrame values row-wise
        flattened = df.to_numpy().flatten()
        logger.info("Successfully flattened ticker data")
        return pd.Series(flattened, index=new_columns)
    except Exception as e:
        logger.error(f"Error flattening data: {e}", exc_info=True)
        raise

def binance_data():
    from .binance_client import binance_client
    
    if binance_client is None:
        raise ValueError("Binance API credentials not found in .env file.")

    client = binance_client

    # Define the symbol for BTC/USDT pair
    symbol = 'BTCUSDT'

    # Define custom start and end time
    start_time = datetime.datetime(2024, 3, 15, 0, 0, 0)
    end_time = datetime.datetime(2024, 6, 15, 0, 0, 0)

    klines = client.get_historical_klines(symbol=symbol, interval=Client.KLINE_INTERVAL_1MINUTE, start_str=str(start_time), end_str=str(end_time))

    # Convert the data into a pandas dataframe for easier manipulation
    df_M = pd.DataFrame(klines, columns=['Open Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close Time', 'Quote Asset Volume', 'Number of Trades', 'Taker Buy Base Asset Volume', 'Taker Buy Quote Asset Volume', 'Ignore'])


    columns_to_convert = ['Open', 'High', 'Low', 'Close', 'Volume', 'Quote Asset Volume', 'Number of Trades', 'Taker Buy Base Asset Volume', 'Taker Buy Quote Asset Volume']

    for col in columns_to_convert:
        df_M[col] = df_M[col].astype(float)

    return df_M


def get_top_30_coins():
    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {
        "vs_currency": "usd",
        "order": "market_cap_desc",
        "per_page": 100,
        "page": 1
    }
    response = requests.get(url, params=params)
    data = response.json()

    # Extract coin symbols and append USDT for Binance compatibility
    top_30_symbols = [f"{coin['symbol'].upper()}USDT" for coin in data]
    return top_30_symbols