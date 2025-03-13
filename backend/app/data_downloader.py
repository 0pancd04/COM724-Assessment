import requests
import yfinance as yf
import pandas as pd
import logging

from .logger import setup_logger

logger = setup_logger("data_downloader_log", "data_downloader_log.log")

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
        logger.info(f"Downloading data for ticker: {ticker}, period: {period}, interval: {interval}")
        data = yf.download(ticker, period=period, interval=interval)
        if data.empty:
            logger.warning(f"No data found for ticker: {ticker}")
        else:
            logger.info(f"Successfully downloaded data for ticker: {ticker}")
        return data
    except Exception as e:
        logger.error(f"Error downloading data for ticker {ticker}: {e}", exc_info=True)
        raise


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
        # Use a custom user-agent to mimic a browser
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                          "(KHTML, like Gecko) Chrome/90.0.4430.93 Safari/537.36"
        }
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # will raise an HTTPError for bad status
        # Use the response content in pd.read_html
        tables = pd.read_html(response.text)
        df = tables[0]
        if "Symbol" in df.columns:
            tickers = df["Symbol"].tolist()
        elif "Ticker" in df.columns:
            tickers = df["Ticker"].tolist()
        else:
            msg = "Ticker column not found in the table"
            logger.error(msg)
            raise ValueError(msg)
        logger.info(f"Found {len(tickers)} tickers, returning top {top_n}")
        return tickers[:top_n]
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
