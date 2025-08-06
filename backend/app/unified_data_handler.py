"""
Unified Data Handler for YFinance and Binance Data Sources
Ensures consistent data structure and incremental updates
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Tuple
import logging
import os

from .database import crypto_db, CryptoDatabase
from .data_downloader import download_data_yfinance, get_top_30_coins
from .download_binance_data import download_binance_ohlcv
from .binance_client import binance_client
from .logger import setup_logger
from .ticker_mapping import get_base_symbol, format_ticker_for_source, get_top_30_base_symbols

# Setup logging
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
logger = setup_logger("unified_data_handler", os.path.join(LOG_DIR, "unified_data_handler.log"))

class UnifiedDataHandler:
    """Handles data from both YFinance and Binance with unified structure"""
    
    def __init__(self):
        self.db = crypto_db
        self.supported_sources = ['yfinance', 'binance']
        self.interval_mapping = {
            'yfinance': {
                '1m': '1m',
                '5m': '5m',
                '15m': '15m',
                '30m': '30m',
                '1h': '60m',
                '1d': '1d',
                '1wk': '1wk',
                '1mo': '1mo'
            },
            'binance': {
                '1m': '1m',
                '5m': '5m',
                '15m': '15m',
                '30m': '30m',
                '1h': '1h',
                '1d': '1d',
                '1wk': '1w',
                '1mo': '1M'
            }
        }
    
    def get_base_symbol(self, ticker: str) -> str:
        """
        Extract base symbol from any ticker format
        
        Args:
            ticker: Ticker in any format (BTC, BTC-USD, BTCUSDT, etc.)
            
        Returns:
            Base symbol (e.g., BTC)
        """
        return get_base_symbol(ticker)
    
    def normalize_ticker(self, ticker: str, source: str) -> str:
        """
        Normalize ticker symbol for different sources
        
        Args:
            ticker: Raw ticker symbol (can be in any format)
            source: Data source ('yfinance' or 'binance')
            
        Returns:
            Normalized ticker symbol for the specific source
        """
        # Use the ticker mapping utility
        base_symbol = get_base_symbol(ticker)
        return format_ticker_for_source(base_symbol, source)
    
    def denormalize_ticker(self, ticker: str) -> str:
        """
        Convert ticker to standard format (e.g., BTC)
        Alias for get_base_symbol for backward compatibility
        
        Args:
            ticker: Normalized ticker symbol
            
        Returns:
            Standard ticker symbol
        """
        return get_base_symbol(ticker)
    
    def unify_dataframe_structure(self, df: pd.DataFrame, source: str) -> pd.DataFrame:
        """
        Unify DataFrame structure from different sources
        
        Args:
            df: Raw DataFrame from data source
            source: Data source
            
        Returns:
            Unified DataFrame with consistent structure
        """
        try:
            logger.info(f"[UnifiedDataHandler.unify_dataframe_structure] Starting unification for {source}, input shape: {df.shape}")
            
            # Create a copy to avoid modifying original
            unified_df = df.copy()
            
            # Handle multi-level columns (common with yfinance)
            if hasattr(unified_df.columns, 'nlevels') and unified_df.columns.nlevels > 1:
                logger.info(f"[UnifiedDataHandler.unify_dataframe_structure] Multi-level columns detected, flattening...")
                # For yfinance, typically the first level has the ticker and second level has OHLCV
                # We only want the OHLCV part
                if unified_df.columns.nlevels == 2:
                    unified_df.columns = unified_df.columns.droplevel(0)  # Remove ticker level
                else:
                    # Join multi-level columns with underscore
                    unified_df.columns = ['_'.join(col).strip() for col in unified_df.columns.values]
                logger.info(f"[UnifiedDataHandler.unify_dataframe_structure] Flattened columns: {list(unified_df.columns)}")
            
            # Ensure index is datetime
            if not isinstance(unified_df.index, pd.DatetimeIndex):
                if 'Date' in unified_df.columns:
                    unified_df.set_index('Date', inplace=True)
                elif 'timestamp' in unified_df.columns:
                    unified_df.set_index('timestamp', inplace=True)
                elif 'Open Time' in unified_df.columns:
                    unified_df.set_index('Open Time', inplace=True)
            
            # Safely convert index to datetime
            try:
                if not isinstance(unified_df.index, pd.DatetimeIndex):
                    unified_df.index = pd.to_datetime(unified_df.index)
                unified_df.index.name = 'timestamp'
            except Exception as idx_error:
                logger.error(f"[UnifiedDataHandler.unify_dataframe_structure] Error converting index to datetime: {idx_error}")
                logger.error(f"[UnifiedDataHandler.unify_dataframe_structure] Index type: {type(unified_df.index)}, Index values: {unified_df.index[:5] if len(unified_df.index) > 0 else 'Empty'}")
                raise
            
            # Standardize column names
            logger.info(f"[UnifiedDataHandler.unify_dataframe_structure] Before column mapping: columns={list(unified_df.columns)}")
            
            column_mapping = {
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume',
                'Open Price': 'Open',
                'High Price': 'High',
                'Low Price': 'Low',
                'Close Price': 'Close',
                'Volume': 'Volume'
            }
            
            unified_df.rename(columns=column_mapping, inplace=True)
            logger.info(f"[UnifiedDataHandler.unify_dataframe_structure] After column mapping: columns={list(unified_df.columns)}")
            
            # Ensure we have all required columns
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in required_columns:
                if col not in unified_df.columns:
                    logger.warning(f"Missing column {col} in {source} data")
                    unified_df[col] = 0.0
            
            # Select only required columns
            unified_df = unified_df[required_columns]
            
            # Convert to numeric
            for col in required_columns:
                unified_df[col] = pd.to_numeric(unified_df[col], errors='coerce')
            
            # Remove any rows with all NaN values
            unified_df.dropna(how='all', inplace=True)
            
            # Forward fill then backward fill remaining NaN values
            unified_df.ffill(inplace=True)
            unified_df.bfill(inplace=True)
            
            # Sort by date
            unified_df.sort_index(inplace=True)
            
            return unified_df
            
        except Exception as e:
            logger.error(f"[UnifiedDataHandler.unify_dataframe_structure] Error unifying DataFrame structure: {e}")
            logger.error(f"[UnifiedDataHandler.unify_dataframe_structure] DataFrame shape: {df.shape}, columns: {list(df.columns)}")
            logger.error(f"[UnifiedDataHandler.unify_dataframe_structure] DataFrame index: {df.index[:3] if len(df) > 0 else 'Empty'}")
            logger.error(f"[UnifiedDataHandler.unify_dataframe_structure] DataFrame dtypes: {df.dtypes}")
            import traceback
            logger.error(f"[UnifiedDataHandler.unify_dataframe_structure] Full traceback: {traceback.format_exc()}")
            raise
    
    def download_and_store_data(self, ticker: str, source: str, 
                               period: str = "90d", interval: str = "1d",
                               update_missing: bool = True) -> pd.DataFrame:
        """
        Download data from source and store in database
        
        Args:
            ticker: Cryptocurrency ticker
            source: Data source ('yfinance' or 'binance')
            period: Time period for data
            interval: Data interval
            update_missing: Whether to update only missing data
            
        Returns:
            Downloaded and stored DataFrame
        """
        try:
            base_symbol = get_base_symbol(ticker)
            normalized_ticker = self.normalize_ticker(ticker, source)
            
            logger.info(f"[UnifiedDataHandler.download_and_store_data] Downloading {base_symbol} from {source} as {normalized_ticker}")
            
            # Calculate date range
            end_date = datetime.now()
            if source == 'yfinance':
                # YFinance accepts period strings like '5y', '90d'
                start_date = None  # Will be handled by yfinance
            else:
                # Binance needs explicit date range
                try:
                    days = int(period.rstrip('d'))
                except:
                    days = 90
                start_date = end_date - timedelta(days=days)
            
            # Check for missing data if update_missing is True
            if update_missing and start_date:
                missing_ranges = self.db.get_missing_date_ranges(
                    normalized_ticker, source, start_date, end_date, interval
                )
                
                if not missing_ranges:
                    logger.info(f"[UnifiedDataHandler.download_and_store_data] No missing data for {normalized_ticker} from {source}")
                    return self.db.get_ohlcv_data(normalized_ticker, source, start_date, end_date, interval)
            else:
                missing_ranges = [(start_date, end_date)] if start_date else [None]
            
            all_data = []
            
            # Download data for each missing range
            for range_item in missing_ranges:
                if source == 'yfinance':
                    df = download_data_yfinance(normalized_ticker, period=period, interval=interval)
                    
                elif source == 'binance':
                    if not binance_client:
                        logger.error("[UnifiedDataHandler.download_and_store_data] Binance client not initialized")
                        continue
                    
                    if range_item:
                        range_start, range_end = range_item
                        days = (range_end - range_start).days
                    else:
                        days = 90
                    
                    # Download for this specific ticker (binance needs base symbol without suffix)
                    df = download_binance_ohlcv([base_symbol], days=days, interval=interval)
                
                else:
                    logger.error(f"Unsupported source: {source}")
                    continue
                
                if df is not None and not df.empty:
                    # Debug: Log DataFrame info before unification
                    logger.info(f"[UnifiedDataHandler.download_and_store_data] Raw DataFrame from {source}: shape={df.shape}, columns={list(df.columns)}, index_type={type(df.index)}")
                    if len(df) > 0:
                        logger.info(f"[UnifiedDataHandler.download_and_store_data] First few rows: {df.head(2).to_dict()}")
                    
                    # Unify structure
                    df = self.unify_dataframe_structure(df, source)
                    
                    # Store in database with normalized ticker
                    self.db.insert_ohlcv_data(df, normalized_ticker, source, interval)
                    
                    all_data.append(df)
                else:
                    logger.warning(f"[UnifiedDataHandler.download_and_store_data] No data retrieved for {ticker} from {source}")
            
            # Combine all data
            if all_data:
                combined_df = pd.concat(all_data)
                combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
                combined_df.sort_index(inplace=True)
                return combined_df
            
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"[UnifiedDataHandler.download_and_store_data] Error downloading and storing data: {e}")
            raise
    
    def download_top_30_cryptos(self, source: str = 'yfinance', 
                                period: str = "90d", interval: str = "1d") -> Dict[str, pd.DataFrame]:
        """
        Download data for top 30 cryptocurrencies
        
        Args:
            source: Data source ('yfinance' or 'binance')
            period: Time period
            interval: Data interval
            
        Returns:
            Dictionary of DataFrames keyed by base symbol
        """
        try:
            results = {}
            
            # Get top 30 base symbols
            base_symbols = get_top_30_base_symbols()
            
            if not base_symbols:
                logger.error("Could not retrieve top 30 symbols")
                return results
            
            successful_downloads = 0
            for base_symbol in base_symbols:
                if successful_downloads >= 30:
                    break
                
                try:
                    # Format ticker for the specific source
                    ticker = format_ticker_for_source(base_symbol, source)
                    logger.info(f"Downloading {ticker} from {source} ({successful_downloads + 1}/30)")
                    
                    df = self.download_and_store_data(
                        ticker, source, period, interval, update_missing=True
                    )
                    
                    if not df.empty:
                        # Use base symbol as key for consistency
                        results[base_symbol] = df
                        successful_downloads += 1
                        logger.info(f"Successfully downloaded {base_symbol} as {ticker}")
                    
                except Exception as e:
                    logger.error(f"Error downloading {base_symbol}: {e}")
                    continue
            
            logger.info(f"Downloaded data for {successful_downloads} cryptocurrencies from {source}")
            return results
            
        except Exception as e:
            logger.error(f"Error downloading top 30 cryptos: {e}")
            return {}
    
    def get_combined_data(self, ticker: str, start_date: Optional[datetime] = None,
                         end_date: Optional[datetime] = None, 
                         interval: str = "1d") -> pd.DataFrame:
        """
        Get combined data from all sources for a ticker
        
        Args:
            ticker: Cryptocurrency ticker
            start_date: Start date
            end_date: End date
            interval: Data interval
            
        Returns:
            Combined DataFrame from all sources
        """
        try:
            standard_ticker = self.denormalize_ticker(ticker)
            
            # Get data from database
            df = self.db.get_ohlcv_data(standard_ticker, None, start_date, end_date, interval)
            
            if df.empty:
                logger.warning(f"No data found for {ticker}")
                
            return df
            
        except Exception as e:
            logger.error(f"Error getting combined data: {e}")
            return pd.DataFrame()
    
    def flatten_ticker_data(self, df: pd.DataFrame) -> pd.Series:
        """
        Flatten a DataFrame into a single row for clustering/modeling
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Flattened Series
        """
        try:
            # Ensure we have the required columns
            cols = ["Open", "High", "Low", "Close", "Volume"]
            df = df[cols].copy()
            
            # Sort by date
            df.sort_index(inplace=True)
            
            # Create new column names
            new_columns = []
            for date in df.index:
                date_str = pd.to_datetime(date).strftime("%Y-%m-%d")
                for col in cols:
                    new_columns.append(f"{date_str}_{col}")
            
            # Flatten the DataFrame
            flattened = df.to_numpy().flatten()
            
            return pd.Series(flattened, index=new_columns)
            
        except Exception as e:
            logger.error(f"Error flattening data: {e}")
            raise
    
    def prepare_data_for_preprocessing(self, source: str = 'both', 
                                       interval: str = "1d") -> pd.DataFrame:
        """
        Prepare data for preprocessing step
        
        Args:
            source: Data source ('yfinance', 'binance', or 'both')
            interval: Data interval
            
        Returns:
            DataFrame ready for preprocessing
        """
        try:
            all_data = {}
            
            if source in ['yfinance', 'both']:
                yf_tickers = self.db.get_all_tickers('yfinance')
                logger.info(f"Found {len(yf_tickers)} tickers for yfinance: {yf_tickers[:5]}...")
                
                for ticker in yf_tickers:
                    df = self.db.get_ohlcv_data(ticker, 'yfinance', interval=interval)
                    if not df.empty:
                        # Use base symbol for consistent naming
                        base_symbol = get_base_symbol(ticker)
                        flattened = self.flatten_ticker_data(df)
                        all_data[f"{base_symbol}_yf"] = flattened
                        logger.debug(f"Added {base_symbol} from yfinance with {len(df)} records")
            
            if source in ['binance', 'both']:
                bn_tickers = self.db.get_all_tickers('binance')
                logger.info(f"Found {len(bn_tickers)} tickers for binance: {bn_tickers[:5]}...")
                
                for ticker in bn_tickers:
                    df = self.db.get_ohlcv_data(ticker, 'binance', interval=interval)
                    if not df.empty:
                        # Use base symbol for consistent naming
                        base_symbol = get_base_symbol(ticker)
                        flattened = self.flatten_ticker_data(df)
                        all_data[f"{base_symbol}_bn"] = flattened
                        logger.debug(f"Added {base_symbol} from binance with {len(df)} records")
            
            if all_data:
                combined_df = pd.DataFrame.from_dict(all_data, orient='index')
                logger.info(f"Prepared data for {source} with shape: {combined_df.shape}")
                return combined_df
            
            logger.warning(f"No data found for source: {source}")
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Error preparing data for preprocessing: {e}", exc_info=True)
            return pd.DataFrame()

# Initialize unified handler instance
unified_handler = UnifiedDataHandler()