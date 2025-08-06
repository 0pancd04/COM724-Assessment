"""
SQLite Database Integration for Cryptocurrency Data
Handles both yfinance and Binance data with unified structure
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, List, Tuple, Dict
import logging
import os
from contextlib import contextmanager

from .logger import setup_logger

# Setup logging
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
logger = setup_logger("database", os.path.join(LOG_DIR, "database.log"))

# Database configuration
DB_PATH = os.path.join(BASE_DIR, "data", "crypto_data.db")
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

@contextmanager
def get_db_connection():
    """Context manager for database connections"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

class CryptoDatabase:
    """Main database handler for cryptocurrency data"""
    
    def __init__(self):
        self.db_path = DB_PATH
        self.initialize_database()
    
    def initialize_database(self):
        """Create database tables if they don't exist"""
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Check if ohlcv_data table exists and has the correct schema
            cursor.execute("PRAGMA table_info(ohlcv_data)")
            existing_columns = [row[1] for row in cursor.fetchall()]
            
            if 'base_symbol' not in existing_columns and existing_columns:
                # Table exists but missing base_symbol column - add it
                logger.info("[CryptoDatabase.initialize_database] Adding missing base_symbol column to ohlcv_data table")
                cursor.execute("ALTER TABLE ohlcv_data ADD COLUMN base_symbol TEXT")
                
                # Update existing records with base_symbol
                cursor.execute("SELECT DISTINCT ticker FROM ohlcv_data WHERE base_symbol IS NULL")
                tickers_to_update = cursor.fetchall()
                
                from .ticker_mapping import get_base_symbol
                for (ticker,) in tickers_to_update:
                    base_symbol = get_base_symbol(ticker)
                    cursor.execute("UPDATE ohlcv_data SET base_symbol = ? WHERE ticker = ?", (base_symbol, ticker))
                
                conn.commit()
                logger.info(f"[CryptoDatabase.initialize_database] Updated {len(tickers_to_update)} tickers with base_symbol")
            
            # Main OHLCV data table - unified structure for both sources
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ohlcv_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL,  -- Original ticker (BTC-USD or BTCUSDT)
                    base_symbol TEXT NOT NULL,  -- Base symbol (BTC) for cross-platform comparison
                    source TEXT NOT NULL,  -- 'yfinance' or 'binance'
                    timestamp DATETIME NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume REAL NOT NULL,
                    interval TEXT NOT NULL,  -- '1m', '1h', '1d', etc.
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(ticker, source, timestamp, interval)
                )
            """)
            
            # Metadata table for tracking data updates
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS data_metadata (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL,
                    source TEXT NOT NULL,
                    interval TEXT NOT NULL,
                    last_update DATETIME,
                    first_date DATETIME,
                    last_date DATETIME,
                    total_records INTEGER,
                    UNIQUE(ticker, source, interval)
                )
            """)
            
            # Preprocessed data table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS preprocessed_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL,
                    source TEXT NOT NULL,
                    feature_data TEXT NOT NULL,  -- JSON string of flattened features
                    max_days INTEGER,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(ticker, source, max_days)
                )
            """)
            
            # Ticker mapping table for data source compatibility
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ticker_mapping (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    base_symbol TEXT NOT NULL,  -- Base symbol (BTC)
                    yfinance_ticker TEXT,  -- YFinance format (BTC-USD)
                    binance_ticker TEXT,   -- Binance format (BTCUSDT)
                    common_name TEXT,      -- Display name (Bitcoin)
                    is_active BOOLEAN DEFAULT 1,
                    last_updated DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(base_symbol)
                )
            """)
            
            # Model metadata table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS model_metadata (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL,
                    model_type TEXT NOT NULL,
                    feature TEXT NOT NULL,
                    train_start_date DATETIME,
                    train_end_date DATETIME,
                    test_size REAL,
                    metrics TEXT,  -- JSON string of evaluation metrics
                    model_path TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(ticker, model_type, feature)
                )
            """)
            
            # Create indexes for better query performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_ohlcv_ticker ON ohlcv_data(ticker)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_ohlcv_source ON ohlcv_data(source)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_ohlcv_timestamp ON ohlcv_data(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_ohlcv_combined ON ohlcv_data(ticker, source, timestamp)")
            
            conn.commit()
            logger.info("Database initialized successfully")
            
        # Initialize ticker mappings for common cryptocurrencies
        self._initialize_ticker_mappings()
    
    def insert_ohlcv_data(self, df: pd.DataFrame, ticker: str, source: str, interval: str = "1d") -> int:
        """
        Insert OHLCV data into the database
        
        Args:
            df: DataFrame with columns [Open, High, Low, Close, Volume] and DateTime index
            ticker: Cryptocurrency ticker symbol (platform-specific: BTC-USD or BTCUSDT)
            source: Data source ('yfinance' or 'binance')
            interval: Time interval of the data
            
        Returns:
            Number of rows inserted
        """
        try:
            # Import ticker mapping utility
            from .ticker_mapping import get_base_symbol
            
            with get_db_connection() as conn:
                cursor = conn.cursor()
                
                # Get base symbol from ticker
                base_symbol = get_base_symbol(ticker)
                
                # Prepare data for insertion
                records = []
                for timestamp, row in df.iterrows():
                    # Ensure timestamp is datetime
                    if not isinstance(timestamp, pd.Timestamp):
                        timestamp = pd.to_datetime(timestamp)
                    
                    records.append((
                        ticker,
                        base_symbol,  # Add base symbol
                        source,
                        timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                        float(row.get('Open', row.get('open', 0))),
                        float(row.get('High', row.get('high', 0))),
                        float(row.get('Low', row.get('low', 0))),
                        float(row.get('Close', row.get('close', 0))),
                        float(row.get('Volume', row.get('volume', 0))),
                        interval
                    ))
                
                # Use INSERT OR REPLACE to handle duplicates
                cursor.executemany("""
                    INSERT OR REPLACE INTO ohlcv_data 
                    (ticker, base_symbol, source, timestamp, open, high, low, close, volume, interval)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, records)
                
                # Update metadata
                self._update_metadata(conn, ticker, source, interval, df)
                
                conn.commit()
                rows_inserted = cursor.rowcount
                logger.info(f"Inserted {rows_inserted} rows for {ticker} from {source}")
                return rows_inserted
                
        except Exception as e:
            logger.error(f"Error inserting OHLCV data: {e}")
            raise
    
    def get_ohlcv_data(self, ticker: str, source: Optional[str] = None, 
                       start_date: Optional[datetime] = None, 
                       end_date: Optional[datetime] = None,
                       interval: str = "1d") -> pd.DataFrame:
        """
        Retrieve OHLCV data from the database
        
        Args:
            ticker: Cryptocurrency ticker symbol
            source: Data source (optional)
            start_date: Start date for data retrieval (optional)
            end_date: End date for data retrieval (optional)
            interval: Time interval
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            with get_db_connection() as conn:
                query = """
                    SELECT timestamp, open, high, low, close, volume, source
                    FROM ohlcv_data
                    WHERE ticker = ? AND interval = ?
                """
                params = [ticker, interval]
                
                if source:
                    query += " AND source = ?"
                    params.append(source)
                
                if start_date:
                    query += " AND timestamp >= ?"
                    params.append(start_date.strftime("%Y-%m-%d %H:%M:%S"))
                
                if end_date:
                    query += " AND timestamp <= ?"
                    params.append(end_date.strftime("%Y-%m-%d %H:%M:%S"))
                
                query += " ORDER BY timestamp"
                
                df = pd.read_sql_query(query, conn, params=params, parse_dates=['timestamp'])
                
                if not df.empty:
                    df.set_index('timestamp', inplace=True)
                    df.columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Source']
                
                return df
                
        except Exception as e:
            logger.error(f"Error retrieving OHLCV data: {e}")
            return pd.DataFrame()
    
    def get_missing_date_ranges(self, ticker: str, source: str, 
                               start_date: datetime, end_date: datetime,
                               interval: str = "1d") -> List[Tuple[datetime, datetime]]:
        """
        Identify missing date ranges in the database for incremental updates
        
        Returns:
            List of tuples containing (start_date, end_date) for missing ranges
        """
        try:
            with get_db_connection() as conn:
                cursor = conn.cursor()
                
                # Get existing dates
                cursor.execute("""
                    SELECT DISTINCT DATE(timestamp) as date
                    FROM ohlcv_data
                    WHERE ticker = ? AND source = ? AND interval = ?
                    AND timestamp >= ? AND timestamp <= ?
                    ORDER BY date
                """, (ticker, source, interval, 
                     start_date.strftime("%Y-%m-%d"),
                     end_date.strftime("%Y-%m-%d")))
                
                existing_dates = set()
                for row in cursor.fetchall():
                    existing_dates.add(pd.to_datetime(row[0]).date())
                
                # Find missing ranges
                missing_ranges = []
                current_date = start_date.date()
                range_start = None
                
                while current_date <= end_date.date():
                    if current_date not in existing_dates:
                        if range_start is None:
                            range_start = current_date
                    else:
                        if range_start is not None:
                            missing_ranges.append((
                                datetime.combine(range_start, datetime.min.time()),
                                datetime.combine(current_date - timedelta(days=1), datetime.max.time())
                            ))
                            range_start = None
                    
                    current_date += timedelta(days=1)
                
                # Handle case where missing range extends to end_date
                if range_start is not None:
                    missing_ranges.append((
                        datetime.combine(range_start, datetime.min.time()),
                        end_date
                    ))
                
                return missing_ranges
                
        except Exception as e:
            logger.error(f"Error identifying missing date ranges: {e}")
            return [(start_date, end_date)]  # Return full range on error
    
    def get_all_tickers(self, source: Optional[str] = None) -> List[str]:
        """Get all unique tickers in the database"""
        try:
            with get_db_connection() as conn:
                cursor = conn.cursor()
                
                if source:
                    cursor.execute("""
                        SELECT DISTINCT ticker FROM ohlcv_data WHERE source = ?
                        ORDER BY ticker
                    """, (source,))
                else:
                    cursor.execute("""
                        SELECT DISTINCT ticker FROM ohlcv_data
                        ORDER BY ticker
                    """)
                
                return [row[0] for row in cursor.fetchall()]
                
        except Exception as e:
            logger.error(f"Error getting tickers: {e}")
            return []
    
    def save_preprocessed_data(self, ticker: str, source: str, 
                               data: pd.DataFrame, max_days: int):
        """Save preprocessed data to database"""
        try:
            import json
            
            with get_db_connection() as conn:
                cursor = conn.cursor()
                
                # Convert DataFrame to JSON
                data_json = data.to_json()
                
                cursor.execute("""
                    INSERT OR REPLACE INTO preprocessed_data
                    (ticker, source, feature_data, max_days)
                    VALUES (?, ?, ?, ?)
                """, (ticker, source, data_json, max_days))
                
                conn.commit()
                logger.info(f"Saved preprocessed data for {ticker} from {source}")
                
        except Exception as e:
            logger.error(f"Error saving preprocessed data: {e}")
            raise
    
    def get_preprocessed_data(self, ticker: str, source: str, 
                             max_days: int) -> Optional[pd.DataFrame]:
        """Retrieve preprocessed data from database"""
        try:
            import json
            
            with get_db_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT feature_data FROM preprocessed_data
                    WHERE ticker = ? AND source = ? AND max_days = ?
                """, (ticker, source, max_days))
                
                row = cursor.fetchone()
                if row:
                    return pd.read_json(row[0])
                
                return None
                
        except Exception as e:
            logger.error(f"Error retrieving preprocessed data: {e}")
            return None
    
    def _update_metadata(self, conn, ticker: str, source: str, 
                        interval: str, df: pd.DataFrame):
        """Update metadata table with data statistics"""
        try:
            cursor = conn.cursor()
            
            # Get date range from DataFrame
            first_date = df.index.min()
            last_date = df.index.max()
            total_records = len(df)
            
            cursor.execute("""
                INSERT OR REPLACE INTO data_metadata
                (ticker, source, interval, last_update, first_date, last_date, total_records)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP, ?, ?, ?)
            """, (ticker, source, interval, 
                 first_date.strftime("%Y-%m-%d %H:%M:%S"),
                 last_date.strftime("%Y-%m-%d %H:%M:%S"),
                 total_records))
            
        except Exception as e:
            logger.error(f"Error updating metadata: {e}")
    
    def get_data_summary(self) -> pd.DataFrame:
        """Get summary of all data in the database"""
        try:
            with get_db_connection() as conn:
                query = """
                    SELECT 
                        ticker,
                        source,
                        interval,
                        first_date,
                        last_date,
                        total_records,
                        last_update
                    FROM data_metadata
                    ORDER BY ticker, source, interval
                """
                
                return pd.read_sql_query(query, conn, parse_dates=['first_date', 'last_date', 'last_update'])
                
        except Exception as e:
            logger.error(f"Error getting data summary: {e}")
            return pd.DataFrame()
    
    def _initialize_ticker_mappings(self):
        """Initialize ticker mappings for common cryptocurrencies"""
        try:
            from .ticker_mapping import CRYPTO_BASE_SYMBOLS, get_common_name, format_ticker_for_source
            
            with get_db_connection() as conn:
                cursor = conn.cursor()
                
                # Check if we already have mappings
                cursor.execute("SELECT COUNT(*) FROM ticker_mapping")
                count = cursor.fetchone()[0]
                
                if count == 0:  # Only initialize if empty
                    logger.info("[CryptoDatabase._initialize_ticker_mappings] Initializing ticker mappings...")
                    
                    for base_symbol in CRYPTO_BASE_SYMBOLS[:30]:  # Top 30
                        try:
                            yfinance_ticker = format_ticker_for_source(base_symbol, 'yfinance')
                            binance_ticker = format_ticker_for_source(base_symbol, 'binance')
                            common_name = get_common_name(base_symbol)
                            
                            cursor.execute("""
                                INSERT OR REPLACE INTO ticker_mapping 
                                (base_symbol, yfinance_ticker, binance_ticker, common_name, is_active)
                                VALUES (?, ?, ?, ?, 1)
                            """, (base_symbol, yfinance_ticker, binance_ticker, common_name))
                            
                        except Exception as e:
                            logger.warning(f"[CryptoDatabase._initialize_ticker_mappings] Error mapping {base_symbol}: {e}")
                    
                    conn.commit()
                    logger.info(f"[CryptoDatabase._initialize_ticker_mappings] Initialized {len(CRYPTO_BASE_SYMBOLS[:30])} ticker mappings")
                    
        except Exception as e:
            logger.error(f"[CryptoDatabase._initialize_ticker_mappings] Error initializing ticker mappings: {e}")
    
    def get_ticker_mapping(self, base_symbol: str) -> dict:
        """Get ticker mapping for a base symbol"""
        try:
            with get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT base_symbol, yfinance_ticker, binance_ticker, common_name, is_active
                    FROM ticker_mapping 
                    WHERE base_symbol = ? AND is_active = 1
                """, (base_symbol.upper(),))
                
                row = cursor.fetchone()
                if row:
                    return {
                        'base_symbol': row[0],
                        'yfinance_ticker': row[1], 
                        'binance_ticker': row[2],
                        'common_name': row[3],
                        'is_active': bool(row[4])
                    }
                return None
                
        except Exception as e:
            logger.error(f"[CryptoDatabase.get_ticker_mapping] Error getting ticker mapping for {base_symbol}: {e}")
            return None
    
    def find_ticker_by_any_format(self, ticker: str) -> dict:
        """Find ticker mapping by any format (base, yfinance, or binance)"""
        try:
            ticker = ticker.upper().strip()
            
            with get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT base_symbol, yfinance_ticker, binance_ticker, common_name, is_active
                    FROM ticker_mapping 
                    WHERE (base_symbol = ? OR yfinance_ticker = ? OR binance_ticker = ?) 
                    AND is_active = 1
                """, (ticker, ticker, ticker))
                
                row = cursor.fetchone()
                if row:
                    return {
                        'base_symbol': row[0],
                        'yfinance_ticker': row[1], 
                        'binance_ticker': row[2],
                        'common_name': row[3],
                        'is_active': bool(row[4])
                    }
                return None
                
        except Exception as e:
            logger.error(f"[CryptoDatabase.find_ticker_by_any_format] Error finding ticker mapping for {ticker}: {e}")
            return None

# Initialize database instance
crypto_db = CryptoDatabase()