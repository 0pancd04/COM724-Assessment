"""
Analysis Storage Module
Stores all analysis results in database for API access
"""

import json
import sqlite3
from datetime import datetime
from typing import Dict, Any, Optional, List
import pandas as pd
import logging
import os

from .logger import setup_logger
from .database import get_db_connection, DB_PATH

# Setup logging
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
logger = setup_logger("analysis_storage", os.path.join(LOG_DIR, "analysis_storage.log"))

class AnalysisStorage:
    """Store and retrieve analysis results from database"""
    
    def __init__(self):
        self.db_path = DB_PATH
        self.initialize_tables()
    
    def initialize_tables(self):
        """Create analysis storage tables"""
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # EDA results table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS eda_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL,
                    analysis_type TEXT NOT NULL,
                    chart_type TEXT,
                    data TEXT NOT NULL,  -- JSON data
                    chart_html TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(ticker, analysis_type, chart_type)
                )
            """)
            
            # Correlation results table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS correlation_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    tickers TEXT NOT NULL,  -- Comma-separated tickers
                    correlation_matrix TEXT NOT NULL,  -- JSON data
                    top_positive TEXT,  -- JSON array
                    top_negative TEXT,  -- JSON array
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Clustering results table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS clustering_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    algorithm TEXT NOT NULL,
                    n_clusters INTEGER,
                    cluster_assignments TEXT NOT NULL,  -- JSON data
                    silhouette_score REAL,
                    chart_data TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Prediction results table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS prediction_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL,
                    model_type TEXT NOT NULL,
                    prediction_date DATE NOT NULL,
                    predicted_value REAL,
                    actual_value REAL,
                    confidence REAL,
                    metrics TEXT,  -- JSON data
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(ticker, model_type, prediction_date)
                )
            """)
            
            # Trading signals table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trading_signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL,
                    signal_date DATE NOT NULL,
                    signal_type TEXT NOT NULL,  -- 'BUY', 'SELL', 'HOLD'
                    price REAL,
                    predicted_price REAL,
                    expected_profit REAL,
                    confidence REAL,
                    indicators TEXT,  -- JSON data (RSI, MACD, etc.)
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(ticker, signal_date)
                )
            """)
            
            # What-if scenarios table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS whatif_scenarios (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL,
                    scenario_name TEXT NOT NULL,
                    parameters TEXT NOT NULL,  -- JSON data
                    results TEXT NOT NULL,  -- JSON data
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # RSS feed cache table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS rss_feed_cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT,
                    title TEXT NOT NULL,
                    link TEXT NOT NULL,
                    description TEXT,
                    published_date DATETIME,
                    source TEXT,
                    sentiment REAL,  -- Sentiment score if analyzed
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(link)
                )
            """)
            
            # Pipeline results summary table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS pipeline_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pipeline_id TEXT NOT NULL UNIQUE,
                    tickers TEXT NOT NULL,
                    sources TEXT NOT NULL,
                    start_time DATETIME,
                    end_time DATETIME,
                    total_duration REAL,
                    steps_completed INTEGER,
                    steps_failed INTEGER,
                    results_summary TEXT,  -- JSON data
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.commit()
            logger.info("Analysis storage tables initialized")
    
    def store_eda_results(self, ticker: str, analysis_type: str, 
                         data: Dict, chart_type: str = None, 
                         chart_html: str = None):
        """Store EDA analysis results"""
        try:
            with get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO eda_results 
                    (ticker, analysis_type, chart_type, data, chart_html)
                    VALUES (?, ?, ?, ?, ?)
                """, (ticker, analysis_type, chart_type, 
                     json.dumps(data), chart_html))
                conn.commit()
                logger.info(f"Stored EDA results for {ticker} - {analysis_type}")
                return True
        except Exception as e:
            logger.error(f"Error storing EDA results: {e}")
            return False
    
    def get_eda_results(self, ticker: str, analysis_type: str = None) -> List[Dict]:
        """Retrieve EDA results"""
        try:
            with get_db_connection() as conn:
                cursor = conn.cursor()
                
                if analysis_type:
                    cursor.execute("""
                        SELECT analysis_type, chart_type, data, chart_html, created_at
                        FROM eda_results
                        WHERE ticker = ? AND analysis_type = ?
                        ORDER BY created_at DESC
                    """, (ticker, analysis_type))
                else:
                    cursor.execute("""
                        SELECT analysis_type, chart_type, data, chart_html, created_at
                        FROM eda_results
                        WHERE ticker = ?
                        ORDER BY created_at DESC
                    """, (ticker,))
                
                results = []
                for row in cursor.fetchall():
                    results.append({
                        'analysis_type': row[0],
                        'chart_type': row[1],
                        'data': json.loads(row[2]),
                        'chart_html': row[3],
                        'created_at': row[4]
                    })
                
                return results
        except Exception as e:
            logger.error(f"Error retrieving EDA results: {e}")
            return []
    
    def store_correlation_results(self, tickers: List[str], correlation_matrix: Dict,
                                 top_positive: List, top_negative: List):
        """Store correlation analysis results"""
        try:
            with get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO correlation_results 
                    (tickers, correlation_matrix, top_positive, top_negative)
                    VALUES (?, ?, ?, ?)
                """, (','.join(tickers), json.dumps(correlation_matrix),
                     json.dumps(top_positive), json.dumps(top_negative)))
                conn.commit()
                logger.info(f"Stored correlation results for {tickers}")
                return True
        except Exception as e:
            logger.error(f"Error storing correlation results: {e}")
            return False
    
    def store_prediction_results(self, ticker: str, model_type: str,
                                prediction_date: str, predicted_value: float,
                                confidence: float = None, metrics: Dict = None):
        """Store prediction results"""
        try:
            with get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO prediction_results 
                    (ticker, model_type, prediction_date, predicted_value, confidence, metrics)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (ticker, model_type, prediction_date, predicted_value,
                     confidence, json.dumps(metrics) if metrics else None))
                conn.commit()
                logger.info(f"Stored prediction for {ticker} on {prediction_date}")
                return True
        except Exception as e:
            logger.error(f"Error storing prediction results: {e}")
            return False
    
    def store_trading_signal(self, ticker: str, signal_date: str, 
                           signal_type: str, price: float,
                           predicted_price: float = None,
                           expected_profit: float = None,
                           confidence: float = None,
                           indicators: Dict = None):
        """Store trading signals"""
        try:
            with get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO trading_signals 
                    (ticker, signal_date, signal_type, price, predicted_price, 
                     expected_profit, confidence, indicators)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (ticker, signal_date, signal_type, price,
                     predicted_price, expected_profit, confidence,
                     json.dumps(indicators) if indicators else None))
                conn.commit()
                logger.info(f"Stored {signal_type} signal for {ticker} on {signal_date}")
                return True
        except Exception as e:
            logger.error(f"Error storing trading signal: {e}")
            return False
    
    def get_trading_signals(self, ticker: str = None, 
                           start_date: str = None,
                           end_date: str = None) -> List[Dict]:
        """Retrieve trading signals"""
        try:
            with get_db_connection() as conn:
                cursor = conn.cursor()
                
                query = "SELECT * FROM trading_signals WHERE 1=1"
                params = []
                
                if ticker:
                    query += " AND ticker = ?"
                    params.append(ticker)
                
                if start_date:
                    query += " AND signal_date >= ?"
                    params.append(start_date)
                
                if end_date:
                    query += " AND signal_date <= ?"
                    params.append(end_date)
                
                query += " ORDER BY signal_date DESC"
                
                cursor.execute(query, params)
                
                columns = [desc[0] for desc in cursor.description]
                results = []
                for row in cursor.fetchall():
                    result = dict(zip(columns, row))
                    if result.get('indicators'):
                        result['indicators'] = json.loads(result['indicators'])
                    results.append(result)
                
                return results
        except Exception as e:
            logger.error(f"Error retrieving trading signals: {e}")
            return []
    
    def store_whatif_scenario(self, ticker: str, scenario_name: str,
                             parameters: Dict, results: Dict):
        """Store what-if scenario results"""
        try:
            with get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO whatif_scenarios 
                    (ticker, scenario_name, parameters, results)
                    VALUES (?, ?, ?, ?)
                """, (ticker, scenario_name, json.dumps(parameters),
                     json.dumps(results)))
                conn.commit()
                logger.info(f"Stored what-if scenario: {scenario_name} for {ticker}")
                return cursor.lastrowid
        except Exception as e:
            logger.error(f"Error storing what-if scenario: {e}")
            return None
    
    def store_rss_feed(self, ticker: str, title: str, link: str,
                      description: str = None, published_date: str = None,
                      source: str = None, sentiment: float = None):
        """Store RSS feed items"""
        try:
            with get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR IGNORE INTO rss_feed_cache 
                    (ticker, title, link, description, published_date, source, sentiment)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (ticker, title, link, description, published_date, source, sentiment))
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Error storing RSS feed: {e}")
            return False
    
    def get_rss_feeds(self, ticker: str = None, limit: int = 50) -> List[Dict]:
        """Retrieve RSS feed items"""
        try:
            with get_db_connection() as conn:
                cursor = conn.cursor()
                
                if ticker:
                    cursor.execute("""
                        SELECT * FROM rss_feed_cache
                        WHERE ticker = ? OR ticker IS NULL
                        ORDER BY published_date DESC
                        LIMIT ?
                    """, (ticker, limit))
                else:
                    cursor.execute("""
                        SELECT * FROM rss_feed_cache
                        ORDER BY published_date DESC
                        LIMIT ?
                    """, (limit,))
                
                columns = [desc[0] for desc in cursor.description]
                results = []
                for row in cursor.fetchall():
                    results.append(dict(zip(columns, row)))
                
                return results
        except Exception as e:
            logger.error(f"Error retrieving RSS feeds: {e}")
            return []
    
    def store_pipeline_results(self, pipeline_id: str, tickers: List[str],
                              sources: List[str], start_time: datetime,
                              end_time: datetime, duration: float,
                              steps_completed: int, steps_failed: int,
                              results_summary: Dict):
        """Store pipeline execution results"""
        try:
            with get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO pipeline_results 
                    (pipeline_id, tickers, sources, start_time, end_time, 
                     total_duration, steps_completed, steps_failed, results_summary)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (pipeline_id, ','.join(tickers), ','.join(sources),
                     start_time.isoformat(), end_time.isoformat(),
                     duration, steps_completed, steps_failed,
                     json.dumps(results_summary)))
                conn.commit()
                logger.info(f"Stored pipeline results: {pipeline_id}")
                return True
        except Exception as e:
            logger.error(f"Error storing pipeline results: {e}")
            return False

# Initialize storage instance
analysis_storage = AnalysisStorage()
