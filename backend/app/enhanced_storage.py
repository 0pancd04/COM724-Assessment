"""
Enhanced Storage Module with MongoDB-like Interface for SQLite
Provides flexible JSON storage while maintaining SQLite compatibility
"""

import json
import sqlite3
from datetime import datetime
from typing import Dict, Any, Optional, List, Union
import pandas as pd
import logging
import os
from contextlib import contextmanager

from .logger import setup_logger
from .database import get_db_connection, DB_PATH

# Setup logging
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
logger = setup_logger("enhanced_storage", os.path.join(LOG_DIR, "enhanced_storage.log"))

class EnhancedStorage:
    """Enhanced storage with MongoDB-like interface for complex data"""
    
    def __init__(self):
        self.db_path = DB_PATH
        self.initialize_enhanced_tables()
    
    def initialize_enhanced_tables(self):
        """Create enhanced storage tables for complex data"""
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Flexible document storage table (MongoDB-like)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    collection TEXT NOT NULL,
                    document_id TEXT,
                    data TEXT NOT NULL,  -- Full JSON document
                    metadata TEXT,  -- Additional metadata
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(collection, document_id)
                )
            """)
            
            # Pipeline execution detailed logs
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS pipeline_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pipeline_id TEXT NOT NULL,
                    step_name TEXT NOT NULL,
                    step_index INTEGER,
                    status TEXT NOT NULL,
                    start_time DATETIME,
                    end_time DATETIME,
                    duration REAL,
                    input_data TEXT,  -- JSON
                    output_data TEXT,  -- JSON
                    error_details TEXT,  -- JSON
                    metrics TEXT,  -- JSON
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Model training detailed results
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS model_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL,
                    model_type TEXT NOT NULL,
                    training_config TEXT NOT NULL,  -- JSON
                    training_data_info TEXT,  -- JSON
                    model_parameters TEXT,  -- JSON
                    performance_metrics TEXT NOT NULL,  -- JSON
                    validation_results TEXT,  -- JSON
                    feature_importance TEXT,  -- JSON
                    predictions TEXT,  -- JSON array
                    model_file_path TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Complex analysis results
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS analysis_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    analysis_type TEXT NOT NULL,
                    analysis_subtype TEXT,
                    ticker TEXT,
                    parameters TEXT NOT NULL,  -- JSON
                    results TEXT NOT NULL,  -- JSON
                    charts TEXT,  -- JSON array of chart data
                    raw_data TEXT,  -- JSON of raw data used
                    computation_time REAL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Time-series data with flexible schema
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS timeseries_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL,
                    source TEXT NOT NULL,
                    data_type TEXT NOT NULL,  -- 'ohlcv', 'indicators', 'predictions'
                    timestamp_start DATETIME NOT NULL,
                    timestamp_end DATETIME NOT NULL,
                    interval_type TEXT NOT NULL,
                    data_points TEXT NOT NULL,  -- JSON array
                    metadata TEXT,  -- JSON
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes for performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_documents_collection ON documents(collection)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_documents_id ON documents(collection, document_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_pipeline_logs_id ON pipeline_logs(pipeline_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_model_results_ticker ON model_results(ticker, model_type)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_analysis_type ON analysis_results(analysis_type, ticker)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_timeseries_ticker ON timeseries_data(ticker, data_type)")
            
            conn.commit()
            logger.info("Enhanced storage tables initialized")
    
    def store_document(self, collection: str, document_id: str, data: Dict[str, Any], 
                      metadata: Dict[str, Any] = None) -> bool:
        """Store a document in MongoDB-like collection"""
        try:
            with get_db_connection() as conn:
                cursor = conn.cursor()
                
                # Add timestamp to data
                if isinstance(data, dict):
                    data['_stored_at'] = datetime.now().isoformat()
                
                cursor.execute("""
                    INSERT OR REPLACE INTO documents 
                    (collection, document_id, data, metadata, updated_at)
                    VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                """, (collection, document_id, json.dumps(data), 
                     json.dumps(metadata) if metadata else None))
                
                conn.commit()
                logger.info(f"Stored document {document_id} in collection {collection}")
                return True
        except Exception as e:
            logger.error(f"Error storing document: {e}")
            return False
    
    def get_document(self, collection: str, document_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a document from collection"""
        try:
            with get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT data, metadata, created_at, updated_at
                    FROM documents
                    WHERE collection = ? AND document_id = ?
                """, (collection, document_id))
                
                row = cursor.fetchone()
                if row:
                    data = json.loads(row[0])
                    data['_metadata'] = json.loads(row[1]) if row[1] else {}
                    data['_created_at'] = row[2]
                    data['_updated_at'] = row[3]
                    return data
                
                return None
        except Exception as e:
            logger.error(f"Error retrieving document: {e}")
            return None
    
    def find_documents(self, collection: str, filter_dict: Dict[str, Any] = None, 
                      limit: int = None) -> List[Dict[str, Any]]:
        """Find documents in collection (basic filtering)"""
        try:
            with get_db_connection() as conn:
                cursor = conn.cursor()
                
                query = "SELECT document_id, data, metadata, created_at, updated_at FROM documents WHERE collection = ?"
                params = [collection]
                
                if limit:
                    query += f" LIMIT {limit}"
                
                cursor.execute(query, params)
                
                results = []
                for row in cursor.fetchall():
                    data = json.loads(row[1])
                    data['_id'] = row[0]
                    data['_metadata'] = json.loads(row[2]) if row[2] else {}
                    data['_created_at'] = row[3]
                    data['_updated_at'] = row[4]
                    
                    # Basic filtering (can be enhanced)
                    if filter_dict:
                        match = True
                        for key, value in filter_dict.items():
                            if key in data and data[key] != value:
                                match = False
                                break
                        if not match:
                            continue
                    
                    results.append(data)
                
                return results
        except Exception as e:
            logger.error(f"Error finding documents: {e}")
            return []
    
    def store_pipeline_step_log(self, pipeline_id: str, step_name: str, step_index: int,
                               status: str, start_time: datetime = None, end_time: datetime = None,
                               input_data: Dict = None, output_data: Dict = None,
                               error_details: Dict = None, metrics: Dict = None) -> bool:
        """Store detailed pipeline step execution log"""
        try:
            with get_db_connection() as conn:
                cursor = conn.cursor()
                
                duration = None
                if start_time and end_time:
                    duration = (end_time - start_time).total_seconds()
                
                cursor.execute("""
                    INSERT INTO pipeline_logs 
                    (pipeline_id, step_name, step_index, status, start_time, end_time, 
                     duration, input_data, output_data, error_details, metrics)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (pipeline_id, step_name, step_index, status,
                     start_time.isoformat() if start_time else None,
                     end_time.isoformat() if end_time else None,
                     duration,
                     json.dumps(input_data) if input_data else None,
                     json.dumps(output_data) if output_data else None,
                     json.dumps(error_details) if error_details else None,
                     json.dumps(metrics) if metrics else None))
                
                conn.commit()
                logger.info(f"Stored pipeline step log: {pipeline_id} - {step_name}")
                return True
        except Exception as e:
            logger.error(f"Error storing pipeline step log: {e}")
            return False
    
    def store_model_results(self, ticker: str, model_type: str, training_config: Dict,
                           performance_metrics: Dict, **kwargs) -> bool:
        """Store comprehensive model training results"""
        try:
            with get_db_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT OR REPLACE INTO model_results 
                    (ticker, model_type, training_config, training_data_info, model_parameters,
                     performance_metrics, validation_results, feature_importance, predictions, 
                     model_file_path, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """, (ticker, model_type, 
                     json.dumps(training_config),
                     json.dumps(kwargs.get('training_data_info', {})),
                     json.dumps(kwargs.get('model_parameters', {})),
                     json.dumps(performance_metrics),
                     json.dumps(kwargs.get('validation_results', {})),
                     json.dumps(kwargs.get('feature_importance', {})),
                     json.dumps(kwargs.get('predictions', [])),
                     kwargs.get('model_file_path')))
                
                conn.commit()
                logger.info(f"Stored model results: {ticker} - {model_type}")
                return True
        except Exception as e:
            logger.error(f"Error storing model results: {e}")
            return False
    
    def store_analysis_results(self, analysis_type: str, parameters: Dict, results: Dict,
                              ticker: str = None, analysis_subtype: str = None,
                              charts: List = None, raw_data: Dict = None,
                              computation_time: float = None) -> bool:
        """Store complex analysis results"""
        try:
            with get_db_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO analysis_results 
                    (analysis_type, analysis_subtype, ticker, parameters, results, 
                     charts, raw_data, computation_time)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (analysis_type, analysis_subtype, ticker,
                     json.dumps(parameters),
                     json.dumps(results),
                     json.dumps(charts) if charts else None,
                     json.dumps(raw_data) if raw_data else None,
                     computation_time))
                
                conn.commit()
                logger.info(f"Stored analysis results: {analysis_type} - {ticker}")
                return True
        except Exception as e:
            logger.error(f"Error storing analysis results: {e}")
            return False
    
    def store_timeseries_data(self, ticker: str, source: str, data_type: str,
                             data_points: List[Dict], interval_type: str,
                             timestamp_start: datetime, timestamp_end: datetime,
                             metadata: Dict = None) -> bool:
        """Store time-series data with flexible schema"""
        try:
            with get_db_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT OR REPLACE INTO timeseries_data 
                    (ticker, source, data_type, timestamp_start, timestamp_end,
                     interval_type, data_points, metadata, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """, (ticker, source, data_type,
                     timestamp_start.isoformat(),
                     timestamp_end.isoformat(),
                     interval_type,
                     json.dumps(data_points),
                     json.dumps(metadata) if metadata else None))
                
                conn.commit()
                logger.info(f"Stored timeseries data: {ticker} - {data_type}")
                return True
        except Exception as e:
            logger.error(f"Error storing timeseries data: {e}")
            return False
    
    def get_pipeline_logs(self, pipeline_id: str) -> List[Dict]:
        """Get detailed pipeline execution logs"""
        try:
            with get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM pipeline_logs
                    WHERE pipeline_id = ?
                    ORDER BY step_index, created_at
                """, (pipeline_id,))
                
                columns = [desc[0] for desc in cursor.description]
                results = []
                for row in cursor.fetchall():
                    result = dict(zip(columns, row))
                    
                    # Parse JSON fields
                    for field in ['input_data', 'output_data', 'error_details', 'metrics']:
                        if result.get(field):
                            result[field] = json.loads(result[field])
                    
                    results.append(result)
                
                return results
        except Exception as e:
            logger.error(f"Error retrieving pipeline logs: {e}")
            return []
    
    def get_model_results(self, ticker: str = None, model_type: str = None) -> List[Dict]:
        """Get comprehensive model results"""
        try:
            with get_db_connection() as conn:
                cursor = conn.cursor()
                
                query = "SELECT * FROM model_results WHERE 1=1"
                params = []
                
                if ticker:
                    query += " AND ticker = ?"
                    params.append(ticker)
                
                if model_type:
                    query += " AND model_type = ?"
                    params.append(model_type)
                
                query += " ORDER BY created_at DESC"
                cursor.execute(query, params)
                
                columns = [desc[0] for desc in cursor.description]
                results = []
                for row in cursor.fetchall():
                    result = dict(zip(columns, row))
                    
                    # Parse JSON fields
                    json_fields = ['training_config', 'training_data_info', 'model_parameters',
                                  'performance_metrics', 'validation_results', 'feature_importance', 'predictions']
                    for field in json_fields:
                        if result.get(field):
                            result[field] = json.loads(result[field])
                    
                    results.append(result)
                
                return results
        except Exception as e:
            logger.error(f"Error retrieving model results: {e}")
            return []
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get comprehensive database statistics"""
        try:
            with get_db_connection() as conn:
                cursor = conn.cursor()
                
                stats = {}
                
                # Table sizes
                tables = ['ohlcv_data', 'documents', 'pipeline_logs', 'model_results', 
                         'analysis_results', 'timeseries_data', 'eda_results', 
                         'correlation_results', 'trading_signals']
                
                for table in tables:
                    try:
                        cursor.execute(f"SELECT COUNT(*) FROM {table}")
                        count = cursor.fetchone()[0]
                        stats[f"{table}_count"] = count
                    except:
                        stats[f"{table}_count"] = 0
                
                # Database size
                stats['database_size_mb'] = os.path.getsize(self.db_path) / (1024 * 1024)
                
                # Recent activity
                cursor.execute("""
                    SELECT COUNT(*) FROM pipeline_logs 
                    WHERE created_at > datetime('now', '-24 hours')
                """)
                stats['recent_pipeline_activity'] = cursor.fetchone()[0]
                
                return stats
        except Exception as e:
            logger.error(f"Error getting database stats: {e}")
            return {}

# Initialize enhanced storage instance
enhanced_storage = EnhancedStorage()
