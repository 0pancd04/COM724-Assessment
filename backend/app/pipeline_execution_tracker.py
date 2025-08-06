"""
Pipeline Execution Tracker - Comprehensive tracking system for pipeline executions
"""

import sqlite3
import uuid
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path

from .logger import setup_enhanced_logger
from .database import crypto_db

# Setup logger
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
logger = setup_enhanced_logger("pipeline_tracker", os.path.join(LOG_DIR, "pipeline_tracker.log"))

@dataclass
class PipelineExecution:
    """Pipeline execution metadata"""
    trace_id: str
    pipeline_type: str  # 'full', 'download_only', 'train_only', etc.
    status: str  # 'running', 'completed', 'failed', 'cancelled'
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    
    # Configuration
    tickers: List[str] = None
    sources: List[str] = None
    period: str = None
    interval: str = None
    max_days: int = None
    feature: str = None
    test_size: float = None
    include_eda: bool = None
    include_clustering: bool = None
    
    # Results summary
    total_steps: int = 0
    completed_steps: int = 0
    failed_steps: int = 0
    success_rate: float = 0.0
    
    # Error information
    error_message: Optional[str] = None
    error_step: Optional[str] = None
    
    # Metadata
    user_agent: Optional[str] = None
    ip_address: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        result = asdict(self)
        # Convert datetime objects to ISO strings
        if self.start_time:
            result['start_time'] = self.start_time.isoformat()
        if self.end_time:
            result['end_time'] = self.end_time.isoformat()
        return result

@dataclass
class PipelineStepExecution:
    """Individual pipeline step execution tracking"""
    trace_id: str
    step_name: str
    step_type: str
    step_order: int
    status: str  # 'running', 'completed', 'failed', 'skipped'
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    
    # Step-specific data
    input_data: Optional[Dict] = None
    output_data: Optional[Dict] = None
    error_message: Optional[str] = None
    
    # Metrics
    records_processed: int = 0
    records_created: int = 0
    files_created: List[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        result = asdict(self)
        if self.start_time:
            result['start_time'] = self.start_time.isoformat()
        if self.end_time:
            result['end_time'] = self.end_time.isoformat()
        return result

class PipelineExecutionTracker:
    """Main pipeline execution tracking system"""
    
    def __init__(self, db_path: str = None):
        self.db_path = db_path or os.path.join(BASE_DIR, "data", "pipeline_executions.db")
        self.ensure_database()
        logger.info(f"Pipeline execution tracker initialized with database: {self.db_path}", "PipelineExecutionTracker")
    
    def ensure_database(self):
        """Create database tables if they don't exist"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS pipeline_executions (
                    trace_id TEXT PRIMARY KEY,
                    pipeline_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    start_time TEXT NOT NULL,
                    end_time TEXT,
                    duration_seconds REAL,
                    
                    -- Configuration
                    tickers TEXT,  -- JSON array
                    sources TEXT,  -- JSON array
                    period TEXT,
                    interval TEXT,
                    max_days INTEGER,
                    feature TEXT,
                    test_size REAL,
                    include_eda BOOLEAN,
                    include_clustering BOOLEAN,
                    
                    -- Results
                    total_steps INTEGER DEFAULT 0,
                    completed_steps INTEGER DEFAULT 0,
                    failed_steps INTEGER DEFAULT 0,
                    success_rate REAL DEFAULT 0.0,
                    
                    -- Error info
                    error_message TEXT,
                    error_step TEXT,
                    
                    -- Metadata
                    user_agent TEXT,
                    ip_address TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS pipeline_step_executions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trace_id TEXT NOT NULL,
                    step_name TEXT NOT NULL,
                    step_type TEXT NOT NULL,
                    step_order INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    start_time TEXT NOT NULL,
                    end_time TEXT,
                    duration_seconds REAL,
                    
                    -- Step data
                    input_data TEXT,  -- JSON
                    output_data TEXT,  -- JSON
                    error_message TEXT,
                    
                    -- Metrics
                    records_processed INTEGER DEFAULT 0,
                    records_created INTEGER DEFAULT 0,
                    files_created TEXT,  -- JSON array
                    
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    
                    FOREIGN KEY (trace_id) REFERENCES pipeline_executions (trace_id)
                )
            """)
            
            # Create indexes for better performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_trace_id ON pipeline_step_executions (trace_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_status ON pipeline_executions (status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_start_time ON pipeline_executions (start_time)")
            
            conn.commit()
            logger.info("Pipeline execution database tables created/verified", "ensure_database")
    
    def generate_trace_id(self) -> str:
        """Generate a unique trace ID for pipeline execution"""
        trace_id = f"pipeline_{int(datetime.now().timestamp())}_{uuid.uuid4().hex[:8]}"
        logger.info(f"Generated new trace ID: {trace_id}", "generate_trace_id")
        return trace_id
    
    def start_pipeline_execution(self, 
                                pipeline_type: str,
                                tickers: List[str] = None,
                                sources: List[str] = None,
                                period: str = None,
                                interval: str = None,
                                max_days: int = None,
                                feature: str = None,
                                test_size: float = None,
                                include_eda: bool = None,
                                include_clustering: bool = None,
                                user_agent: str = None,
                                ip_address: str = None) -> str:
        """Start tracking a new pipeline execution"""
        
        trace_id = self.generate_trace_id()
        execution = PipelineExecution(
            trace_id=trace_id,
            pipeline_type=pipeline_type,
            status="running",
            start_time=datetime.now(),
            tickers=tickers or [],
            sources=sources or [],
            period=period,
            interval=interval,
            max_days=max_days,
            feature=feature,
            test_size=test_size,
            include_eda=include_eda,
            include_clustering=include_clustering,
            user_agent=user_agent,
            ip_address=ip_address
        )
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO pipeline_executions (
                    trace_id, pipeline_type, status, start_time,
                    tickers, sources, period, interval, max_days,
                    feature, test_size, include_eda, include_clustering,
                    user_agent, ip_address
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                execution.trace_id, execution.pipeline_type, execution.status,
                execution.start_time.isoformat(),
                json.dumps(execution.tickers), json.dumps(execution.sources),
                execution.period, execution.interval, execution.max_days,
                execution.feature, execution.test_size, execution.include_eda,
                execution.include_clustering, execution.user_agent, execution.ip_address
            ))
            conn.commit()
        
        logger.info(f"Started pipeline execution tracking: {trace_id} ({pipeline_type})", "start_pipeline_execution")
        return trace_id
    
    def start_step_execution(self,
                           trace_id: str,
                           step_name: str,
                           step_type: str,
                           step_order: int,
                           input_data: Dict = None) -> int:
        """Start tracking a pipeline step execution"""
        
        step_execution = PipelineStepExecution(
            trace_id=trace_id,
            step_name=step_name,
            step_type=step_type,
            step_order=step_order,
            status="running",
            start_time=datetime.now(),
            input_data=input_data or {},
            files_created=[]
        )
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                INSERT INTO pipeline_step_executions (
                    trace_id, step_name, step_type, step_order, status, start_time, input_data
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                step_execution.trace_id, step_execution.step_name, step_execution.step_type,
                step_execution.step_order, step_execution.status, step_execution.start_time.isoformat(),
                json.dumps(step_execution.input_data)
            ))
            step_id = cursor.lastrowid
            conn.commit()
        
        logger.info(f"Started step execution: {step_name} (ID: {step_id}) for trace {trace_id}", "start_step_execution")
        return step_id
    
    def complete_step_execution(self,
                              step_id: int,
                              status: str = "completed",
                              output_data: Dict = None,
                              error_message: str = None,
                              records_processed: int = 0,
                              records_created: int = 0,
                              files_created: List[str] = None):
        """Complete a pipeline step execution"""
        
        end_time = datetime.now()
        
        with sqlite3.connect(self.db_path) as conn:
            # Get start time to calculate duration
            cursor = conn.execute("SELECT start_time FROM pipeline_step_executions WHERE id = ?", (step_id,))
            row = cursor.fetchone()
            if row:
                start_time = datetime.fromisoformat(row[0])
                duration_seconds = (end_time - start_time).total_seconds()
            else:
                duration_seconds = 0
            
            conn.execute("""
                UPDATE pipeline_step_executions 
                SET status = ?, end_time = ?, duration_seconds = ?, output_data = ?,
                    error_message = ?, records_processed = ?, records_created = ?,
                    files_created = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (
                status, end_time.isoformat(), duration_seconds, json.dumps(output_data or {}),
                error_message, records_processed, records_created,
                json.dumps(files_created or []), step_id
            ))
            conn.commit()
        
        logger.info(f"Completed step execution ID {step_id} with status: {status}", "complete_step_execution")
    
    def complete_pipeline_execution(self,
                                  trace_id: str,
                                  status: str = "completed",
                                  error_message: str = None,
                                  error_step: str = None):
        """Complete a pipeline execution"""
        
        end_time = datetime.now()
        
        with sqlite3.connect(self.db_path) as conn:
            # Get start time and calculate step statistics
            cursor = conn.execute("""
                SELECT start_time FROM pipeline_executions WHERE trace_id = ?
            """, (trace_id,))
            row = cursor.fetchone()
            if row:
                start_time = datetime.fromisoformat(row[0])
                duration_seconds = (end_time - start_time).total_seconds()
            else:
                duration_seconds = 0
            
            # Calculate step statistics
            cursor = conn.execute("""
                SELECT 
                    COUNT(*) as total_steps,
                    SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed_steps,
                    SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed_steps
                FROM pipeline_step_executions 
                WHERE trace_id = ?
            """, (trace_id,))
            stats = cursor.fetchone()
            
            total_steps = stats[0] if stats[0] else 0
            completed_steps = stats[1] if stats[1] else 0
            failed_steps = stats[2] if stats[2] else 0
            success_rate = (completed_steps / total_steps) if total_steps > 0 else 0.0
            
            # Update pipeline execution
            conn.execute("""
                UPDATE pipeline_executions 
                SET status = ?, end_time = ?, duration_seconds = ?,
                    total_steps = ?, completed_steps = ?, failed_steps = ?, success_rate = ?,
                    error_message = ?, error_step = ?, updated_at = CURRENT_TIMESTAMP
                WHERE trace_id = ?
            """, (
                status, end_time.isoformat(), duration_seconds,
                total_steps, completed_steps, failed_steps, success_rate,
                error_message, error_step, trace_id
            ))
            conn.commit()
        
        logger.info(f"Completed pipeline execution {trace_id} with status: {status} ({completed_steps}/{total_steps} steps)", "complete_pipeline_execution")
    
    def get_pipeline_execution(self, trace_id: str) -> Optional[Dict]:
        """Get pipeline execution details"""
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            # Get pipeline execution
            cursor = conn.execute("SELECT * FROM pipeline_executions WHERE trace_id = ?", (trace_id,))
            pipeline_row = cursor.fetchone()
            
            if not pipeline_row:
                return None
            
            # Get step executions
            cursor = conn.execute("""
                SELECT * FROM pipeline_step_executions 
                WHERE trace_id = ? 
                ORDER BY step_order, start_time
            """, (trace_id,))
            step_rows = cursor.fetchall()
            
            # Convert to dictionaries and parse JSON fields
            pipeline_data = dict(pipeline_row)
            if pipeline_data['tickers']:
                pipeline_data['tickers'] = json.loads(pipeline_data['tickers'])
            if pipeline_data['sources']:
                pipeline_data['sources'] = json.loads(pipeline_data['sources'])
            
            steps_data = []
            for step_row in step_rows:
                step_data = dict(step_row)
                if step_data['input_data']:
                    step_data['input_data'] = json.loads(step_data['input_data'])
                if step_data['output_data']:
                    step_data['output_data'] = json.loads(step_data['output_data'])
                if step_data['files_created']:
                    step_data['files_created'] = json.loads(step_data['files_created'])
                steps_data.append(step_data)
            
            return {
                'pipeline': pipeline_data,
                'steps': steps_data
            }
    
    def get_pipeline_executions(self, 
                               limit: int = 50, 
                               status: str = None,
                               pipeline_type: str = None) -> List[Dict]:
        """Get list of pipeline executions"""
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            query = "SELECT * FROM pipeline_executions"
            params = []
            conditions = []
            
            if status:
                conditions.append("status = ?")
                params.append(status)
            
            if pipeline_type:
                conditions.append("pipeline_type = ?")
                params.append(pipeline_type)
            
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
            
            query += " ORDER BY start_time DESC LIMIT ?"
            params.append(limit)
            
            cursor = conn.execute(query, params)
            rows = cursor.fetchall()
            
            executions = []
            for row in rows:
                execution_data = dict(row)
                if execution_data['tickers']:
                    execution_data['tickers'] = json.loads(execution_data['tickers'])
                if execution_data['sources']:
                    execution_data['sources'] = json.loads(execution_data['sources'])
                executions.append(execution_data)
            
            return executions
    
    def get_execution_statistics(self) -> Dict:
        """Get overall execution statistics"""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT 
                    COUNT(*) as total_executions,
                    SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed_executions,
                    SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed_executions,
                    SUM(CASE WHEN status = 'running' THEN 1 ELSE 0 END) as running_executions,
                    AVG(duration_seconds) as avg_duration,
                    AVG(success_rate) as avg_success_rate
                FROM pipeline_executions
            """)
            stats = cursor.fetchone()
            
            return {
                'total_executions': stats[0] or 0,
                'completed_executions': stats[1] or 0,
                'failed_executions': stats[2] or 0,
                'running_executions': stats[3] or 0,
                'avg_duration_seconds': stats[4] or 0,
                'avg_success_rate': stats[5] or 0
            }

# Global instance
pipeline_tracker = PipelineExecutionTracker()
