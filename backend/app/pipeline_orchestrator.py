"""
Pipeline Orchestrator - Complete Cryptocurrency Analysis Pipeline
Executes the entire workflow: Download → Preprocess → EDA → Model Training → Evaluation
"""

import asyncio
import os
import json
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import logging

from .logger import setup_enhanced_logger, log_function_entry_exit
from .database import crypto_db
from .unified_data_handler import unified_handler
from .data_preprocessing import preprocess_data
from .eda_analysis import perform_eda_analysis
from .grouping_analysis import perform_dimensionality_reduction, perform_clustering_analysis
from .correlation_analysis import perform_correlation_analysis
from .model_comparison import model_comparison
from .websocket_manager import ws_manager
from .analysis_storage import analysis_storage
from .enhanced_storage import enhanced_storage
from .pipeline_execution_tracker import pipeline_tracker

# Setup enhanced logging
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
logger = setup_enhanced_logger("pipeline_orchestrator", os.path.join(LOG_DIR, "pipeline_orchestrator.log"))

class PipelineStep:
    """Base class for pipeline steps"""
    
    def __init__(self, name: str):
        self.name = name
        self.status = "pending"
        self.start_time = None
        self.end_time = None
        self.result = None
        self.error = None
    
    def start(self):
        self.status = "running"
        self.start_time = datetime.now()
        logger.info(f"Starting step: {self.name}", "PipelineStep")
    
    def complete(self, result: Any = None):
        self.status = "completed"
        self.end_time = datetime.now()
        self.result = result
        duration = (self.end_time - self.start_time).total_seconds()
        logger.info(f"Completed step: {self.name} in {duration:.2f}s", "PipelineStep")
    
    def fail(self, error: str):
        self.status = "failed"
        self.end_time = datetime.now()
        self.error = error
        logger.error(f"Failed step: {self.name} - {error}", "PipelineStep")

class DataDownloadStep(PipelineStep):
    """Step for downloading cryptocurrency data"""
    
    def __init__(self, tickers: List[str], sources: List[str], period: str, interval: str):
        super().__init__("Data Download")
        self.tickers = tickers
        self.sources = sources
        self.period = period
        self.interval = interval
    
    async def execute(self) -> Dict:
        self.start()
        try:
            results = {}
            
            for source in self.sources:
                source_results = {}
                
                if len(self.tickers) == 1 and self.tickers[0] == "TOP30":
                    # Download top 30
                    data = unified_handler.download_top_30_cryptos(source, self.period, self.interval)
                    source_results = {
                        'count': len(data),
                        'tickers': list(data.keys()),
                        'method': 'top30'
                    }
                else:
                    # Download specific tickers
                    for ticker in self.tickers:
                        try:
                            df = unified_handler.download_and_store_data(
                                ticker, source, self.period, self.interval, update_missing=True
                            )
                            if not df.empty:
                                source_results[ticker] = {
                                    'records': len(df),
                                    'first_date': str(df.index.min()),
                                    'last_date': str(df.index.max())
                                }
                            else:
                                logger.warning(f"Empty data received for {ticker} from {source}")
                                source_results[ticker] = {
                                    'records': 0,
                                    'warning': 'No data available for this ticker and time period'
                                }
                        except Exception as e:
                            logger.warning(f"Failed to download {ticker} from {source}: {e}")
                            source_results[ticker] = {'error': str(e)}
                
                results[source] = source_results
            
            self.complete(results)
            return results
            
        except Exception as e:
            self.fail(str(e))
            raise

class DataPreprocessingStep(PipelineStep):
    """Step for data preprocessing"""
    
    def __init__(self, source: str, max_days: int):
        super().__init__(f"Data Preprocessing ({source})")
        self.source = source
        self.max_days = max_days
    
    async def execute(self) -> Dict:
        self.start()
        try:
            # Prepare data from database for preprocessing
            logger.info(f"Preparing data for preprocessing from source: {self.source}")
            
            # First check what tickers are available in the database
            available_tickers = unified_handler.db.get_all_tickers(self.source)
            logger.info(f"Found {len(available_tickers)} tickers for {self.source}: {available_tickers[:10]}")
            
            combined_data = unified_handler.prepare_data_for_preprocessing(self.source)
            logger.info(f"Combined data shape: {combined_data.shape}")
            
            if combined_data.empty:
                logger.warning(f"No data available for preprocessing from {self.source}")
                logger.info(f"Debug: Available tickers in DB for {self.source}: {available_tickers}")
                self.complete({'skipped': True, 'reason': f'No data available from {self.source}', 'available_tickers': len(available_tickers)})
                return {'skipped': True, 'source': self.source, 'available_tickers': len(available_tickers)}
            
            # Save to temporary file for preprocessing
            temp_file = f"data/temp_{self.source}_data.csv"
            os.makedirs(os.path.dirname(temp_file), exist_ok=True)
            combined_data.to_csv(temp_file)
            
            # Preprocess the data
            output_file = f"data/preprocessed_{self.source}_{self.max_days}d.csv"
            df_scaled, report = preprocess_data(temp_file, self.max_days, output_file)
            
            # Clean up temp file
            if os.path.exists(temp_file):
                os.remove(temp_file)
            
            result = {
                'output_file': output_file,
                'report': report,
                'shape': df_scaled.shape
            }
            
            self.complete(result)
            return result
            
        except Exception as e:
            self.fail(str(e))
            raise

class EDAStep(PipelineStep):
    """Step for Exploratory Data Analysis"""
    
    def __init__(self, ticker: str, preprocessed_file: str = None, use_database: bool = True):
        super().__init__(f"EDA Analysis ({ticker})")
        self.ticker = ticker
        self.preprocessed_file = preprocessed_file
        self.use_database = use_database
    
    async def execute(self) -> Dict:
        self.start()
        try:
            # Use database-based EDA
            report, charts = perform_eda_analysis(
                ticker=self.ticker,
                preprocessed_file=self.preprocessed_file,
                use_database=self.use_database,
                source=getattr(self, 'source', 'yfinance')
            )
            
            # Results are already stored in database by perform_eda_analysis
            logger.info(f"EDA results for {self.ticker} stored in database")
            
            result = {
                'ticker': self.ticker,
                'report': report,
                'charts': charts
            }
            
            self.complete(result)
            return result
            
        except Exception as e:
            self.fail(str(e))
            raise

class DimensionalityReductionStep(PipelineStep):
    """Step for dimensionality reduction"""
    
    def __init__(self, input_file: str = None, max_days: int = 90, source: str = 'yfinance', use_database: bool = True):
        super().__init__("Dimensionality Reduction")
        self.input_file = input_file
        self.max_days = max_days
        self.source = source
        self.use_database = use_database
    
    async def execute(self) -> Dict:
        self.start()
        try:
            output_file = "data/dim_reduced_best.csv"
            chart_file = "data/dim_reduction_chart.json"
            
            reduced_df, report, best_algo, fig = perform_dimensionality_reduction(
                input_file=self.input_file,
                output_file=output_file,
                chart_file=chart_file,
                use_database=self.use_database,
                source=self.source
            )
            
            # Results are already stored in database by perform_dimensionality_reduction
            logger.info("Dimensionality reduction results stored in database")
            
            result = {
                'output_file': output_file,
                'chart_file': chart_file,
                'best_algorithm': best_algo,
                'report': report
            }
            
            self.complete(result)
            return result
            
        except Exception as e:
            self.fail(str(e))
            raise

class CorrelationAnalysisStep(PipelineStep):
    """Step for correlation analysis"""
    
    def __init__(self, tickers: list = None, feature: str = "Close", source: str = 'yfinance', use_database: bool = True):
        super().__init__("Correlation Analysis")
        self.tickers = tickers or ["BTC", "ETH", "ADA", "DOT"]
        self.feature = feature
        self.source = source
        self.use_database = use_database
    
    async def execute(self) -> Dict:
        self.start()
        try:
            output_file = "data/correlation_matrix.csv"
            chart_file = "data/correlation_chart.json"
            
            corr_df, report, fig = perform_correlation_analysis(
                preprocessed_file=None,
                selected_tickers=self.tickers,
                feature=self.feature,
                output_file=output_file,
                chart_file=chart_file,
                use_database=self.use_database,
                source=self.source
            )
            
            logger.info("Correlation analysis results stored in database")
            
            result = {
                'output_file': output_file,
                'chart_file': chart_file,
                'correlation_matrix': corr_df.to_dict(),
                'report': report,
                'tickers': self.tickers,
                'feature': self.feature
            }
            
            self.complete(result)
            return result
            
        except Exception as e:
            self.fail(str(e))
            raise

class ConditionalDimensionalityReductionStep(PipelineStep):
    """Conditional step for dimensionality reduction that checks if input exists"""
    
    def __init__(self, input_file: str = None, max_days: int = 90, source: str = 'yfinance', use_database: bool = True):
        super().__init__("Dimensionality Reduction")
        self.input_file = input_file
        self.max_days = max_days
        self.source = source
        self.use_database = use_database
    
    async def execute(self) -> Dict:
        self.start()
        try:
            # If using database, directly run dimensionality reduction
            if self.use_database:
                output_file = "data/dim_reduced_best.csv"
                chart_file = "data/dim_reduction_chart.json"
                
                reduced_df, report, best_algo, fig = perform_dimensionality_reduction(
                    input_file=None,
                    output_file=output_file,
                    chart_file=chart_file,
                    use_database=True,
                    source=self.source
                )
            else:
                # Check if input file exists for file-based mode
                if not os.path.exists(self.input_file):
                    logger.warning(f"Skipping dimensionality reduction: {self.input_file} not found")
                    self.complete({'skipped': True, 'reason': f'Input file {self.input_file} not found'})
                    return {'skipped': True}
                
                output_file = "data/dim_reduced_best.csv"
                chart_file = "data/dim_reduction_chart.json"
                
                reduced_df, report, best_algo, fig = perform_dimensionality_reduction(
                    self.input_file, output_file, chart_file, use_database=False
                )
            
            result = {
                'output_file': output_file,
                'chart_file': chart_file,
                'best_algorithm': best_algo,
                'report': report,
                'source': self.source
            }
            
            self.complete(result)
            return result
            
        except Exception as e:
            self.fail(str(e))
            # Don't raise if optional step fails
            return {'failed': True, 'error': str(e)}

class ClusteringStep(PipelineStep):
    """Step for clustering analysis"""
    
    def __init__(self, reduced_file: str = None, use_database: bool = True, source: str = 'yfinance'):
        super().__init__("Clustering Analysis")
        self.reduced_file = reduced_file
        self.use_database = use_database
        self.source = source
    
    async def execute(self) -> Dict:
        self.start()
        try:
            # If using database, directly run clustering
            if self.use_database:
                output_file = "data/clustering_result.csv"
                chart_file = "data/clustering_chart.json"
                
                cluster_df, report, fig = perform_clustering_analysis(
                    reduced_data_file=None,
                    output_file=output_file,
                    chart_file=chart_file,
                    use_database=True,
                    source=self.source
                )
            else:
                # Check if input file exists for file-based mode
                if not os.path.exists(self.reduced_file):
                    logger.warning(f"Skipping clustering: {self.reduced_file} not found")
                    self.complete({'skipped': True, 'reason': f'Input file {self.reduced_file} not found'})
                    return {'skipped': True}
                
                output_file = "data/clustering_result.csv"
                chart_file = "data/clustering_chart.json"
                
                cluster_df, report, fig = perform_clustering_analysis(
                    self.reduced_file, output_file, chart_file, use_database=False
                )
            
            result = {
                'output_file': output_file,
                'chart_file': chart_file,
                'report': report
            }
            
            self.complete(result)
            return result
            
        except Exception as e:
            self.fail(str(e))
            # Don't raise if optional step fails
            return {'failed': True, 'error': str(e)}

class ModelTrainingStep(PipelineStep):
    """Step for model training and comparison"""
    
    def __init__(self, tickers: List[str], feature: str, test_size: float):
        super().__init__("Model Training & Comparison")
        self.tickers = tickers
        self.feature = feature
        self.test_size = test_size
    
    async def execute(self) -> Dict:
        self.start()
        try:
            results = {}
            
            for ticker in self.tickers:
                try:
                    # Train all models for this ticker
                    model_results = model_comparison.train_all_models(
                        ticker, self.feature, self.test_size
                    )
                    results[ticker] = model_results
                    
                except Exception as e:
                    logger.warning(f"Failed to train models for {ticker}: {e}")
                    results[ticker] = {'error': str(e)}
            
            self.complete(results)
            return results
            
        except Exception as e:
            self.fail(str(e))
            raise

class PipelineOrchestrator:
    """Main orchestrator for the complete cryptocurrency analysis pipeline"""
    
    def __init__(self, trace_id: str = None):
        self.steps = []
        self.results = {}
        self.start_time = None
        self.end_time = None
        self.trace_id = trace_id  # Add trace ID for tracking
        logger.info(f"Pipeline orchestrator initialized with trace_id: {trace_id}", "PipelineOrchestrator")
    
    def add_step(self, step: PipelineStep):
        """Add a step to the pipeline"""
        self.steps.append(step)
    
    async def execute_pipeline(self) -> Dict:
        """Execute the complete pipeline with WebSocket updates"""
        logger.info(f"Starting pipeline execution with trace_id: {self.trace_id}", "PipelineOrchestrator")
        
        self.start_time = datetime.now()
        pipeline_id = self.trace_id or f"pipeline_{int(self.start_time.timestamp())}"
        
        logger.info(f"Using pipeline_id: {pipeline_id}", "PipelineOrchestrator")
        logger.info(f"Pipeline has {len(self.steps)} steps defined", "PipelineOrchestrator")
        
        # Check if there are any steps to execute
        if not self.steps:
            logger.error("No pipeline steps defined - this could cause division by zero!", "PipelineOrchestrator")
            return {
                'error': 'No pipeline steps defined. Please check pipeline configuration.',
                'pipeline_id': pipeline_id,
                'status': 'error'
            }
        
        # Check if another pipeline is running
        if ws_manager.is_pipeline_running():
            return {
                'error': 'Another pipeline is currently running. Please wait for it to complete.',
                'pipeline_id': pipeline_id,
                'status': 'rejected'
            }
        
        # Start pipeline with WebSocket notification
        logger.info(f"About to start WebSocket pipeline with {len(self.steps)} steps", "PipelineOrchestrator")
        await ws_manager.start_pipeline(pipeline_id, len(self.steps))
        
        logger.info(f"Starting complete cryptocurrency analysis pipeline with {len(self.steps)} steps", "PipelineOrchestrator")
        
        pipeline_results = {
            'pipeline_id': pipeline_id,
            'start_time': self.start_time.isoformat(),
            'steps': {},
            'summary': {
                'total_steps': len(self.steps),
                'completed': 0,
                'failed': 0
            }
        }
        
        try:
            logger.info(f"About to iterate through {len(self.steps)} steps", "PipelineOrchestrator")
            
            if not self.steps:
                logger.error("CRITICAL: Steps array is empty during execution!", "PipelineOrchestrator")
                raise ValueError("No pipeline steps to execute")
            
            for i, step in enumerate(self.steps, 1):
                logger.info(f"Processing step {i}/{len(self.steps)}: {step.name}", "PipelineOrchestrator")
                
                # Start step tracking
                step_input_data = {
                    'step_type': type(step).__name__,
                    'step_config': {k: v for k, v in step.__dict__.items() 
                                   if not k.startswith('_') and k not in ['start_time', 'end_time', 'result', 'error']}
                }
                
                step_id = None
                if self.trace_id:
                    step_id = pipeline_tracker.start_step_execution(
                        trace_id=self.trace_id,
                        step_name=step.name,
                        step_type=type(step).__name__,
                        step_order=i,
                        input_data=step_input_data
                    )
                
                try:
                    # Update WebSocket with current step
                    await ws_manager.update_step(step.name, i, f"Starting {step.name}")
                    
                    logger.info(f"[PipelineOrchestrator.execute_pipeline] Executing step {i}/{len(self.steps)}: {step.name} ({type(step).__name__})")
                    result = await step.execute()
                    logger.info(f"[PipelineOrchestrator.execute_pipeline] Step {step.name} completed with status: {step.status}")
                    
                    # Mark step as complete in WebSocket
                    await ws_manager.complete_step(step.name, result)
                    
                    # Store detailed step log in enhanced storage
                    enhanced_storage.store_pipeline_step_log(
                        pipeline_id=pipeline_id,
                        step_name=step.name,
                        step_index=i,
                        status=step.status,
                        start_time=step.start_time,
                        end_time=step.end_time,
                        input_data=step_input_data,
                        output_data=result if isinstance(result, dict) else {'result': str(result)},
                        metrics={
                            'duration': (step.end_time - step.start_time).total_seconds() if step.end_time and step.start_time else None,
                            'success': step.status == 'completed'
                        }
                    )
                    
                    pipeline_results['steps'][step.name] = {
                        'status': step.status,
                        'duration': (step.end_time - step.start_time).total_seconds() if step.end_time else None,
                        'result': result
                    }
                    pipeline_results['summary']['completed'] += 1
                    
                    # Complete step tracking
                    if step_id and self.trace_id:
                        pipeline_tracker.complete_step_execution(
                            step_id=step_id,
                            status="completed",
                            output_data=result if isinstance(result, dict) else {"result": str(result)},
                            records_processed=getattr(step, 'records_processed', 0),
                            records_created=getattr(step, 'records_created', 0),
                            files_created=getattr(step, 'files_created', [])
                        )
                    
                except Exception as e:
                    pipeline_results['steps'][step.name] = {
                        'status': step.status,
                        'error': step.error or str(e),
                        'duration': (step.end_time - step.start_time).total_seconds() if step.end_time else None
                    }
                    pipeline_results['summary']['failed'] += 1
                    
                    # Complete step tracking with failure
                    if step_id and self.trace_id:
                        pipeline_tracker.complete_step_execution(
                            step_id=step_id,
                            status="failed",
                            error_message=str(e)
                        )
                    
                    # Update WebSocket with error
                    await ws_manager.update_step(step.name, i, f"Failed: {str(e)}")
                    
                    # Continue with other steps even if one fails
                    logger.warning(f"Step {step.name} failed, continuing with remaining steps")
            
            self.end_time = datetime.now()
            pipeline_results['end_time'] = self.end_time.isoformat()
            pipeline_results['total_duration'] = (self.end_time - self.start_time).total_seconds()
            
            # Store pipeline results in both databases
            tickers = []
            sources = []
            for step in self.steps:
                if hasattr(step, 'tickers'):
                    tickers.extend(step.tickers if isinstance(step.tickers, list) else [step.tickers])
                if hasattr(step, 'sources'):
                    sources.extend(step.sources if isinstance(step.sources, list) else [step.sources])
            
            # Store in traditional analysis storage
            analysis_storage.store_pipeline_results(
                pipeline_id=pipeline_id,
                tickers=list(set(tickers)) if tickers else ['N/A'],
                sources=list(set(sources)) if sources else ['N/A'],
                start_time=self.start_time,
                end_time=self.end_time,
                duration=pipeline_results['total_duration'],
                steps_completed=pipeline_results['summary']['completed'],
                steps_failed=pipeline_results['summary']['failed'],
                results_summary=pipeline_results['summary']
            )
            
            # Store detailed results in enhanced storage
            enhanced_storage.store_document(
                collection="pipeline_executions",
                document_id=pipeline_id,
                data={
                    "pipeline_id": pipeline_id,
                    "tickers": list(set(tickers)) if tickers else ['N/A'],
                    "sources": list(set(sources)) if sources else ['N/A'],
                    "execution_details": pipeline_results,
                    "step_count": len(self.steps),
                    "success_rate": (pipeline_results['summary']['completed'] / len(self.steps)) if len(self.steps) > 0 else 0.0
                },
                metadata={
                    "execution_type": "full_pipeline",
                    "version": "1.0"
                }
            )
            
            # End pipeline in WebSocket
            success = pipeline_results['summary']['failed'] == 0
            total_steps = len(self.steps) if self.steps else 1  # Avoid division by zero
            completed_steps = pipeline_results['summary']['completed']
            
            logger.info(f"Pipeline completion stats: completed={completed_steps}, total={total_steps}, steps_array_length={len(self.steps)}", "PipelineOrchestrator")
            logger.info(f"Pipeline results summary: {pipeline_results['summary']}", "PipelineOrchestrator")
            
            # Extra safety checks for division by zero
            if total_steps > 0 and isinstance(completed_steps, (int, float)):
                try:
                    success_rate = completed_steps / total_steps
                    logger.info(f"Calculated success rate: {success_rate} ({completed_steps}/{total_steps})", "PipelineOrchestrator")
                except ZeroDivisionError as zde:
                    logger.error(f"Division by zero in success rate calculation: completed={completed_steps}, total={total_steps}", "PipelineOrchestrator")
                    success_rate = 0.0
                except Exception as calc_error:
                    logger.error(f"Error calculating success rate: {calc_error}, completed={completed_steps}, total={total_steps}", "PipelineOrchestrator")
                    success_rate = 0.0
            else:
                logger.warning(f"Invalid values for success rate calculation: total_steps={total_steps}, completed_steps={completed_steps} (type: {type(completed_steps)})", "PipelineOrchestrator")
                success_rate = 0.0
            
            await ws_manager.end_pipeline(success, 
                f"Pipeline completed: {pipeline_results['summary']['completed']} steps succeeded, "
                f"{pipeline_results['summary']['failed']} steps failed (Success rate: {success_rate:.1%})")
            
            # Complete pipeline tracking
            if self.trace_id:
                pipeline_tracker.complete_pipeline_execution(
                    trace_id=self.trace_id,
                    status="completed" if success else "failed"
                )
            
            logger.info(f"Pipeline completed in {pipeline_results['total_duration']:.2f}s")
            return pipeline_results
            
        except Exception as e:
            logger.error(f"Pipeline execution failed with exception: {str(e)}", "PipelineOrchestrator")
            logger.exception("Full exception details:", "PipelineOrchestrator")
            
            # Additional debug info for division by zero
            if "division by zero" in str(e).lower():
                logger.error(f"Division by zero detected! Steps count: {len(self.steps)}", "PipelineOrchestrator")
                logger.error(f"Pipeline results summary: {pipeline_results.get('summary', 'Not available')}", "PipelineOrchestrator")
                logger.error(f"Pipeline steps: {[step.name for step in self.steps]}", "PipelineOrchestrator")
                logger.error(f"Pipeline start time: {self.start_time}", "PipelineOrchestrator")
                logger.error(f"Pipeline end time: {self.end_time}", "PipelineOrchestrator")
            
            # Complete pipeline tracking with failure
            if self.trace_id:
                error_step = None
                if "division by zero" in str(e).lower():
                    error_step = "calculation"
                pipeline_tracker.complete_pipeline_execution(
                    trace_id=self.trace_id,
                    status="failed",
                    error_message=str(e),
                    error_step=error_step
                )
            
            # Ensure we end the pipeline properly even if there's an error
            try:
                await ws_manager.end_pipeline(False, f"Pipeline failed: {str(e)}")
            except Exception as ws_error:
                logger.error(f"Error ending pipeline in WebSocket: {ws_error}", "PipelineOrchestrator")
            
            raise

class PipelineFactory:
    """Factory for creating different types of pipelines"""
    
    @staticmethod
    def create_full_pipeline(tickers: List[str] = None, 
                           sources: List[str] = None,
                           period: str = "90d",
                           interval: str = "1d",
                           max_days: int = 90,
                           feature: str = "Close",
                           test_size: float = 0.2,
                           include_eda: bool = True,
                           include_clustering: bool = True,
                           trace_id: str = None) -> PipelineOrchestrator:
        """
        Create a complete analysis pipeline
        
        Args:
            tickers: List of tickers or ["TOP30"] for top 30
            sources: List of sources ['yfinance', 'binance']
            period: Data period
            interval: Data interval
            max_days: Max days for preprocessing
            feature: Feature for model training
            test_size: Test split size
            include_eda: Whether to include EDA analysis
            include_clustering: Whether to include clustering analysis
        """
        
        logger.info("Creating full pipeline", "PipelineFactory")
        
        if tickers is None:
            tickers = ["TOP30"]
        if sources is None:
            sources = ["yfinance", "binance"]
        
        # Add 5-year option
        if period == "5y":
            period = "1825d"
        
        logger.info(f"Pipeline config: tickers={tickers}, sources={sources}, period={period}, interval={interval}", "PipelineFactory")
        logger.info(f"Pipeline options: max_days={max_days}, feature={feature}, test_size={test_size}", "PipelineFactory")
        logger.info(f"Pipeline flags: include_eda={include_eda}, include_clustering={include_clustering}", "PipelineFactory")
        
        orchestrator = PipelineOrchestrator(trace_id=trace_id)
        initial_step_count = 0
        
        # Step 1: Data Download
        logger.info("Adding DataDownloadStep", "PipelineFactory")
        orchestrator.add_step(DataDownloadStep(tickers, sources, period, interval))
        initial_step_count += 1
        
        # Step 2: Data Preprocessing (for each source)
        logger.info(f"Adding DataPreprocessingStep for {len(sources)} sources: {sources}", "PipelineFactory")
        for source in sources:
            orchestrator.add_step(DataPreprocessingStep(source, max_days))
            initial_step_count += 1
        
        # Step 3: EDA Analysis (for sample tickers if requested)
        if include_eda and tickers != ["TOP30"]:
            for ticker in tickers[:3]:  # Limit to first 3 tickers for EDA
                # Use database-based EDA
                orchestrator.add_step(EDAStep(ticker, use_database=True))
        
        # Step 4: Dimensionality Reduction and Clustering (if requested)
        if include_clustering:
            # Use the first available preprocessed file
            for source in sources:
                # Use database-based dimensionality reduction and clustering
                orchestrator.add_step(ConditionalDimensionalityReductionStep(
                    use_database=True, 
                    max_days=max_days, 
                    source=source
                ))
                # Add clustering step with database support
                orchestrator.add_step(ClusteringStep(
                    use_database=True,
                    source=source
                ))
                break  # Only use one source for clustering
        
        # Step 4.5: Correlation Analysis (for sample tickers)
        correlation_tickers = tickers if tickers != ["TOP30"] else ["BTC", "ETH", "ADA", "DOT"]
        if len(correlation_tickers) >= 2:  # Need at least 2 tickers for correlation
            logger.info(f"Adding CorrelationAnalysisStep for tickers: {correlation_tickers[:4]}", "PipelineFactory")
            orchestrator.add_step(CorrelationAnalysisStep(
                tickers=correlation_tickers[:4],  # Limit to 4 tickers
                feature=feature,
                source=sources[0] if sources else 'yfinance',  # Use first source
                use_database=True
            ))
            initial_step_count += 1
        
        # Step 5: Model Training (for specific tickers or sample from TOP30)
        training_tickers = tickers if tickers != ["TOP30"] else ["BTC", "ETH", "ADA"]  # Sample tickers
        logger.info(f"Adding ModelTrainingStep for tickers: {training_tickers}", "PipelineFactory")
        orchestrator.add_step(ModelTrainingStep(training_tickers, feature, test_size))
        initial_step_count += 1
        
        final_step_count = len(orchestrator.steps)
        logger.info(f"Pipeline creation completed: expected {initial_step_count} steps, actual {final_step_count} steps", "PipelineFactory")
        
        # Debug step information
        step_names = [step.name for step in orchestrator.steps]
        logger.info(f"Created pipeline steps: {step_names}", "PipelineFactory")
        
        if final_step_count == 0:
            logger.error("CRITICAL: Pipeline created with 0 steps! This will cause division by zero!", "PipelineFactory")
            logger.error(f"Debug - Pipeline creation parameters: tickers={tickers}, sources={sources}, include_eda={include_eda}, include_clustering={include_clustering}", "PipelineFactory")
            raise ValueError("Pipeline creation failed - no steps were added")
        elif final_step_count != initial_step_count:
            logger.warning(f"Step count mismatch: expected {initial_step_count}, got {final_step_count}", "PipelineFactory")
        
        return orchestrator
    
    @staticmethod
    def create_download_and_train_pipeline(tickers: List[str],
                                         sources: List[str] = None,
                                         period: str = "90d",
                                         interval: str = "1d",
                                         feature: str = "Close",
                                         test_size: float = 0.2,
                                         trace_id: str = None) -> PipelineOrchestrator:
        """Create a simple download and train pipeline"""
        
        if sources is None:
            sources = ["yfinance"]
        
        orchestrator = PipelineOrchestrator(trace_id=trace_id)
        
        # Download data
        orchestrator.add_step(DataDownloadStep(tickers, sources, period, interval))
        
        # Train models
        orchestrator.add_step(ModelTrainingStep(tickers, feature, test_size))
        
        return orchestrator
    
    @staticmethod
    def create_preprocessing_pipeline(sources: List[str] = None,
                                    max_days: int = 90,
                                    trace_id: str = None) -> PipelineOrchestrator:
        """Create a preprocessing-only pipeline"""
        
        if sources is None:
            sources = ["yfinance", "binance"]
        
        orchestrator = PipelineOrchestrator(trace_id=trace_id)
        
        for source in sources:
            orchestrator.add_step(DataPreprocessingStep(source, max_days))
        
        return orchestrator

# Initialize the orchestrator instance
pipeline_factory = PipelineFactory()
