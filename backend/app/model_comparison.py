"""
Model Comparison and Evaluation Module
Handles training, evaluation, and comparison of multiple models
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import json
import os
import joblib
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, TimeSeriesSplit
import logging

from .logger import setup_logger
from .database import crypto_db
from .unified_data_handler import unified_handler
from .predictive_modeling import (
    train_arima,
    train_sarima,
    train_random_forest,
    train_xgboost,
    evaluate_forecasts
)

# Setup logging
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
logger = setup_logger("model_comparison", os.path.join(LOG_DIR, "model_comparison.log"))

# Model directory
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

class ModelComparison:
    """Handles training and comparison of multiple models"""
    
    def __init__(self):
        self.db = crypto_db
        self.models = ['arima', 'sarima', 'random_forest', 'xgboost', 'lstm']
        self.metrics = {}
        
    def prepare_time_series_data(self, ticker: str, feature: str = 'Close',
                                 source: Optional[str] = None,
                                 start_date: Optional[datetime] = None,
                                 end_date: Optional[datetime] = None) -> pd.Series:
        """
        Prepare time series data for modeling
        
        Args:
            ticker: Cryptocurrency ticker
            feature: Feature to predict (default 'Close')
            source: Data source (optional)
            start_date: Start date for data
            end_date: End date for data
            
        Returns:
            Time series data as pandas Series
        """
        try:
            # Get data from database
            standard_ticker = unified_handler.denormalize_ticker(ticker)
            df = self.db.get_ohlcv_data(standard_ticker, source, start_date, end_date)
            
            if df.empty:
                logger.error(f"No data found for {ticker}")
                return pd.Series()
            
            # Extract the feature
            if feature not in df.columns:
                logger.error(f"Feature {feature} not found in data")
                return pd.Series()
            
            series = df[feature].dropna()
            series.name = f"{ticker}_{feature}"
            
            return series
            
        except Exception as e:
            logger.error(f"Error preparing time series data: {e}")
            return pd.Series()
    
    def train_all_models(self, ticker: str, feature: str = 'Close',
                        test_size: float = 0.2,
                        source: Optional[str] = None) -> Dict[str, Any]:
        """
        Train all available models for a ticker
        
        Args:
            ticker: Cryptocurrency ticker
            feature: Feature to predict
            test_size: Fraction of data to use for testing
            source: Data source
            
        Returns:
            Dictionary with model results and metrics
        """
        try:
            results = {
                'ticker': ticker,
                'feature': feature,
                'models': {},
                'best_model': None,
                'training_date': datetime.now().isoformat()
            }
            
            # Prepare data
            series = self.prepare_time_series_data(ticker, feature, source)
            if series.empty:
                logger.error(f"No data available for {ticker}")
                return results
            
            # Split data
            split_index = int(len(series) * (1 - test_size))
            train = series[:split_index]
            test = series[split_index:]
            
            logger.info(f"Training models for {ticker} - Train: {len(train)}, Test: {len(test)}")
            
            # Store test data info
            results['data_info'] = {
                'total_samples': len(series),
                'train_samples': len(train),
                'test_samples': len(test),
                'train_start': str(train.index[0]),
                'train_end': str(train.index[-1]),
                'test_start': str(test.index[0]),
                'test_end': str(test.index[-1])
            }
            
            best_rmse = float('inf')
            
            # Train ARIMA
            if 'arima' in self.models:
                try:
                    logger.info(f"Training ARIMA for {ticker}")
                    arima_model = train_arima(ticker, train)
                    arima_forecast = arima_model.forecast(steps=len(test))
                    arima_metrics = self.calculate_metrics(test, arima_forecast)
                    
                    # Save model
                    model_path = os.path.join(MODEL_DIR, f"{ticker}_arima.pkl")
                    joblib.dump(arima_model, model_path)
                    
                    results['models']['arima'] = {
                        'metrics': arima_metrics,
                        'model_path': model_path,
                        'parameters': {
                            'order': arima_model.order if hasattr(arima_model, 'order') else None
                        }
                    }
                    
                    if arima_metrics['rmse'] < best_rmse:
                        best_rmse = arima_metrics['rmse']
                        results['best_model'] = 'arima'
                        
                except Exception as e:
                    logger.error(f"Error training ARIMA: {e}")
                    results['models']['arima'] = {'error': str(e)}
            
            # Train SARIMA
            if 'sarima' in self.models:
                try:
                    logger.info(f"Training SARIMA for {ticker}")
                    sarima_model = train_sarima(ticker, train)
                    sarima_forecast = sarima_model.forecast(steps=len(test))
                    sarima_metrics = self.calculate_metrics(test, sarima_forecast)
                    
                    # Save model
                    model_path = os.path.join(MODEL_DIR, f"{ticker}_sarima.pkl")
                    joblib.dump(sarima_model, model_path)
                    
                    results['models']['sarima'] = {
                        'metrics': sarima_metrics,
                        'model_path': model_path,
                        'parameters': {
                            'order': sarima_model.order if hasattr(sarima_model, 'order') else None,
                            'seasonal_order': sarima_model.seasonal_order if hasattr(sarima_model, 'seasonal_order') else None
                        }
                    }
                    
                    if sarima_metrics['rmse'] < best_rmse:
                        best_rmse = sarima_metrics['rmse']
                        results['best_model'] = 'sarima'
                        
                except Exception as e:
                    logger.error(f"Error training SARIMA: {e}")
                    results['models']['sarima'] = {'error': str(e)}
            
            # Train Random Forest
            if 'random_forest' in self.models:
                try:
                    logger.info(f"Training Random Forest for {ticker}")
                    rf_model, rf_lag = train_random_forest(ticker, train)
                    
                    # Create lagged features for test data
                    test_features = self.create_lagged_features(pd.concat([train, test]), rf_lag, len(test))
                    rf_forecast = pd.Series(rf_model.predict(test_features), index=test.index)
                    rf_metrics = self.calculate_metrics(test, rf_forecast)
                    
                    # Save model
                    model_path = os.path.join(MODEL_DIR, f"{ticker}_rf.pkl")
                    joblib.dump((rf_model, rf_lag), model_path)
                    
                    results['models']['random_forest'] = {
                        'metrics': rf_metrics,
                        'model_path': model_path,
                        'parameters': {
                            'n_estimators': rf_model.n_estimators,
                            'max_depth': rf_model.max_depth,
                            'lag_features': rf_lag
                        }
                    }
                    
                    if rf_metrics['rmse'] < best_rmse:
                        best_rmse = rf_metrics['rmse']
                        results['best_model'] = 'random_forest'
                        
                except Exception as e:
                    logger.error(f"Error training Random Forest: {e}")
                    results['models']['random_forest'] = {'error': str(e)}
            
            # Train XGBoost
            if 'xgboost' in self.models:
                try:
                    logger.info(f"Training XGBoost for {ticker}")
                    xgb_model, xgb_lag = train_xgboost(ticker, train)
                    
                    # Create lagged features for test data
                    test_features = self.create_lagged_features(pd.concat([train, test]), xgb_lag, len(test))
                    xgb_forecast = pd.Series(xgb_model.predict(test_features), index=test.index)
                    xgb_metrics = self.calculate_metrics(test, xgb_forecast)
                    
                    # Save model
                    model_path = os.path.join(MODEL_DIR, f"{ticker}_xgb.pkl")
                    joblib.dump((xgb_model, xgb_lag), model_path)
                    
                    results['models']['xgboost'] = {
                        'metrics': xgb_metrics,
                        'model_path': model_path,
                        'parameters': {
                            'n_estimators': xgb_model.n_estimators,
                            'max_depth': xgb_model.max_depth,
                            'learning_rate': xgb_model.learning_rate,
                            'lag_features': xgb_lag
                        }
                    }
                    
                    if xgb_metrics['rmse'] < best_rmse:
                        best_rmse = xgb_metrics['rmse']
                        results['best_model'] = 'xgboost'
                        
                except Exception as e:
                    logger.error(f"Error training XGBoost: {e}")
                    results['models']['xgboost'] = {'error': str(e)}
            
            # Save results to database
            self.save_model_metadata(ticker, results)
            
            logger.info(f"Model training completed for {ticker}. Best model: {results['best_model']}")
            return results
            
        except Exception as e:
            logger.error(f"Error in train_all_models: {e}")
            return {'error': str(e)}
    
    def calculate_metrics(self, actual: pd.Series, predicted: pd.Series) -> Dict[str, float]:
        """
        Calculate evaluation metrics with division by zero protection
        
        Args:
            actual: Actual values
            predicted: Predicted values
            
        Returns:
            Dictionary of metrics
        """
        try:
            # Align indices
            actual, predicted = actual.align(predicted, join='inner')
            
            # Check for empty data
            if len(actual) == 0 or len(predicted) == 0:
                logger.warning("Empty data provided for metrics calculation")
                return {
                    'rmse': 0.0,
                    'mae': 0.0,
                    'mape': 0.0,
                    'r2': 0.0,
                    'direction_accuracy': 0.0
                }
            
            # Basic metrics
            rmse = float(np.sqrt(mean_squared_error(actual, predicted)))
            mae = float(mean_absolute_error(actual, predicted))
            r2 = float(r2_score(actual, predicted))
            
            # MAPE with division by zero protection
            mask_nonzero = actual != 0
            if np.any(mask_nonzero):
                mape = float(np.mean(np.abs((actual[mask_nonzero] - predicted[mask_nonzero]) / actual[mask_nonzero])) * 100)
            else:
                logger.warning("All actual values are zero, cannot calculate MAPE")
                mape = 0.0
            
            # Direction accuracy with diff protection
            actual_diff = actual.diff().dropna()
            predicted_diff = predicted.diff().dropna()
            
            if len(actual_diff) > 0 and len(predicted_diff) > 0:
                # Align the diff series
                actual_diff, predicted_diff = actual_diff.align(predicted_diff, join='inner')
                if len(actual_diff) > 0:
                    direction_accuracy = float(np.mean(np.sign(actual_diff) == np.sign(predicted_diff)) * 100)
                else:
                    direction_accuracy = 0.0
            else:
                direction_accuracy = 0.0
            
            metrics = {
                'rmse': rmse,
                'mae': mae,
                'mape': mape,
                'r2': r2,
                'direction_accuracy': direction_accuracy
            }
            
            # Sanitize any NaN or infinite values
            for key, value in metrics.items():
                if np.isnan(value) or np.isinf(value):
                    logger.warning(f"Invalid {key} value: {value}, setting to 0.0")
                    metrics[key] = 0.0
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            return {
                'rmse': 0.0,
                'mae': 0.0,
                'mape': 0.0,
                'r2': 0.0,
                'direction_accuracy': 0.0,
                'error': str(e)
            }
    
    def create_lagged_features(self, series: pd.Series, lag: int, n_samples: int) -> np.ndarray:
        """
        Create lagged features for tree-based models
        
        Args:
            series: Time series data
            lag: Number of lag features
            n_samples: Number of samples to generate
            
        Returns:
            Feature matrix
        """
        try:
            features = []
            series_values = series.values
            
            start_idx = len(series_values) - n_samples
            for i in range(start_idx, len(series_values)):
                if i >= lag:
                    features.append(series_values[i-lag:i])
            
            return np.array(features)
            
        except Exception as e:
            logger.error(f"Error creating lagged features: {e}")
            return np.array([])
    
    def save_model_metadata(self, ticker: str, results: Dict):
        """Save model metadata to database"""
        try:
            import sqlite3
            from .database import DB_PATH
            
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            
            for model_name, model_info in results['models'].items():
                if 'error' not in model_info:
                    cursor.execute("""
                        INSERT OR REPLACE INTO model_metadata
                        (ticker, model_type, feature, train_start_date, train_end_date,
                         test_size, metrics, model_path, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                    """, (
                        ticker,
                        model_name,
                        results['feature'],
                        results['data_info']['train_start'],
                        results['data_info']['train_end'],
                        results['data_info']['test_samples'] / results['data_info']['total_samples'],
                        json.dumps(model_info['metrics']),
                        model_info['model_path']
                    ))
            
            conn.commit()
            conn.close()
            logger.info(f"Saved model metadata for {ticker}")
            
        except Exception as e:
            logger.error(f"Error saving model metadata: {e}")
    
    def get_model_comparison_report(self, ticker: str) -> Dict:
        """
        Get comprehensive model comparison report
        
        Args:
            ticker: Cryptocurrency ticker
            
        Returns:
            Comparison report with all models and metrics
        """
        try:
            import sqlite3
            from .database import DB_PATH
            
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT model_type, feature, metrics, train_start_date, train_end_date, updated_at
                FROM model_metadata
                WHERE ticker = ?
                ORDER BY updated_at DESC
            """, (ticker,))
            
            rows = cursor.fetchall()
            conn.close()
            
            if not rows:
                return {'message': f'No models found for {ticker}'}
            
            report = {
                'ticker': ticker,
                'models': {},
                'best_model': None,
                'comparison_date': datetime.now().isoformat()
            }
            
            best_rmse = float('inf')
            
            for row in rows:
                model_type, feature, metrics_json, train_start, train_end, updated_at = row
                metrics = json.loads(metrics_json)
                
                if model_type not in report['models']:
                    report['models'][model_type] = {
                        'feature': feature,
                        'metrics': metrics,
                        'train_period': f"{train_start} to {train_end}",
                        'last_updated': updated_at
                    }
                    
                    if metrics.get('rmse', float('inf')) < best_rmse:
                        best_rmse = metrics['rmse']
                        report['best_model'] = model_type
            
            return report
            
        except Exception as e:
            logger.error(f"Error getting model comparison report: {e}")
            return {'error': str(e)}
    
    def cross_validate_models(self, ticker: str, feature: str = 'Close',
                            n_splits: int = 5) -> Dict:
        """
        Perform time series cross-validation for all models
        
        Args:
            ticker: Cryptocurrency ticker
            feature: Feature to predict
            n_splits: Number of cross-validation splits
            
        Returns:
            Cross-validation results
        """
        try:
            # Prepare data
            series = self.prepare_time_series_data(ticker, feature)
            if series.empty:
                return {'error': f'No data available for {ticker}'}
            
            # Initialize TimeSeriesSplit
            tscv = TimeSeriesSplit(n_splits=n_splits)
            
            cv_results = {
                'ticker': ticker,
                'feature': feature,
                'n_splits': n_splits,
                'models': {}
            }
            
            for model_name in self.models:
                cv_scores = {
                    'rmse': [],
                    'mae': [],
                    'mape': []
                }
                
                for train_idx, test_idx in tscv.split(series):
                    train_fold = series.iloc[train_idx]
                    test_fold = series.iloc[test_idx]
                    
                    try:
                        if model_name == 'arima':
                            model = train_arima(ticker, train_fold)
                            forecast = model.forecast(steps=len(test_fold))
                        elif model_name == 'sarima':
                            model = train_sarima(ticker, train_fold)
                            forecast = model.forecast(steps=len(test_fold))
                        elif model_name == 'random_forest':
                            model, lag = train_random_forest(ticker, train_fold)
                            test_features = self.create_lagged_features(series, lag, len(test_fold))
                            forecast = pd.Series(model.predict(test_features), index=test_fold.index)
                        elif model_name == 'xgboost':
                            model, lag = train_xgboost(ticker, train_fold)
                            test_features = self.create_lagged_features(series, lag, len(test_fold))
                            forecast = pd.Series(model.predict(test_features), index=test_fold.index)
                        else:
                            continue
                        
                        metrics = self.calculate_metrics(test_fold, forecast)
                        cv_scores['rmse'].append(metrics['rmse'])
                        cv_scores['mae'].append(metrics['mae'])
                        cv_scores['mape'].append(metrics['mape'])
                        
                    except Exception as e:
                        logger.warning(f"Error in CV fold for {model_name}: {e}")
                        continue
                
                if cv_scores['rmse']:
                    cv_results['models'][model_name] = {
                        'mean_rmse': float(np.mean(cv_scores['rmse'])),
                        'std_rmse': float(np.std(cv_scores['rmse'])),
                        'mean_mae': float(np.mean(cv_scores['mae'])),
                        'std_mae': float(np.std(cv_scores['mae'])),
                        'mean_mape': float(np.mean(cv_scores['mape'])),
                        'std_mape': float(np.std(cv_scores['mape']))
                    }
            
            # Determine best model based on mean RMSE
            if cv_results['models']:
                best_model = min(cv_results['models'].items(), 
                               key=lambda x: x[1]['mean_rmse'])
                cv_results['best_model'] = best_model[0]
            
            return cv_results
            
        except Exception as e:
            logger.error(f"Error in cross-validation: {e}")
            return {'error': str(e)}

# Initialize model comparison instance
model_comparison = ModelComparison()