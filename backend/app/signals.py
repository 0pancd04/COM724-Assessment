import pandas as pd
import numpy as np
import json
from datetime import datetime
from typing import Dict, Optional
from .predictive_modeling import compute_rsi, compute_macd, backtest_signals
from .database import get_db_connection
from .logger import setup_logger
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
logger = setup_logger("signals", os.path.join(LOG_DIR, "signals.log"))

def check_existing_signals(ticker: str, model_type: str, start_date: str = None) -> Optional[Dict]:
    """Check if signals already exist in database"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            query = """
                SELECT signal_date, signal_type, price, predicted_price, 
                       expected_profit, confidence, indicators
                FROM trading_signals 
                WHERE ticker = ? AND indicators LIKE ?
            """
            params = [ticker, f'%"model_type": "{model_type}"%']
            
            if start_date:
                query += " AND signal_date >= ?"
                params.append(start_date)
            
            query += " ORDER BY signal_date"
            cursor.execute(query, params)
            
            rows = cursor.fetchall()
            if rows:
                signals = {}
                for row in rows:
                    date = row[0]
                    signals[date] = {
                        'signal': row[1],
                        'price': row[2],
                        'predicted_price': row[3],
                        'expected_profit': row[4],
                        'confidence': row[5],
                        'indicators': json.loads(row[6]) if row[6] else {}
                    }
                logger.info(f"Found existing signals for {ticker} with {model_type}")
                return signals
    except Exception as e:
        logger.error(f"Error checking existing signals: {e}")
    return None

# Generate simple threshold signals

def generate_signals(forecast: pd.Series, threshold: float = 0.01) -> pd.DataFrame:
    df = pd.DataFrame({'forecast': forecast})
    df['pct_change'] = df['forecast'].pct_change()
    df['signal'] = np.where(df['pct_change'] > threshold, 'BUY',
                      np.where(df['pct_change'] < -threshold, 'SELL', 'HOLD'))
    return df

# PnL from signals

def estimate_pnl(prices: pd.Series, signals: pd.Series) -> pd.Series:
    pnl = pd.Series(dtype=float)
    position = 0
    entry_price = 0.0
    for date, signal in signals.items():
        price = prices.get(pd.to_datetime(date)) if isinstance(date, str) else prices.get(date)
        if signal == 'BUY' and position == 0:
            position = 1
            entry_price = price
        elif signal == 'SELL' and position == 1:
            pnl.loc[date] = price - entry_price
            position = 0
    return pnl

def store_signals_to_db(ticker: str, signals_df: pd.DataFrame, model_type: str, 
                       prices: pd.Series = None, confidence: float = None):
    """Store generated signals to database"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            for date, row in signals_df.iterrows():
                signal_date = date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date)
                price = prices.loc[date] if prices is not None and date in prices.index else row.get('forecast', 0)
                
                # Calculate expected profit if we have forecast
                expected_profit = None
                if 'pct_change' in row:
                    expected_profit = float(row['pct_change'] * 100) if pd.notna(row['pct_change']) else None
                
                indicators = {
                    'model_type': model_type,
                    'threshold': threshold if 'threshold' in locals() else 0.01,
                    'forecast': float(row['forecast']) if 'forecast' in row else None
                }
                
                cursor.execute("""
                    INSERT OR REPLACE INTO trading_signals
                    (ticker, signal_date, signal_type, price, predicted_price, 
                     expected_profit, confidence, indicators, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    ticker, signal_date, row['signal'], float(price),
                    float(row['forecast']) if 'forecast' in row else None,
                    expected_profit, confidence,
                    json.dumps(indicators), datetime.now()
                ))
            
            conn.commit()
            logger.info(f"Stored {len(signals_df)} signals for {ticker}")
            return True
    except Exception as e:
        logger.error(f"Error storing signals: {e}")
        return False

# Backtest including technical indicators

def backtest_ticker(prices: pd.Series, forecast: pd.Series, threshold: float, 
                    ticker: str = None, model_type: str = None):
    # 1) Generate signals DataFrame
    signals_df = generate_signals(forecast, threshold)
    signals_orig = signals_df['signal']

    # 2) Store signals to database if ticker provided
    if ticker and model_type:
        store_signals_to_db(ticker, signals_df, model_type, prices)

    # 3) Compute performance using original timestamp index
    perf = backtest_signals(prices, signals_orig)

    # 4) Stringify and export signals
    signals_series = signals_orig.copy()
    signals_series.index = signals_series.index.astype(str)
    signals_dict = signals_series.to_dict()

    # 5) Compute RSI and stringify
    rsi = compute_rsi(prices).dropna()
    rsi.index = rsi.index.astype(str)
    rsi_dict = rsi.to_dict()

    # 6) Compute MACD and stringify
    macd_df = compute_macd(prices).dropna()
    macd_df.index = macd_df.index.astype(str)
    macd_dict = macd_df.to_dict(orient='index')

    # 7) Store indicators to database if ticker provided
    if ticker:
        store_indicators_to_db(ticker, rsi, macd_df)

    return {
        'signals': signals_dict,
        'rsi': rsi_dict,
        'macd': macd_dict,
        'performance': perf
    }

def store_indicators_to_db(ticker: str, rsi: pd.Series, macd_df: pd.DataFrame):
    """Store technical indicators to database"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Store as JSON in analysis results
            indicators_data = {
                'rsi': rsi.to_dict() if not rsi.empty else {},
                'macd': macd_df.to_dict(orient='index') if not macd_df.empty else {}
            }
            
            cursor.execute("""
                INSERT OR REPLACE INTO eda_results
                (ticker, analysis_type, chart_type, data_json, created_at)
                VALUES (?, ?, ?, ?, ?)
            """, (
                ticker, 'technical_indicators', 'indicators',
                json.dumps(indicators_data), datetime.now()
            ))
            
            conn.commit()
            logger.info(f"Stored technical indicators for {ticker}")
    except Exception as e:
        logger.error(f"Error storing indicators: {e}")
