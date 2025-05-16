import pandas as pd
import numpy as np
from .predictive_modeling import compute_rsi, compute_macd, backtest_signals

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

# Backtest including technical indicators

def backtest_ticker(prices: pd.Series, forecast: pd.Series, threshold: float):
    # 1) Generate signals DataFrame
    signals_df = generate_signals(forecast, threshold)
    signals_orig = signals_df['signal']

    # 2) Compute performance using original timestamp index
    perf = backtest_signals(prices, signals_orig)

    # 3) Stringify and export signals
    signals_series = signals_orig.copy()
    signals_series.index = signals_series.index.astype(str)
    signals_dict = signals_series.to_dict()

    # 4) Compute RSI and stringify
    rsi = compute_rsi(prices).dropna()
    rsi.index = rsi.index.astype(str)
    rsi_dict = rsi.to_dict()

    # 5) Compute MACD and stringify
    macd_df = compute_macd(prices).dropna()
    macd_df.index = macd_df.index.astype(str)
    macd_dict = macd_df.to_dict(orient='index')

    return {
        'signals': signals_dict,
        'rsi': rsi_dict,
        'macd': macd_dict,
        'performance': perf
    }
