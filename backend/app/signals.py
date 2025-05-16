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
        price = prices.get(date, np.nan)
        if signal == 'BUY' and position == 0:
            position = 1
            entry_price = price
        elif signal == 'SELL' and position == 1:
            pnl.loc[date] = price - entry_price
            position = 0
    return pnl

# Backtest including technical indicators

def backtest_ticker(prices: pd.Series, forecast: pd.Series, threshold: float):
    # generate signals
    signals_df = generate_signals(forecast, threshold)['signal']
    # compute indicators
    rsi = compute_rsi(prices)
    macd_df = compute_macd(prices)
    # backtest returns
    perf = backtest_signals(prices, signals_df)
    return {
        'signals': signals_df.to_dict(),
        'rsi': rsi.dropna().to_dict(),
        'macd': macd_df.dropna().to_dict(),
        'performance': perf
    }