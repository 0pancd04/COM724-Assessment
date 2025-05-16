import os
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

MODEL_DIR = "models"


def ensure_model_dir():
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)


def evaluate_forecasts(actual: pd.Series, predicted: pd.Series):
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae = mean_absolute_error(actual, predicted)
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    return {"RMSE": rmse, "MAE": mae, "MAPE": mape}


def train_arima(ticker: str, series: pd.Series, order=(5,1,0)):
    ensure_model_dir()
    model = ARIMA(series, order=order).fit()
    path = os.path.join(MODEL_DIR, f"{ticker}_arima.pkl")
    with open(path, "wb") as f:
        pickle.dump(model, f)
    return model


def train_sarima(ticker: str, series: pd.Series, order=(1,1,1), seasonal_order=(1,1,1,12)):
    ensure_model_dir()
    model = SARIMAX(series, order=order, seasonal_order=seasonal_order).fit(disp=False)
    path = os.path.join(MODEL_DIR, f"{ticker}_sarima.pkl")
    with open(path, "wb") as f:
        pickle.dump(model, f)
    return model


def _create_lag_features(series: pd.Series, n_lags: int):
    df = pd.DataFrame({'y': series.values})
    for lag in range(1, n_lags+1):
        df[f'lag_{lag}'] = df['y'].shift(lag)
    return df.dropna()


def train_random_forest(ticker: str, series: pd.Series, n_lags: int = 5):
    ensure_model_dir()
    df_lag = _create_lag_features(series, n_lags)
    X = df_lag.drop('y', axis=1).values
    y = df_lag['y'].values
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X, y)
    path = os.path.join(MODEL_DIR, f"{ticker}_rf.pkl")
    with open(path, "wb") as f:
        pickle.dump((model, n_lags), f)
    return model, n_lags


def train_xgboost(ticker: str, series: pd.Series, n_lags: int = 5):
    ensure_model_dir()
    df_lag = _create_lag_features(series, n_lags)
    X = df_lag.drop('y', axis=1).values
    y = df_lag['y'].values
    model = XGBRegressor(n_estimators=100)
    model.fit(X, y)
    path = os.path.join(MODEL_DIR, f"{ticker}_xgb.pkl")
    with open(path, "wb") as f:
        pickle.dump((model, n_lags), f)
    return model, n_lags


def train_lstm(ticker: str, series: pd.Series, n_lags: int = 5, epochs: int = 20, batch_size: int = 32):
    ensure_model_dir()
    df_lag = _create_lag_features(series, n_lags)
    X = df_lag.drop('y', axis=1).values.reshape(-1, n_lags, 1)
    y = df_lag['y'].values

    model = Sequential([
        LSTM(50, input_shape=(n_lags, 1)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    es = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
    model.fit(X, y, epochs=epochs, batch_size=batch_size, callbacks=[es], verbose=0)

    path = os.path.join(MODEL_DIR, f"{ticker}_lstm.pkl")
    with open(path, "wb") as f:
        pickle.dump((model, n_lags), f)
    return model, n_lags


# Technical Indicators

def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_macd(series: pd.Series,
                 fast_window: int = 12,
                 slow_window: int = 26,
                 signal_window: int = 9) -> pd.DataFrame:
    ema_fast = series.ewm(span=fast_window, adjust=False).mean()
    ema_slow = series.ewm(span=slow_window, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal = macd.ewm(span=signal_window, adjust=False).mean()
    hist = macd - signal
    return pd.DataFrame({"MACD": macd, "Signal": signal, "Hist": hist})

# Backtesting performance

def backtest_signals(prices: pd.Series, signals: pd.Series) -> dict:
    # simple backtest: entry at next open, exit on signal reversal
    returns = []
    position = 0
    entry_price = 0.0
    for date, sig in signals.iteritems():
        price = prices.get(date)
        if sig == "BUY" and position == 0:
            position = 1
            entry_price = price
        elif sig == "SELL" and position == 1:
            ret = (price - entry_price) / entry_price
            returns.append(ret)
            position = 0
    total_return = np.prod([(1 + r) for r in returns]) - 1 if returns else 0
    # Sharpe-like: mean/std * sqrt(N)
    if len(returns) > 1:
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(len(returns))
    else:
        sharpe = None
    return {"total_return": total_return, "trades": len(returns), "sharpe": sharpe}