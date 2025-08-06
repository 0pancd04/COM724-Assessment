import os
import pickle
import joblib
import numpy as np
import pandas as pd
from datetime import timedelta
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

MODEL_DIR = "models"

def load_model(ticker: str, model_type: str):
    path = os.path.join(MODEL_DIR, f"{ticker}_{model_type}.pkl")
    try:
        # Try joblib first (used by model_comparison.py)
        return joblib.load(path)
    except Exception:
        # Fallback to pickle (used by predictive_modeling.py)
        with open(path, "rb") as f:
            return pickle.load(f)

# Forecast with confidence intervals for ARIMA/SARIMA

def forecast_arima(model, steps: int):
    fc = model.get_forecast(steps=steps)
    mean = fc.predicted_mean
    ci   = fc.conf_int()
    df = pd.DataFrame({"forecast": mean, "lower": ci.iloc[:,0], "upper": ci.iloc[:,1]})
    return df


def forecast_sarima(model, steps: int):
    fc = model.get_forecast(steps=steps)
    mean = fc.predicted_mean
    ci   = fc.conf_int()
    df = pd.DataFrame({"forecast": mean, "lower": ci.iloc[:,0], "upper": ci.iloc[:,1]})
    return df

# Non-parametric models return point forecasts only

def forecast_random_forest(pkl, last_series: pd.Series, steps: int):
    model, n_lags = pkl
    history = last_series.tolist()
    preds = []
    for _ in range(steps):
        x = np.array(history[-n_lags:]).reshape(1, -1)
        yhat = model.predict(x)[0]
        preds.append(yhat)
        history.append(yhat)
    idx = pd.date_range(start=last_series.index[-1] + timedelta(1), periods=steps)
    return pd.DataFrame({"forecast": preds}, index=idx)


def forecast_xgboost(pkl, last_series: pd.Series, steps: int):
    model, n_lags = pkl
    history = last_series.tolist()
    preds = []
    for _ in range(steps):
        x = np.array(history[-n_lags:]).reshape(1, -1)
        yhat = model.predict(x)[0]
        preds.append(yhat)
        history.append(yhat)
    idx = pd.date_range(start=last_series.index[-1] + timedelta(1), periods=steps)
    return pd.DataFrame({"forecast": preds}, index=idx)


def forecast_lstm(pkl, last_series: pd.Series, steps: int):
    model, n_lags = pkl
    history = last_series.tolist()
    preds = []
    for _ in range(steps):
        x = np.array(history[-n_lags:]).reshape(1, n_lags, 1)
        yhat = model.predict(x, verbose=0)[0][0]
        preds.append(yhat)
        history.append(yhat)
    idx = pd.date_range(start=last_series.index[-1] + timedelta(1), periods=steps)
    return pd.DataFrame({"forecast": preds}, index=idx)