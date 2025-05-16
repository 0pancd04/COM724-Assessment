import os
import logging
from logging.handlers import RotatingFileHandler
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse
import pandas as pd
import plotly.io as pio

from .logger import setup_logger
from .data_preprocessing import preprocess_data
from .data_downloader import download_data_yfinance, get_top_30_coins, flatten_ticker_data
from .download_binance_data import download_binance_ohlcv
from .grouping_analysis import perform_dimensionality_reduction, perform_clustering_analysis
from .correlation_analysis import perform_correlation_analysis
from .eda_analysis import perform_eda_analysis, unflatten_ticker_data
from .predictive_modeling import (
    train_arima,
    train_sarima,
    train_random_forest,
    train_xgboost,
    train_lstm,
    evaluate_forecasts,
    compute_rsi,
    compute_macd
)
from .forecasting import (
    load_model,
    forecast_arima,
    forecast_sarima,
    forecast_random_forest,
    forecast_xgboost,
    forecast_lstm
)
from .signals import generate_signals, estimate_pnl, backtest_ticker
from sklearn.model_selection import train_test_split

# --- Paths and Logging ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
app_logger = setup_logger("app_logger", os.path.join(LOG_DIR, "app.log"))


# Uvicorn logger
uvicorn_logger = logging.getLogger("uvicorn")
uvicorn_handler = RotatingFileHandler(
    os.path.join(LOG_DIR, "uvicorn.log"), maxBytes=5 * 1024 * 1024, backupCount=3
)
uvicorn_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
uvicorn_handler.setFormatter(uvicorn_formatter)
uvicorn_logger.addHandler(uvicorn_handler)

# --- FastAPI App ---
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins= ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def clear_logs():
    """Remove old log files on startup."""
    for fname in os.listdir(LOG_DIR):
        file_path = os.path.join(LOG_DIR, fname)
        if os.path.isfile(file_path):
            try:
                os.remove(file_path)
            except PermissionError:
                pass
    app_logger.info(f"Cleared logs in {LOG_DIR}")


@app.get("/")
async def read_root():
    app_logger.info("Root endpoint accessed")
    return {"message": "Welcome to the Crypto Prediction Platform"}

@app.get("/download/{ticker}")
async def download_ticker_data(
    ticker: str,
    period: str = Query("5y", description="Time period (e.g. '5y' for 5 years)"),
    interval: str = Query("1d", description="Data interval (e.g. '1d' for daily)")
):
    """
    Download historical data for a given crypto ticker.
    """
    try:
        data = download_data_yfinance(ticker, period=period, interval=interval)
        if data.empty:
            raise HTTPException(status_code=404, detail="No data found for the provided ticker.")
        app_logger.info(f"Downloaded data for ticker: {ticker}")
        # Return data as JSON
        return JSONResponse(content=data.to_dict())
    except Exception as e:
        app_logger.error(f"Error downloading data for {ticker}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/download_all")
async def download_all_data(
    period: str = Query("90d", description="For yfinance: '5y'; for Binance: days as '90d'"),
    interval: str = Query("1d", description="Data interval (e.g. '1d' for daily)"),
    datasource: str = Query("yfinance", description="'yfinance' or 'binance'")
):
    """
    Download and flatten data for top tickers from the selected datasource.
    For 'binance', period must be number of days as string, e.g. '90d'.
    """
    try:
        """
        Download and flatten data for top tickers from the selected datasource.
        Stops once 30 tickers have been successfully downloaded & flattened.
        """
        tickers = get_top_30_coins()
        if not tickers:
            raise HTTPException(404, "Could not retrieve top tickers.")
        
        flattened_data = {}
        for ticker in tickers:
            # Stop as soon as we have 30 valid tickers
            if len(flattened_data) >= 30:
                break

            try:
                if datasource == "yfinance":
                    df = download_data_yfinance(ticker, period=period, interval=interval)
                    if df.empty:
                        app_logger.warning(f"YFinance empty for {ticker}, skipping")
                        continue
                    flat = flatten_ticker_data(df)
                
                elif datasource == "binance":
                    try:
                        days = int(period.rstrip("d"))
                    except:
                        days = 90
                    df = download_binance_ohlcv([ticker], days=days, interval=interval)
                    if df is None or df.empty:
                        app_logger.warning(f"Binance empty for {ticker}, skipping")
                        continue
                    df = df.set_index("Open Time")[["Close"]]
                    df.index.name = "Date"
                    flat = flatten_ticker_data(df)
                
                else:
                    raise HTTPException(400, "Invalid datasource.")
                
                flattened_data[ticker] = flat
                app_logger.info(f"Flattened data for {ticker} ({len(flattened_data)}/30)")
            
            except Exception as e:
                app_logger.error(f"Error with {ticker}: {e}", exc_info=True)
                # skip this one, continue to next

        if len(flattened_data) == 0:
            raise HTTPException(500, "No tickers could be downloaded/flattened.")
        
        # Combine & save
        combined_df = pd.DataFrame.from_dict(flattened_data, orient="index")
        data_dir = os.path.join(BASE_DIR, "..", "..", "data")
        os.makedirs(data_dir, exist_ok=True)
        filename = f"{datasource}_{interval}_{period}.csv"
        combined_df.to_csv(os.path.join(data_dir, filename))
        app_logger.info(f"Saved {len(flattened_data)} tickers to {filename}")

        return JSONResponse({"message": "Data downloaded and stored", "file": filename})

    except HTTPException:
        raise
    except Exception as e:
        app_logger.exception(f"Unexpected error in download_all_data: {e}")
        raise HTTPException(status_code=500, detail="Server error during data download.")



@app.get("/preprocess_data")
async def preprocess_data_api(
    file_path: str = Query("data/yfinance_1d_5y.csv", description="Path to the raw CSV file"),
    max_days: int = Query(365, description="Maximum number of days to use")
):
    """
    Executes the data preprocessing function on the raw CSV data,
    stores the preprocessed data in a new CSV file, and returns a report
    detailing what values were updated and why.
    """
    try:
        output_file = "data/preprocessed_yfinance_1d_5y.csv"
        df_scaled, report = preprocess_data(file_path, max_days, output_file=output_file)
        app_logger.info("Preprocessing completed; returning report")
        return JSONResponse(content={"message": "Data preprocessing completed", "report": report})
    except Exception as e:
        app_logger.error(f"Error during preprocessing: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
    
@app.get("/dim_reduction")
async def dim_reduction_api(
    input_file: str = Query("data/preprocessed_yfinance_1d_5y.csv", description="Path to preprocessed CSV file"),
    max_days: int = Query(365, description="Maximum number of days")
):
    """
    Performs dimensionality reduction (comparing PCA and TSNE), selects the best based on silhouette scores,
    stores the reduced data and the interactive chart, and returns a report along with the chart file path.
    """
    try:
        output_file = "data/dim_reduced_best.csv"
        chart_file = "data/dim_reduction_chart.json"
        reduced_df, report, best_algo, fig = perform_dimensionality_reduction(input_file, output_file, chart_file)
        return JSONResponse(content={
            "message": "Dimensionality reduction completed",
            "report": report,
            "best_algorithm": best_algo,
            "chart_file": chart_file
        })
    except Exception as e:
        app_logger.error(f"Error in dimensionality reduction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/dim_reduction_chart", response_class=HTMLResponse)
async def interactive_chart():
    """
    Returns an HTML page rendering the interactive chart from the saved HTML file.
    """
    try:
        # The HTML file is expected to be stored with the same base name as the JSON file
        chart_html_file = "data/dim_reduction_chart.html"
        if not os.path.exists(chart_html_file):
            raise HTTPException(status_code=404, detail="Chart HTML file not found")
        
        with open(chart_html_file, "r") as f:
            html_content = f.read()
        
        return HTMLResponse(content=html_content)
    except Exception as e:
        app_logger.error(f"Error rendering interactive chart HTML: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/clustering_analysis")
async def clustering_analysis_api(
    reduced_file: str = Query("data/dim_reduced_best.csv", description="Path to reduced CSV file")
):
    """
    Performs clustering analysis on the reduced data (comparing KMeans, Agglomerative, and DBSCAN),
    selects the best clustering based on silhouette scores, stores the cluster assignments,
    and returns an interactive graph along with a report.
    """
    try:
        output_file = "data/clustering_result.csv"
        chart_file = "data/clustering_chart.json"
        cluster_df, report, fig = perform_clustering_analysis(reduced_file, output_file, chart_file)
        return JSONResponse(content={
            "message": "Clustering analysis completed",
            "report": report,
            "chart_file_json": report.get("chart_file_json"),
            "chart_file_html": report.get("chart_file_html")
        })
    except Exception as e:
        app_logger.error(f"Error in clustering analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
    
@app.get("/clustering_chart", response_class=HTMLResponse)
async def clustering_chart():
    """
    Returns an HTML page rendering the clustering interactive chart from the saved HTML file.
    """
    try:
        chart_html_file = "data/clustering_chart.html"
        if not os.path.exists(chart_html_file):
            raise HTTPException(status_code=404, detail="Clustering chart HTML file not found")
        
        with open(chart_html_file, "r") as f:
            html_content = f.read()
        
        return HTMLResponse(content=html_content)
    except Exception as e:
        app_logger.error(f"Error rendering clustering chart HTML: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/available_tickers_based_on_clusters_grouping")
async def available_tickers():
    """
    Returns a list of available tickers grouped by cluster.
    This allows the user to see which tickers are available from each cluster.
    """
    clustering_result_file = "data/clustering_result.csv"
    if not os.path.exists(clustering_result_file):
        raise HTTPException(status_code=404, detail="Clustering result file not found. Run clustering analysis first.")
    
    try:
        df = pd.read_csv(clustering_result_file, index_col=0)
        if "Cluster" not in df.columns:
            raise HTTPException(status_code=500, detail="Clustering result file does not contain 'Cluster' column.")
        
        grouped = df.groupby("Cluster").apply(lambda x: x.index.tolist()).to_dict()
        return JSONResponse(content={"available_tickers": grouped})
    except Exception as e:
        app_logger.error(f"Error retrieving available tickers: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
    
@app.get("/correlation_analysis")
async def correlation_analysis_api(
    selected_tickers: str = Query(..., description="Comma-separated list of 4 selected crypto tickers (one from each cluster)"),
    feature: str = Query("Close", description="Feature for correlation analysis, e.g. 'Close'")
):
    """
    Performs correlation analysis for the 4 selected cryptocurrencies.
    Computes the correlation matrix for the chosen feature, identifies the top 4 positive and negative pairs,
    saves the correlation matrix and interactive chart (JSON and HTML), and returns a report.
    """
    try:
        # Convert comma-separated string to list and trim whitespace
        tickers = [t.strip() for t in selected_tickers.split(",")]
        if len(tickers) != 4:
            raise HTTPException(status_code=400, detail="Exactly 4 tickers must be provided.")
        
        preprocessed_file = "data/preprocessed_yfinance_1d_5y.csv"  # Adjust if needed
        output_file = "data/correlation_matrix.csv"
        chart_file = "data/correlation_chart.json"
        
        corr_df, report, fig = perform_correlation_analysis(preprocessed_file, tickers, feature, output_file, chart_file)
        
        return JSONResponse(content={
            "message": "Correlation analysis completed",
            "report": report
        })
    except Exception as e:
        app_logger.error(f"Error in correlation analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
    
@app.get("/correlation_chart", response_class=HTMLResponse)
async def correlation_chart():
    """
    Returns an HTML page rendering the correlation interactive chart from the saved HTML file.
    """
    try:
        chart_html_file = "data/correlation_chart.html"
        if not os.path.exists(chart_html_file):
            raise HTTPException(status_code=404, detail="Correlation chart HTML file not found")
        
        with open(chart_html_file, "r") as f:
            html_content = f.read()
        
        return HTMLResponse(content=html_content)
    except Exception as e:
        app_logger.error(f"Error rendering correlation chart HTML: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
    
@app.get("/eda_analysis")
async def eda_analysis_api(
    ticker: str = Query(..., description="Ticker symbol for the cryptocurrency"),
    preprocessed_file: str = Query("data/preprocessed_yfinance_1d_5y.csv", description="Path to preprocessed CSV file")
):
    """
    Performs Exploratory Data Analysis (EDA) for the selected cryptocurrency.
    Generates interactive charts (temporal trends, histograms, box plots, rolling averages),
    saves them in both JSON and HTML formats, and returns a report with file paths.
    """
    try:
        report, charts = perform_eda_analysis(ticker, preprocessed_file)
        return JSONResponse(content={
            "message": f"EDA analysis completed for {ticker}",
            "report": report,
            "charts": charts
        })
    except Exception as e:
        app_logger.error(f"Error in EDA analysis for {ticker}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    

def get_chart_html(chart_path: str) -> HTMLResponse:
    if not os.path.exists(chart_path):
        raise HTTPException(status_code=404, detail=f"Chart file {chart_path} not found. Please run EDA analysis first.")
    with open(chart_path, "r") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

@app.get("/eda_chart/temporal_line", response_class=HTMLResponse)
async def get_temporal_line_chart(ticker: str = Query(..., description="Ticker symbol for the cryptocurrency")):
    """
    Returns the HTML page for the Temporal Line Chart for the given ticker.
    """
    chart_path = f"data/eda/{ticker}/temporal_line_chart.html"
    return get_chart_html(chart_path)

@app.get("/eda_chart/histograms", response_class=HTMLResponse)
async def get_histograms_chart(ticker: str = Query(..., description="Ticker symbol for the cryptocurrency")):
    """
    Returns the HTML page for the Distribution Histograms for the given ticker.
    """
    chart_path = f"data/eda/{ticker}/histograms.html"
    return get_chart_html(chart_path)

@app.get("/eda_chart/box_plots", response_class=HTMLResponse)
async def get_box_plots_chart(ticker: str = Query(..., description="Ticker symbol for the cryptocurrency")):
    """
    Returns the HTML page for the Box Plots for the given ticker.
    """
    chart_path = f"data/eda/{ticker}/box_plots.html"
    return get_chart_html(chart_path)

@app.get("/eda_chart/rolling_average", response_class=HTMLResponse)
async def get_rolling_average_chart(ticker: str = Query(..., description="Ticker symbol for the cryptocurrency")):
    """
    Returns the HTML page for the 7-Day Rolling Average Chart for the given ticker.
    """
    chart_path = f"data/eda/{ticker}/rolling_average_chart.html"
    return get_chart_html(chart_path)

@app.get("/eda_chart/candlestick", response_class=HTMLResponse)
async def get_candlestick_chart(ticker: str = Query(..., description="Ticker symbol for the cryptocurrency")):
    """
    Returns the HTML page for the Candlestick Chart for the given ticker.
    """
    chart_path = f"data/eda/{ticker}/candlestick_chart.html"
    return get_chart_html(chart_path)

@app.get("/eda_chart/rolling_volatility", response_class=HTMLResponse)
async def get_rolling_volatility_chart(ticker: str = Query(..., description="Ticker symbol for the cryptocurrency")):
    """
    Returns the HTML page for the Rolling Volatility Chart for the given ticker.
    """
    chart_path = f"data/eda/{ticker}/rolling_volatility_chart.html"
    return get_chart_html(chart_path)


@app.post("/train/{ticker}")
async def train_models(
    ticker: str,
    feature: str = Query("Close", description="Feature column"),
    test_size: float = Query(0.2, description="Test split fraction")
):
    """Train ARIMA, SARIMA, RF, XGB, LSTM models for a given ticker"""
    df_ticker = unflatten_ticker_data(ticker, preprocessed_file="data/preprocessed_yfinance_1d_5y.csv")
    series    = df_ticker[feature].dropna()
    train, test = train_test_split(series, test_size=test_size, shuffle=False)

    # 1) Fit each model
    arima_m   = train_arima(ticker, train)
    sarima_m  = train_sarima(ticker, train)
    rf_m, rf_lag    = train_random_forest(ticker, train)
    xgb_m, xgb_lag  = train_xgboost(ticker, train)
    lstm_m, lstm_lag = train_lstm(ticker, train)

    # 2) Forecast on test
    preds = {
        "ARIMA":        arima_m.forecast(steps=len(test)),
        "SARIMA":       sarima_m.forecast(steps=len(test)),
        "RandomForest": forecast_random_forest((rf_m, rf_lag), train, len(test)),
        "XGBoost":      forecast_xgboost((xgb_m, xgb_lag), train, len(test)),
        "LSTM":         forecast_lstm((lstm_m, lstm_lag), train, len(test)),
    }

    # 3) Evaluate
    metrics = {
        name: evaluate_forecasts(test, pred)
        for name, pred in preds.items()
    }

    return JSONResponse(content={"message": "Models trained", "metrics": metrics})


@app.get("/forecast/{ticker}")
async def get_forecast(
    ticker: str,
    model_type: str = Query("arima", description="arima|sarima|rf|xgb|lstm"),
    periods: int = Query(7, description="Number of periods to forecast")
):
    """Return forecast for next `periods` days"""
    # load the requested model
    pkl = load_model(ticker, model_type)
    df_ticker   = unflatten_ticker_data(ticker, "data/preprocessed_yfinance_1d_5y.csv")
    last_series = df_ticker["Close"]

    # dispatch to the right forecaster
    if model_type == "arima":
        fc = forecast_arima(pkl, periods)
    elif model_type == "sarima":
        fc = forecast_sarima(pkl, periods)
    elif model_type == "rf":
        fc = forecast_random_forest(pkl, last_series, periods)
    elif model_type == "xgb":
        fc = forecast_xgboost(pkl, last_series, periods)
    elif model_type == "lstm":
        fc = forecast_lstm(pkl, last_series, periods)
    else:
        raise HTTPException(400, "Unknown model_type")

    # convert the index to ISO‐format strings, then emit an index‐oriented dict
    fc_str = fc.copy()
    fc_str.index = fc_str.index.astype(str)               # e.g. "2025-05-17 00:00:00"
    forecast_payload = fc_str.to_dict(orient="index")

    return JSONResponse(content={
        "ticker": ticker,
        "model": model_type,
        "forecast": forecast_payload
    })



@app.get("/signals/{ticker}")
async def get_signals(
    ticker: str,
    model_type: str = Query("arima", description="arima|sarima|rf|xgb|lstm"),
    periods: int = Query(7, description="Forecast horizon"),
    threshold: float = Query(0.01, description="Threshold for signal generation")
):
    """Generate buy/sell signals and PnL based on forecast"""
    # 1. Reconstruct time series
    df_ticker   = unflatten_ticker_data(ticker, "data/preprocessed_yfinance_1d_5y.csv")
    last_series = df_ticker["Close"]

    # 2. Load model & produce forecast DataFrame
    pkl = load_model(ticker, model_type)
    if model_type == "arima":
        fc_df = forecast_arima(pkl, periods)
    elif model_type == "sarima":
        fc_df = forecast_sarima(pkl, periods)
    elif model_type == "rf":
        fc_df = forecast_random_forest(pkl, last_series, periods)
    elif model_type == "xgb":
        fc_df = forecast_xgboost(pkl, last_series, periods)
    elif model_type == "lstm":
        fc_df = forecast_lstm(pkl, last_series, periods)
    else:
        raise HTTPException(400, "Unknown model_type")

    # 3. Extract the point forecast series and stringify the index
    if "forecast" in fc_df.columns:
        series_fc = fc_df["forecast"].copy()
    else:
        # single-column case
        series_fc = fc_df.iloc[:, 0].copy()
    series_fc.index = series_fc.index.astype(str)

    # 4. Generate signals & PnL
    signals_df = generate_signals(series_fc, threshold)
    # Combine historical + forecast for PnL
    all_prices = pd.concat([last_series, series_fc])
    pnl = estimate_pnl(all_prices, signals_df["signal"])

    # 5. Return JSON-safe dicts
    return JSONResponse(content={
        "signals": signals_df["signal"].to_dict(),
        "pnl": pnl.fillna(0).to_dict()
    })

    

@app.get("/indicators/{ticker}")
async def get_indicators(
    ticker: str,
    window_rsi: int = Query(14),
    fast: int = Query(12),
    slow: int = Query(26),
    signal: int = Query(9)
):
    """Retrieve RSI and MACD for a ticker"""
    # Reconstruct time series
    df_ticker = unflatten_ticker_data(ticker, "data/preprocessed_yfinance_1d_5y.csv")
    prices = df_ticker['Close']

    # Compute RSI and stringify index
    rsi_series = compute_rsi(prices, window_rsi).dropna()
    rsi_series.index = rsi_series.index.astype(str)
    rsi_dict = rsi_series.to_dict()

    # Compute MACD and stringify index
    macd_df = compute_macd(prices, fast, slow, signal).dropna()
    macd_df.index = macd_df.index.astype(str)
    macd_dict = macd_df.to_dict(orient='index')

    return JSONResponse(content={
        'rsi': rsi_dict,
        'macd': macd_dict
    })

@app.get("/forecast_outputs/{ticker}")
async def forecast_outputs(
    ticker: str,
    model_type: str = Query('arima', description='Model type'),
    short_days: int = Query(1),
    short_weeks: int = Query(7),
    medium_month: int = Query(30),
    medium_quarter: int = Query(90)
):
    "Returns multiple horizons with confidence intervals and past accuracy"
    # load and forecast for each horizon
    horizons = {
        'short_day': short_days,
        'short_week': short_weeks,
        'medium_month': medium_month,
        'medium_quarter': medium_quarter
    }
    df = unflatten_ticker_data(ticker, "data/preprocessed_yfinance_1d_5y.csv")
    series = df['Close']
    pkl = load_model(ticker, model_type)
    outputs = {}
    for name, days in horizons.items():
        # 1) generate the raw forecast DataFrame
        fc = globals()[f'forecast_{model_type}'](pkl, days)

        # 2) make a copy and stringify the index
        fc_str = fc.copy()
        fc_str.index = fc_str.index.astype(str)

        # 3) emit an orient="index" dict so each timestamp-string maps to its row dict
        outputs[name] = fc_str.to_dict(orient="index")

    # accuracy: use evaluate on last short_window days
    # user can call /train to get full metrics
    return JSONResponse(content={'forecasts': outputs})

@app.get("/backtest/{ticker}")
async def backtest(
    ticker: str,
    model_type: str = Query('arima'),
    periods: int = Query(7),
    threshold: float = Query(0.01)
):
    "Run backtest with signals, indicators, and performance"
    df = unflatten_ticker_data(ticker, "data/preprocessed_yfinance_1d_5y.csv")
    prices = df['Close']
    pkl = load_model(ticker, model_type)
    forecast_df = globals()[f'forecast_{model_type}'](pkl, periods)
    result = backtest_ticker(prices, forecast_df['forecast'], threshold)
    return JSONResponse(content=result)