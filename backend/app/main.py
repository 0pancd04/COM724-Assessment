import os
import logging
from datetime import datetime, timedelta
from logging.handlers import RotatingFileHandler
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
import pandas as pd
import plotly.io as pio

from .logger import setup_enhanced_logger
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

# Import new database and unified handler modules
from .database import crypto_db
from .unified_data_handler import unified_handler
from .model_comparison import model_comparison
from .pipeline_orchestrator import pipeline_factory
from .websocket_manager import ws_manager
from .analysis_storage import analysis_storage
from .rss_feed_handler import rss_handler
from .whatif_scenarios import whatif_analyzer

# --- Paths and Logging ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
app_logger = setup_enhanced_logger("app_logger", os.path.join(LOG_DIR, "app.log"))


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

# WebSocket endpoint for real-time updates
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time pipeline updates"""
    await ws_manager.connect(websocket)
    try:
        while True:
            # Keep connection alive and receive messages
            data = await websocket.receive_text()
            # Echo back or handle commands
            await ws_manager.send_personal_message(f"Echo: {data}", websocket)
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
        app_logger.info("WebSocket client disconnected")


@app.get("/")
async def read_root():
    app_logger.info("Root endpoint accessed")
    return {"message": "Welcome to the Crypto Prediction Platform"}

@app.get("/database/summary")
async def get_database_summary():
    """
    Get summary of all data stored in the database
    """
    try:
        summary = crypto_db.get_data_summary()
        if summary.empty:
            return JSONResponse(content={"message": "No data in database", "data": []})
        
        # Convert datetime columns to string for JSON serialization
        summary['first_date'] = summary['first_date'].astype(str)
        summary['last_date'] = summary['last_date'].astype(str)
        summary['last_update'] = summary['last_update'].astype(str)
        
        return JSONResponse(content={
            "message": "Database summary retrieved",
            "total_records": len(summary),
            "data": summary.to_dict(orient='records')
        })
    except Exception as e:
        app_logger.error(f"Error getting database summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/database/tickers")
async def get_available_tickers(
    source: str = Query(None, description="Filter by source: 'yfinance' or 'binance'")
):
    """
    Get all available tickers in the database
    """
    try:
        tickers = crypto_db.get_all_tickers(source)
        return JSONResponse(content={
            "message": "Available tickers retrieved",
            "source": source or "all",
            "count": len(tickers),
            "tickers": tickers
        })
    except Exception as e:
        app_logger.error(f"Error getting available tickers: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/download/unified/{ticker}")
async def download_unified_data(
    ticker: str,
    source: str = Query("both", description="Data source: 'yfinance', 'binance', or 'both'"),
    period: str = Query("90d", description="Time period (e.g. '90d' for 90 days)"),
    interval: str = Query("1d", description="Data interval (e.g. '1d' for daily)"),
    update_missing: bool = Query(True, description="Only update missing data")
):
    """
    Download and store cryptocurrency data using unified handler
    Automatically handles incremental updates and data structure unification
    """
    try:
        results = {}
        
        if source in ['yfinance', 'both']:
            app_logger.info(f"Downloading {ticker} from yfinance")
            df = unified_handler.download_and_store_data(
                ticker, 'yfinance', period, interval, update_missing
            )
            if not df.empty:
                results['yfinance'] = {
                    'records': len(df),
                    'first_date': str(df.index.min()),
                    'last_date': str(df.index.max())
                }
        
        if source in ['binance', 'both']:
            app_logger.info(f"Downloading {ticker} from binance")
            df = unified_handler.download_and_store_data(
                ticker, 'binance', period, interval, update_missing
            )
            if not df.empty:
                results['binance'] = {
                    'records': len(df),
                    'first_date': str(df.index.min()),
                    'last_date': str(df.index.max())
                }
        
        if not results:
            raise HTTPException(status_code=404, detail="No data downloaded")
        
        return JSONResponse(content={
            "message": f"Data downloaded and stored for {ticker}",
            "ticker": ticker,
            "results": results
        })
    except Exception as e:
        app_logger.error(f"Error downloading unified data for {ticker}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/download/top30")
async def download_top_30_unified(
    source: str = Query("both", description="Data source: 'yfinance', 'binance', or 'both'"),
    period: str = Query("90d", description="Time period"),
    interval: str = Query("1d", description="Data interval")
):
    """
    Download data for top 30 cryptocurrencies and store in database
    """
    try:
        results = {}
        
        if source in ['yfinance', 'both']:
            app_logger.info("Downloading top 30 from yfinance")
            yf_data = unified_handler.download_top_30_cryptos('yfinance', period, interval)
            results['yfinance'] = {
                'count': len(yf_data),
                'tickers': list(yf_data.keys())
            }
        
        if source in ['binance', 'both']:
            app_logger.info("Downloading top 30 from binance")
            bn_data = unified_handler.download_top_30_cryptos('binance', period, interval)
            results['binance'] = {
                'count': len(bn_data),
                'tickers': list(bn_data.keys())
            }
        
        return JSONResponse(content={
            "message": "Top 30 cryptocurrencies downloaded",
            "results": results
        })
    except Exception as e:
        app_logger.error(f"Error downloading top 30: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/data/{ticker}")
async def get_ticker_data(
    ticker: str,
    source: str = Query(None, description="Filter by source"),
    start_date: str = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: str = Query(None, description="End date (YYYY-MM-DD)"),
    interval: str = Query("1d", description="Data interval")
):
    """
    Retrieve ticker data from database
    """
    try:
        start_dt = pd.to_datetime(start_date) if start_date else None
        end_dt = pd.to_datetime(end_date) if end_date else None
        
        standard_ticker = unified_handler.denormalize_ticker(ticker)
        df = crypto_db.get_ohlcv_data(standard_ticker, source, start_dt, end_dt, interval)
        
        if df.empty:
            raise HTTPException(status_code=404, detail=f"No data found for {ticker}")
        
        # Convert to JSON-serializable format
        df_dict = df.reset_index().to_dict(orient='records')
        for record in df_dict:
            record['timestamp'] = str(record['timestamp'])
        
        return JSONResponse(content={
            "message": f"Data retrieved for {ticker}",
            "ticker": ticker,
            "records": len(df),
            "data": df_dict
        })
    except Exception as e:
        app_logger.error(f"Error retrieving data for {ticker}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

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
    datasource: str = Query("both", description="'yfinance', 'binance', or 'both'")
):
    """
    Download and flatten data for top 30 tickers from the selected datasource.
    Uses the unified handler to ensure consistent data structure and database storage.
    """
    try:
        # Use unified handler for downloading
        results = unified_handler.download_top_30_cryptos(datasource, period, interval)
        
        if not results:
            raise HTTPException(500, "No tickers could be downloaded.")
        
        # Prepare flattened data for backward compatibility
        flattened_data = {}
        for ticker, df in results.items():
            if not df.empty:
                flat = unified_handler.flatten_ticker_data(df)
                flattened_data[ticker] = flat

        if len(flattened_data) == 0:
            raise HTTPException(500, "No tickers could be flattened.")
        
        # Combine & save for backward compatibility
        combined_df = pd.DataFrame.from_dict(flattened_data, orient="index")
        data_dir = os.path.join(BASE_DIR, "data")
        os.makedirs(data_dir, exist_ok=True)
        
        # Save with appropriate filename
        if datasource == "both":
            filename = f"unified_{interval}_{period}.csv"
        else:
            filename = f"{datasource}_{interval}_{period}.csv"
        
        combined_df.to_csv(os.path.join(data_dir, filename))
        app_logger.info(f"Saved {len(flattened_data)} tickers to {filename}")

        return JSONResponse({
            "message": "Data downloaded, stored in database, and saved to CSV",
            "file": filename,
            "tickers_count": len(flattened_data),
            "tickers": list(flattened_data.keys())
        })

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


@app.get("/preprocess_binance_data")
async def preprocess_binance_data_api(
    file_path: str = Query("data/binance_1d_90d.csv", description="Path to the raw Binance CSV file"),
    max_days: int = Query(90, description="Maximum number of days to use")
):
    """
    Executes the data preprocessing function on Binance CSV data,
    stores the preprocessed data in a new CSV file, and returns a report
    detailing what values were updated and why.
    """
    try:
        output_file = "data/preprocessed_binance_1d_90d.csv"
        df_scaled, report = preprocess_data(file_path, max_days, output_file=output_file)
        app_logger.info("Binance preprocessing completed; returning report")
        return JSONResponse(content={"message": "Binance data preprocessing completed", "report": report})
    except Exception as e:
        app_logger.error(f"Error during Binance preprocessing: {e}")
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
    selected_tickers: str = Query("BTC,ETH,ADA,DOT", description="Comma-separated list of selected crypto tickers"),
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
        report, charts = perform_eda_analysis(ticker, preprocessed_file, source='yfinance')
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


@app.get("/eda/{ticker}")
async def get_eda_results(
    ticker: str,
    source: str = Query("yfinance", description="Data source: yfinance or binance")
):
    """
    Get comprehensive EDA analysis results for a ticker with chart data
    """
    try:
        # Normalize ticker for the specific source
        normalized_ticker = unified_handler.normalize_ticker(ticker, source)
        app_logger.info(f"[main.get_eda_results] Getting EDA for ticker: {ticker} -> {normalized_ticker} from {source}")
        
        # First, try to get raw ticker data for chart generation
        ticker_data = None
        try:
            data_response = crypto_db.get_ohlcv_data(normalized_ticker, source=source)
            if data_response.empty:
                # Try to download if not available
                app_logger.info(f"[main.get_eda_results] No data found for {normalized_ticker}, attempting to download...")
                downloaded_data = unified_handler.download_and_store_data(normalized_ticker, source, "90d", "1d")
                if not downloaded_data.empty:
                    data_response = downloaded_data
                    app_logger.info(f"[main.get_eda_results] Successfully downloaded data for {normalized_ticker}")
            
            if not data_response.empty:
                # Convert timestamps to strings to avoid JSON serialization issues
                data_df = data_response.reset_index()
                if 'timestamp' in data_df.columns:
                    data_df['timestamp'] = data_df['timestamp'].dt.strftime('%Y-%m-%d')
                ticker_data = data_df.to_dict('records')
        except Exception as data_error:
            app_logger.warning(f"[main.get_eda_results] Could not fetch ticker data for {normalized_ticker}: {data_error}")
        
        # Get stored EDA results from analysis storage
        stored_results = analysis_storage.get_eda_results(ticker)
        
        # If no stored results, try to run EDA analysis
        if not stored_results:
            app_logger.info(f"No stored EDA results for {ticker}, attempting to run analysis", "main.get_eda_results")
            
            try:
                # Import EDA function
                from .eda_analysis import perform_eda_analysis
                
                app_logger.info(f"Running EDA analysis for normalized ticker: {normalized_ticker}", "main.get_eda_results")
                
                # Run EDA analysis using database with normalized ticker
                report, charts = perform_eda_analysis(
                    ticker=normalized_ticker,
                    use_database=True,
                    source=source
                )
                
                app_logger.info(f"EDA analysis completed. Report keys: {list(report.keys())}, Charts: {len(charts)}", "main.get_eda_results")
                
                # Convert charts to the format expected by frontend
                charts_data = {}
                for chart_name, chart_path in charts.items():
                    chart_type = chart_name.replace('_json', '').replace('_html', '').replace('_chart', '')
                    if chart_type not in charts_data:
                        charts_data[chart_type] = {}
                    
                    if chart_name.endswith('_json'):
                        try:
                            with open(chart_path, 'r', encoding='utf-8') as f:
                                chart_json = f.read()
                            charts_data[chart_type]['data'] = chart_json
                        except Exception as e:
                            app_logger.warning(f"Could not read chart JSON {chart_path}: {e}", "main.get_eda_results")
                    elif chart_name.endswith('_html'):
                        try:
                            with open(chart_path, 'r', encoding='utf-8') as f:
                                chart_html = f.read()
                            charts_data[chart_type]['html'] = chart_html
                        except Exception as e:
                            app_logger.warning(f"Could not read chart HTML {chart_path}: {e}", "main.get_eda_results")
                
                # Try to get the newly stored results from database
                stored_results = analysis_storage.get_eda_results(ticker)
                if stored_results:
                    app_logger.info(f"Found {len(stored_results)} stored EDA results after analysis", "main.get_eda_results")
                
            except Exception as eda_error:
                app_logger.error(f"Error running EDA analysis for {ticker}: {eda_error}", "main.get_eda_results")
                app_logger.exception("Full EDA error traceback:", "main.get_eda_results")
                # Continue with empty results but provide ticker data if available
        
        # Process stored results into frontend-friendly format
        if 'charts_data' not in locals():
            charts_data = {}
        report_data = {}
        
        if stored_results:
            app_logger.info(f"Processing {len(stored_results)} stored EDA results", "main.get_eda_results")
            for result in stored_results:
                chart_type = result.get('chart_type', 'unknown')
                analysis_type = result.get('analysis_type', 'unknown')
                
                # Store chart data
                if chart_type != 'unknown':
                    charts_data[chart_type] = {
                        'data': result.get('data', {}),
                        'html': result.get('chart_html'),
                        'analysis_type': analysis_type,
                        'created_at': result.get('created_at')
                    }
                
                # Accumulate report data
                if isinstance(result.get('data'), dict):
                    report_data.update(result['data'])
        else:
            app_logger.warning(f"No stored results found for {ticker}", "main.get_eda_results")
        
        # Generate comprehensive response
        app_logger.info(f"Generating response for {ticker}: charts_count={len(charts_data)}, data_records={len(ticker_data) if ticker_data else 0}", "main.get_eda_results")
        
        response_data = {
            "ticker": ticker,
            "source": source,
            "normalized_ticker": normalized_ticker,
            "success": True,
            "data_available": ticker_data is not None,
            "charts_count": len(charts_data),
            "report": report_data,
            "charts": charts_data,
            "raw_data": ticker_data if ticker_data else [],  # Include all data for frontend
            "available_chart_types": [
                "temporal_line", "histograms", "box_plots", 
                "rolling_average", "candlestick", "rolling_volatility"
            ],
            "statistics": {
                "total_records": len(ticker_data) if ticker_data else 0,
                "date_range": {
                    "start": str(ticker_data[0]['timestamp']) if ticker_data else None,
                    "end": str(ticker_data[-1]['timestamp']) if ticker_data else None
                } if ticker_data else None
            },
            "debug_info": {
                "stored_results_count": len(stored_results) if stored_results else 0,
                "has_raw_data": ticker_data is not None,
                "charts_available": list(charts_data.keys()) if charts_data else []
            }
        }
        
        if stored_results:
            response_data["message"] = f"EDA data loaded for {ticker}"
        else:
            response_data["message"] = f"No stored EDA analysis found for {ticker}"
            response_data["suggestion"] = "Run the full pipeline to generate comprehensive EDA analysis"
        
        return JSONResponse(content=response_data)
        
    except Exception as e:
        app_logger.error(f"Error retrieving EDA results for {ticker}: {e}")
        return JSONResponse(
            status_code=500, 
            content={
                "ticker": ticker,
                "success": False,
                "error": str(e),
                "message": "Failed to retrieve EDA data"
            }
        )


@app.get("/database/ticker-data/{ticker}")
async def get_ticker_data_for_charts(
    ticker: str,
    source: str = Query("yfinance", description="Data source: yfinance or binance"),
    limit: int = Query(90, description="Number of recent records to return")
):
    """
    Get raw ticker data for chart generation in frontend
    """
    try:
        # Normalize ticker for the specific source
        normalized_ticker = unified_handler.normalize_ticker(ticker, source)
        app_logger.info(f"Getting data for ticker: {ticker} -> {normalized_ticker} from {source}")
        
        # Get OHLCV data from database
        data = crypto_db.get_ohlcv_data(normalized_ticker, source=source)
        
        if data.empty:
            app_logger.info(f"No data found for {normalized_ticker}, attempting to download...")
            # Try to download the data if not available
            downloaded_data = unified_handler.download_and_store_data(
                ticker=normalized_ticker,
                source=source,
                period="90d",
                interval="1d"
            )
            
            if not downloaded_data.empty:
                data = downloaded_data
                app_logger.info(f"Successfully downloaded data for {normalized_ticker}")
            else:
                app_logger.warning(f"Failed to download data for {normalized_ticker}")
                return JSONResponse(content={
                    "ticker": ticker,
                    "source": source,
                    "data": [],
                    "count": 0,
                    "message": f"No data available for {ticker} from {source}. Attempted to download but failed."
                })
        
        # Limit to recent records and convert to records format
        recent_data = data.tail(limit).reset_index()
        records = []
        
        for _, row in recent_data.iterrows():
            records.append({
                "timestamp": row['timestamp'].isoformat() if hasattr(row['timestamp'], 'isoformat') else str(row['timestamp']),
                "open": float(row['Open']),
                "high": float(row['High']),
                "low": float(row['Low']),
                "close": float(row['Close']),
                "volume": float(row['Volume'])
            })
        
        return JSONResponse(content={
            "ticker": ticker,
            "source": source,
            "data": records,
            "count": len(records),
            "date_range": {
                "start": records[0]['timestamp'] if records else None,
                "end": records[-1]['timestamp'] if records else None
            }
        })
        
    except Exception as e:
        app_logger.error(f"Error retrieving ticker data for {ticker}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/eda/charts/{ticker}")
async def get_eda_chart_data(
    ticker: str,
    chart_type: str = Query(..., description="Chart type: candlestick, volume, moving_averages, volatility"),
    source: str = Query("yfinance", description="Data source: yfinance or binance"),
    period: int = Query(90, description="Number of days to include")
):
    """
    Get specific chart data for EDA visualization
    """
    try:
        # Get raw ticker data
        data = crypto_db.get_ohlcv_data(ticker, source=source)
        
        if data.empty:
            return JSONResponse(content={
                "ticker": ticker,
                "chart_type": chart_type,
                "error": "No data available",
                "data": []
            })
        
        # Limit to requested period
        recent_data = data.tail(period).reset_index()
        
        chart_data = []
        chart_config = {}
        
        if chart_type == "candlestick":
            for _, row in recent_data.iterrows():
                chart_data.append({
                    "x": row['timestamp'].isoformat(),
                    "y": [float(row['Open']), float(row['High']), float(row['Low']), float(row['Close'])]
                })
            chart_config = {
                "type": "candlestick",
                "title": f"{ticker} Price Action",
                "yAxis": {"title": "Price ($)"}
            }
            
        elif chart_type == "volume":
            for _, row in recent_data.iterrows():
                chart_data.append({
                    "x": row['timestamp'].isoformat(),
                    "y": float(row['Volume'])
                })
            chart_config = {
                "type": "bar",
                "title": f"{ticker} Trading Volume",
                "yAxis": {"title": "Volume"}
            }
            
        elif chart_type == "moving_averages":
            # Calculate moving averages
            recent_data['MA7'] = recent_data['Close'].rolling(window=7).mean()
            recent_data['MA20'] = recent_data['Close'].rolling(window=20).mean()
            recent_data['MA50'] = recent_data['Close'].rolling(window=50).mean()
            
            ma_data = {
                "price": [],
                "ma7": [],
                "ma20": [],
                "ma50": []
            }
            
            for _, row in recent_data.iterrows():
                timestamp = row['timestamp'].isoformat()
                ma_data["price"].append({"x": timestamp, "y": float(row['Close'])})
                if not pd.isna(row['MA7']):
                    ma_data["ma7"].append({"x": timestamp, "y": float(row['MA7'])})
                if not pd.isna(row['MA20']):
                    ma_data["ma20"].append({"x": timestamp, "y": float(row['MA20'])})
                if not pd.isna(row['MA50']):
                    ma_data["ma50"].append({"x": timestamp, "y": float(row['MA50'])})
            
            chart_data = ma_data
            chart_config = {
                "type": "line",
                "title": f"{ticker} Moving Averages",
                "yAxis": {"title": "Price ($)"},
                "series": ["Price", "MA7", "MA20", "MA50"]
            }
            
        elif chart_type == "volatility":
            # Calculate rolling volatility
            returns = recent_data['Close'].pct_change()
            volatility = returns.rolling(window=20).std() * (252 ** 0.5) * 100  # Annualized
            
            for i, (_, row) in enumerate(recent_data.iterrows()):
                if not pd.isna(volatility.iloc[i]):
                    chart_data.append({
                        "x": row['timestamp'].isoformat(),
                        "y": float(volatility.iloc[i])
                    })
            
            chart_config = {
                "type": "area",
                "title": f"{ticker} Price Volatility (20-day)",
                "yAxis": {"title": "Volatility (%)"}
            }
        
        return JSONResponse(content={
            "ticker": ticker,
            "chart_type": chart_type,
            "source": source,
            "data": chart_data,
            "config": chart_config,
            "count": len(chart_data) if isinstance(chart_data, list) else sum(len(v) for v in chart_data.values()),
            "period": period
        })
        
    except Exception as e:
        app_logger.error(f"Error generating {chart_type} chart for {ticker}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/train/{ticker}")
async def train_models(
    ticker: str,
    feature: str = Query("Close", description="Feature column"),
    test_size: float = Query(0.2, description="Test split fraction"),
    source: str = Query(None, description="Data source: 'yfinance' or 'binance'")
):
    """Train and compare all models using the unified model comparison framework"""
    try:
        # Use the model comparison framework for comprehensive training
        results = model_comparison.train_all_models(ticker, feature, test_size, source)
        
        if 'error' in results:
            raise HTTPException(status_code=500, detail=results['error'])
        
        return JSONResponse(content={
            "message": f"Models trained for {ticker}",
            "results": results
        })
    except Exception as e:
        app_logger.error(f"Error training models for {ticker}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models/comparison/{ticker}")
async def get_model_comparison(ticker: str):
    """Get comprehensive model comparison report for a ticker"""
    try:
        report = model_comparison.get_model_comparison_report(ticker)
        
        if 'error' in report:
            raise HTTPException(status_code=500, detail=report['error'])
        
        return JSONResponse(content=report)
    except Exception as e:
        app_logger.error(f"Error getting model comparison for {ticker}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/models/cross-validate/{ticker}")
async def cross_validate_models(
    ticker: str,
    feature: str = Query("Close", description="Feature to predict"),
    n_splits: int = Query(5, description="Number of cross-validation splits")
):
    """Perform time series cross-validation for all models"""
    try:
        results = model_comparison.cross_validate_models(ticker, feature, n_splits)
        
        if 'error' in results:
            raise HTTPException(status_code=500, detail=results['error'])
        
        return JSONResponse(content={
            "message": f"Cross-validation completed for {ticker}",
            "results": results
        })
    except Exception as e:
        app_logger.error(f"Error in cross-validation for {ticker}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/forecast/{ticker}")
async def get_forecast(
    ticker: str,
    model_type: str = Query("arima", description="arima|sarima|rf|xgb|lstm"),
    periods: int = Query(7, description="Number of periods to forecast"),
    source: str = Query("yfinance", description="Data source: yfinance or binance")
):
    """Return forecast for next `periods` days starting from tomorrow"""
    try:
        # Get data from database
        df_ticker = unified_handler.get_data_from_db(ticker, source=source)
        if df_ticker is None or df_ticker.empty:
            # Try to download the data if not available
            app_logger.info(f"Data not found for {ticker}, attempting to download...")
            await unified_handler.download_and_store_data(
                ticker=ticker,
                source=source,
                period="90d",
                interval="1d"
            )
            df_ticker = unified_handler.get_data_from_db(ticker, source=source)
            if df_ticker is None or df_ticker.empty:
                raise HTTPException(404, f"No data available for {ticker}")
        
        last_series = df_ticker["Close"] if "Close" in df_ticker.columns else df_ticker["close"]
        
        # Load or train model
        try:
            pkl = load_model(ticker, model_type)
        except Exception as e:
            app_logger.warning(f"Model not found for {ticker} {model_type}: {e}")
            from .model_comparison import model_comparison
            app_logger.info(f"Training {model_type} model for {ticker}...")
            model_comparison.train_all_models([ticker], feature="Close", test_size=0.2)
            pkl = load_model(ticker, model_type)

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

        # regenerate the index from tomorrow for `periods` days
        start_date = datetime.utcnow().date() + timedelta(days=1)
        new_idx = pd.date_range(start=start_date, periods=periods, freq="D")
        fc.index = new_idx

        fc_str = fc.copy()
        fc_str.index = fc.index.astype(str)
        forecast_payload = fc_str.to_dict(orient="index")

        return JSONResponse({
            "ticker": ticker,
            "model": model_type,
            "forecast": forecast_payload
        })
    except HTTPException:
        raise
    except Exception as e:
        app_logger.error(f"Error generating forecast for {ticker}: {e}")
        raise HTTPException(500, str(e))




@app.get("/signals/{ticker}")
async def get_signals(
    ticker: str,
    model_type: str = Query("arima", description="arima|sarima|rf|xgb|lstm"),
    periods: int = Query(7, description="Forecast horizon"),
    threshold: float = Query(0.01, description="Threshold for signal generation"),
    source: str = Query("yfinance", description="Data source: yfinance or binance")
):
    """Generate buy/sell signals and PnL for the next `periods` days starting tomorrow"""
    try:
        # Check if signals already exist in database
        from .signals import check_existing_signals
        existing_signals = check_existing_signals(ticker, model_type)
        if existing_signals:
            app_logger.info(f"Using existing signals for {ticker} from database")
            # Convert to required format
            signals_out = {date: data['signal'] for date, data in existing_signals.items()}
            pnl_out = {date: data.get('expected_profit', 0) for date, data in existing_signals.items()}
            return JSONResponse({
                "signals": signals_out,
                "pnl": pnl_out,
                "from_cache": True
            })
        
        # 1) Get data from database instead of CSV
        df_ticker = unified_handler.get_data_from_db(ticker, source=source)
        if df_ticker is None or df_ticker.empty:
            # Try to download the data if not available
            app_logger.info(f"Data not found for {ticker}, attempting to download...")
            await unified_handler.download_and_store_data(
                ticker=ticker,
                source=source,
                period="90d",
                interval="1d"
            )
            # Try again after download
            df_ticker = unified_handler.get_data_from_db(ticker, source=source)
            if df_ticker is None or df_ticker.empty:
                raise HTTPException(404, f"No data available for {ticker}")
        
        last_series = df_ticker["Close"] if "Close" in df_ticker.columns else df_ticker["close"]
        
        # Check if model exists
        try:
            pkl = load_model(ticker, model_type)
        except Exception as e:
            app_logger.warning(f"Model not found for {ticker} {model_type}: {e}")
            # Train the model if it doesn't exist
            from .model_comparison import model_comparison
            app_logger.info(f"Training {model_type} model for {ticker}...")
            metrics = model_comparison.train_all_models([ticker], feature="Close", test_size=0.2)
            pkl = load_model(ticker, model_type)

        # 2) forecast
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

        # 3) override index to tomorrow forward
        start_date = datetime.utcnow().date() + timedelta(days=1)
        new_idx = pd.date_range(start=start_date, periods=periods, freq="D")
        fc_df.index = new_idx

        # 4) extract point forecasts (keep DatetimeIndex)
        if "forecast" in fc_df.columns:
            series_fc = fc_df["forecast"].copy()
        else:
            series_fc = fc_df.iloc[:, 0].copy()

        # 5) generate signals and PnL
        signals_df = generate_signals(series_fc, threshold)
        all_prices = pd.concat([last_series, series_fc])
        pnl_series = estimate_pnl(all_prices, signals_df["signal"])
        
        # Store signals to database
        from .signals import store_signals_to_db
        store_signals_to_db(ticker, signals_df, model_type, all_prices)

        # 6) stringify for JSON output
        signals_out = signals_df["signal"].copy()
        signals_out.index = signals_out.index.astype(str)
        pnl_out = pnl_series.copy().fillna(0)
        pnl_out.index = pnl_out.index.astype(str)

        return JSONResponse({
            "signals": signals_out.to_dict(),
            "pnl": pnl_out.to_dict(),
            "from_cache": False
        })
        
    except HTTPException:
        raise
    except Exception as e:
        app_logger.error(f"Error generating signals for {ticker}: {e}")
        raise HTTPException(500, str(e))

    

@app.get("/indicators/{ticker}")
async def get_indicators(
    ticker: str,
    window_rsi: int = Query(14),
    fast: int = Query(12),
    slow: int = Query(26),
    signal: int = Query(9),
    source: str = Query("yfinance", description="Data source: yfinance or binance")
):
    """Retrieve RSI and MACD for a ticker"""
    try:
        # Get data from database instead of CSV
        df_ticker = unified_handler.get_data_from_db(ticker, source)
        if df_ticker is None or df_ticker.empty:
            raise HTTPException(404, f"No data found for {ticker} in database")
        
        prices = df_ticker["Close"] if "Close" in df_ticker.columns else df_ticker["close"]

        # Compute RSI and stringify index
        rsi_series = compute_rsi(prices, window_rsi).dropna()
        rsi_series.index = rsi_series.index.astype(str)
        rsi_dict = rsi_series.to_dict()

        # Compute MACD and stringify index
        macd_df = compute_macd(prices, fast, slow, signal).dropna()
        macd_df.index = macd_df.index.astype(str)
        macd_dict = macd_df.to_dict(orient='index')
        
        # Store indicators in database
        from .signals import store_indicators_to_db
        store_indicators_to_db(ticker, rsi_series, macd_df)

        return JSONResponse(content={
            'rsi': rsi_dict,
            'macd': macd_dict
        })
    except Exception as e:
        app_logger.error(f"Error calculating indicators for {ticker}: {e}")
        raise HTTPException(500, str(e))

@app.get("/forecast_outputs/{ticker}")
async def forecast_outputs(
    ticker: str,
    model_type: str = Query('arima', description='Model type'),
    short_days: int = Query(1),
    short_weeks: int = Query(7),
    medium_month: int = Query(30),
    medium_quarter: int = Query(90),
    source: str = Query("yfinance", description="Data source: yfinance or binance")
):
    "Returns multiple horizons with confidence intervals and past accuracy"
    try:
        # Get data from database
        df = unified_handler.get_data_from_db(ticker, source=source)
        if df is None or df.empty:
            # Try to download the data if not available
            app_logger.info(f"Data not found for {ticker}, attempting to download...")
            await unified_handler.download_and_store_data(
                ticker=ticker,
                source=source,
                period="90d",
                interval="1d"
            )
            df = unified_handler.get_data_from_db(ticker, source=source)
            if df is None or df.empty:
                raise HTTPException(404, f"No data available for {ticker}")
        
        series = df["Close"] if "Close" in df.columns else df["close"]
        
        # Load or train model
        try:
            pkl = load_model(ticker, model_type)
        except Exception as e:
            app_logger.warning(f"Model not found for {ticker} {model_type}: {e}")
            from .model_comparison import model_comparison
            app_logger.info(f"Training {model_type} model for {ticker}...")
            model_comparison.train_all_models([ticker], feature="Close", test_size=0.2)
            pkl = load_model(ticker, model_type)
        
        # load and forecast for each horizon
        horizons = {
            'short_day': short_days,
            'short_week': short_weeks,
            'medium_month': medium_month,
            'medium_quarter': medium_quarter
        }
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
    except HTTPException:
        raise
    except Exception as e:
        app_logger.error(f"Error generating forecast outputs for {ticker}: {e}")
        raise HTTPException(500, str(e))

@app.get("/backtest/{ticker}")
async def backtest(
    ticker: str,
    model_type: str = Query('arima'),
    periods: int = Query(7),
    threshold: float = Query(0.01),
    source: str = Query("yfinance", description="Data source: yfinance or binance")
):
    "Run backtest with signals, indicators, and performance"
    try:
        # Get data from database
        df = unified_handler.get_data_from_db(ticker, source=source)
        if df is None or df.empty:
            # Try to download the data if not available
            app_logger.info(f"Data not found for {ticker}, attempting to download...")
            await unified_handler.download_and_store_data(
                ticker=ticker,
                source=source,
                period="90d",
                interval="1d"
            )
            df = unified_handler.get_data_from_db(ticker, source=source)
            if df is None or df.empty:
                raise HTTPException(404, f"No data available for {ticker}")
        
        prices = df["Close"] if "Close" in df.columns else df["close"]
        
        # Load or train model
        try:
            pkl = load_model(ticker, model_type)
        except Exception as e:
            app_logger.warning(f"Model not found for {ticker} {model_type}: {e}")
            from .model_comparison import model_comparison
            app_logger.info(f"Training {model_type} model for {ticker}...")
            model_comparison.train_all_models([ticker], feature="Close", test_size=0.2)
            pkl = load_model(ticker, model_type)
        
        forecast_df = globals()[f'forecast_{model_type}'](pkl, periods)
        result = backtest_ticker(prices, forecast_df['forecast'], threshold, ticker, model_type)
        return JSONResponse(content=result)
    except HTTPException:
        raise
    except Exception as e:
        app_logger.error(f"Error running backtest for {ticker}: {e}")
        raise HTTPException(500, str(e))


MODEL_DIR = "models"  # make sure this matches wherever you're saving .pkl files

# ===== INDIVIDUAL TICKER PREPROCESSING =====

@app.post("/preprocess/ticker/{ticker}")
async def preprocess_individual_ticker(
    ticker: str,
    source: str = Query("yfinance", description="Data source: 'yfinance' or 'binance'"),
    max_days: int = Query(90, description="Maximum days to use")
):
    """
    Preprocess data for an individual ticker
    """
    try:
        # Get data from database
        standard_ticker = unified_handler.denormalize_ticker(ticker)
        df = crypto_db.get_ohlcv_data(standard_ticker, source)
        
        if df.empty:
            # Try to download if not in database
            df = unified_handler.download_and_store_data(ticker, source, "90d", "1d")
            if df.empty:
                raise HTTPException(status_code=404, detail=f"No data available for {ticker}")
        
        # Flatten the data
        flattened = unified_handler.flatten_ticker_data(df)
        
        # Create temporary DataFrame for preprocessing
        temp_df = pd.DataFrame([flattened])
        temp_df.index = [ticker]
        
        # Save to temporary file
        temp_file = f"data/temp_{ticker}_{source}.csv"
        os.makedirs("data", exist_ok=True)
        temp_df.to_csv(temp_file)
        
        # Preprocess
        output_file = f"data/preprocessed_{ticker}_{source}.csv"
        df_scaled, report = preprocess_data(temp_file, max_days, output_file)
        
        # Clean up temp file
        if os.path.exists(temp_file):
            os.remove(temp_file)
        
        # Store in database
        crypto_db.save_preprocessed_data(standard_ticker, source, df_scaled, max_days)
        
        return JSONResponse(content={
            "message": f"Preprocessing completed for {ticker}",
            "ticker": ticker,
            "source": source,
            "report": report,
            "output_file": output_file
        })
        
    except Exception as e:
        app_logger.error(f"Error preprocessing {ticker}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ===== RSS FEED AND NEWS ENDPOINTS =====

@app.get("/news/feed")
async def get_news_feed(
    ticker: str = Query(None, description="Filter by ticker"),
    limit: int = Query(20, description="Number of articles to return"),
    refresh: bool = Query(False, description="Fetch fresh news")
):
    """
    Get cryptocurrency news from RSS feeds
    """
    try:
        if refresh:
            # Fetch fresh news
            if ticker:
                articles = rss_handler.fetch_ticker_news(ticker, limit)
            else:
                articles = rss_handler.fetch_all_feeds(limit_per_source=limit//8)
        else:
            # Get from database
            articles = analysis_storage.get_rss_feeds(ticker, limit)
            
            if not articles and ticker:
                # If no cached news, fetch fresh
                articles = rss_handler.fetch_ticker_news(ticker, limit)
        
        return JSONResponse(content={
            "ticker": ticker,
            "count": len(articles),
            "articles": articles
        })
        
    except Exception as e:
        app_logger.error(f"Error fetching news: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/news/sentiment")
async def get_market_sentiment(
    ticker: str = Query(None, description="Ticker for sentiment analysis")
):
    """
    Get market sentiment from news analysis
    """
    try:
        sentiment = rss_handler.get_market_sentiment(ticker)
        return JSONResponse(content=sentiment)
        
    except Exception as e:
        app_logger.error(f"Error analyzing sentiment: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ===== WHAT-IF SCENARIO ENDPOINTS =====

@app.post("/whatif/price-change")
async def analyze_price_change_scenario(
    ticker: str = Query(..., description="Cryptocurrency ticker"),
    current_price: float = Query(None, description="Current price (if None, uses latest)"),
    target_prices: str = Query(None, description="Comma-separated target prices"),
    quantities: str = Query(None, description="Comma-separated quantities")
):
    """
    Analyze what-if scenarios for price changes
    """
    try:
        # Parse parameters
        target_prices_list = [float(p) for p in target_prices.split(",")] if target_prices else None
        quantities_list = [float(q) for q in quantities.split(",")] if quantities else None
        
        result = whatif_analyzer.analyze_price_change(
            ticker, current_price, target_prices_list, quantities_list
        )
        
        return JSONResponse(content=result)
        
    except Exception as e:
        app_logger.error(f"Error in price change scenario: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/whatif/trading-strategy")
async def analyze_trading_strategy_scenario(
    ticker: str = Query(..., description="Cryptocurrency ticker"),
    investment_amount: float = Query(..., description="Amount to invest"),
    buy_price: float = Query(None, description="Entry price"),
    sell_price: float = Query(None, description="Exit price"),
    holding_period_days: int = Query(30, description="Holding period in days"),
    stop_loss_pct: float = Query(10, description="Stop loss percentage"),
    take_profit_pct: float = Query(20, description="Take profit percentage")
):
    """
    Analyze a specific trading strategy
    """
    try:
        result = whatif_analyzer.analyze_trading_strategy(
            ticker, investment_amount, buy_price, sell_price,
            holding_period_days, stop_loss_pct, take_profit_pct
        )
        
        return JSONResponse(content=result)
        
    except Exception as e:
        app_logger.error(f"Error in trading strategy scenario: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/whatif/portfolio")
async def analyze_portfolio_scenario(
    tickers: str = Query(..., description="Comma-separated ticker list"),
    total_investment: float = Query(..., description="Total investment amount"),
    allocations: str = Query(None, description="Comma-separated allocations (percentages)"),
    rebalance_period_days: int = Query(30, description="Rebalancing period")
):
    """
    Analyze portfolio allocation scenarios
    """
    try:
        ticker_list = [t.strip() for t in tickers.split(",")]
        
        if allocations:
            alloc_list = [float(a) for a in allocations.split(",")]
            alloc_dict = dict(zip(ticker_list, alloc_list))
        else:
            alloc_dict = None
        
        result = whatif_analyzer.analyze_portfolio_allocation(
            ticker_list, total_investment, alloc_dict, rebalance_period_days
        )
        
        return JSONResponse(content=result)
        
    except Exception as e:
        app_logger.error(f"Error in portfolio scenario: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/whatif/dca")
async def analyze_dca_scenario(
    ticker: str = Query(..., description="Cryptocurrency ticker"),
    periodic_investment: float = Query(..., description="Amount per period"),
    frequency_days: int = Query(7, description="Days between investments"),
    total_periods: int = Query(52, description="Number of periods")
):
    """
    Analyze Dollar Cost Averaging (DCA) strategy
    """
    try:
        result = whatif_analyzer.analyze_dca_strategy(
            ticker, periodic_investment, frequency_days, total_periods
        )
        
        return JSONResponse(content=result)
        
    except Exception as e:
        app_logger.error(f"Error in DCA scenario: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ===== STORED ANALYSIS RESULTS ENDPOINTS =====

@app.get("/analysis/eda/{ticker}")
async def get_stored_eda_results(
    ticker: str,
    analysis_type: str = Query(None, description="Filter by analysis type")
):
    """
    Get stored EDA analysis results for a ticker
    """
    try:
        results = analysis_storage.get_eda_results(ticker, analysis_type)
        
        return JSONResponse(content={
            "ticker": ticker,
            "count": len(results),
            "results": results
        })
        
    except Exception as e:
        app_logger.error(f"Error retrieving EDA results: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analysis/signals")
async def get_stored_trading_signals(
    ticker: str = Query(None, description="Filter by ticker"),
    start_date: str = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: str = Query(None, description="End date (YYYY-MM-DD)")
):
    """
    Get stored trading signals
    """
    try:
        signals = analysis_storage.get_trading_signals(ticker, start_date, end_date)
        
        return JSONResponse(content={
            "count": len(signals),
            "signals": signals
        })
        
    except Exception as e:
        app_logger.error(f"Error retrieving trading signals: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ===== PIPELINE ORCHESTRATION ENDPOINTS =====

@app.post("/pipeline/full")
async def run_full_pipeline(
    tickers: str = Query("TOP30", description="Comma-separated tickers or 'TOP30'"),
    sources: str = Query("yfinance,binance", description="Comma-separated sources"),
    period: str = Query("90d", description="Data period"),
    interval: str = Query("1d", description="Data interval"),
    max_days: int = Query(90, description="Max days for preprocessing"),
    feature: str = Query("Close", description="Feature for model training"),
    test_size: float = Query(0.2, description="Test split fraction"),
    include_eda: bool = Query(True, description="Include EDA analysis"),
    include_clustering: bool = Query(True, description="Include clustering analysis")
):
    """
    Execute the complete cryptocurrency analysis pipeline:
    1. Download data from both YFinance and Binance
    2. Preprocess data for both sources
    3. Perform EDA analysis
    4. Run dimensionality reduction and clustering
    5. Train and compare all models
    
    This is the main endpoint that fulfills all COM724 assessment requirements.
    """
    try:
        app_logger.info("Starting full cryptocurrency analysis pipeline", "main.run_full_pipeline")
        
        # Parse parameters
        ticker_list = ["TOP30"] if tickers == "TOP30" else [t.strip() for t in tickers.split(",")]
        source_list = [s.strip() for s in sources.split(",")]
        
        app_logger.info(f"Parsed parameters: tickers={ticker_list}, sources={source_list}", "main.run_full_pipeline")
        app_logger.info(f"Pipeline settings: period={period}, interval={interval}, max_days={max_days}", "main.run_full_pipeline")
        app_logger.info(f"Model settings: feature={feature}, test_size={test_size}", "main.run_full_pipeline")
        app_logger.info(f"Analysis flags: include_eda={include_eda}, include_clustering={include_clustering}", "main.run_full_pipeline")
        
        # Create and execute pipeline
        app_logger.info("About to create pipeline using factory", "main.run_full_pipeline")
        pipeline = pipeline_factory.create_full_pipeline(
            tickers=ticker_list,
            sources=source_list,
            period=period,
            interval=interval,
            max_days=max_days,
            feature=feature,
            test_size=test_size,
            include_eda=include_eda,
            include_clustering=include_clustering
        )
        
        if pipeline is None:
            app_logger.error("Pipeline factory returned None! This will cause issues.", "main.run_full_pipeline")
            raise HTTPException(status_code=500, detail="Pipeline creation failed - factory returned None")
        
        app_logger.info(f"Pipeline created successfully with {len(pipeline.steps)} steps", "main.run_full_pipeline")
        
        # Debug pipeline step information
        step_names = [step.name for step in pipeline.steps] if hasattr(pipeline, 'steps') else []
        app_logger.info(f"Pipeline steps: {step_names}", "main.run_full_pipeline")
        
        if len(pipeline.steps) == 0:
            app_logger.error("Pipeline created with 0 steps - this will cause division by zero!", "main.run_full_pipeline")
            app_logger.error(f"Pipeline creation debug - tickers: {ticker_list}, sources: {source_list}", "main.run_full_pipeline")
            raise HTTPException(status_code=500, detail="Pipeline creation failed - no steps were added")
        
        app_logger.info("About to execute pipeline", "main.run_full_pipeline")
        results = await pipeline.execute_pipeline()
        
        # Check if pipeline was rejected due to another running
        if 'error' in results and results.get('status') == 'rejected':
            raise HTTPException(status_code=409, detail=results['error'])
        
        return JSONResponse(content={
            "message": "Full cryptocurrency analysis pipeline completed",
            "pipeline_results": results
        })
        
    except Exception as e:
        app_logger.error(f"Error in full pipeline execution: {str(e)}", "main.run_full_pipeline")
        app_logger.exception("Full exception traceback:", "main.run_full_pipeline")
        
        # Special handling for division by zero
        if "division by zero" in str(e).lower():
            app_logger.error("Division by zero detected in pipeline execution!", "main.run_full_pipeline")
            error_msg = f"Pipeline execution failed due to division by zero: {str(e)}"
        else:
            error_msg = f"Pipeline execution failed: {str(e)}"
        
        raise HTTPException(status_code=500, detail=error_msg)

@app.post("/pipeline/download-and-train")
async def run_download_and_train_pipeline(
    tickers: str = Query(..., description="Comma-separated ticker list (e.g., 'BTC,ETH,ADA')"),
    sources: str = Query("yfinance,binance", description="Comma-separated sources"),
    period: str = Query("90d", description="Data period"),
    interval: str = Query("1d", description="Data interval"),
    feature: str = Query("Close", description="Feature for model training"),
    test_size: float = Query(0.2, description="Test split fraction")
):
    """
    Execute download and model training pipeline for specific tickers
    """
    try:
        app_logger.info(f"Starting download and train pipeline for {tickers}")
        
        # Parse parameters
        ticker_list = [t.strip() for t in tickers.split(",")]
        source_list = [s.strip() for s in sources.split(",")]
        
        # Create and execute pipeline
        pipeline = pipeline_factory.create_download_and_train_pipeline(
            tickers=ticker_list,
            sources=source_list,
            period=period,
            interval=interval,
            feature=feature,
            test_size=test_size
        )
        
        results = await pipeline.execute_pipeline()
        
        return JSONResponse(content={
            "message": f"Download and train pipeline completed for {len(ticker_list)} tickers",
            "pipeline_results": results
        })
        
    except Exception as e:
        app_logger.error(f"Error in download-and-train pipeline: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/pipeline/preprocessing")
async def run_preprocessing_pipeline(
    sources: str = Query("yfinance,binance", description="Comma-separated sources"),
    max_days: int = Query(90, description="Maximum days to use for preprocessing")
):
    """
    Execute preprocessing pipeline for existing data in database
    """
    try:
        app_logger.info("Starting preprocessing pipeline")
        
        # Parse parameters
        source_list = [s.strip() for s in sources.split(",")]
        
        # Create and execute pipeline
        pipeline = pipeline_factory.create_preprocessing_pipeline(
            sources=source_list,
            max_days=max_days
        )
        
        results = await pipeline.execute_pipeline()
        
        return JSONResponse(content={
            "message": "Preprocessing pipeline completed",
            "pipeline_results": results
        })
        
    except Exception as e:
        app_logger.error(f"Error in preprocessing pipeline: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/pipeline/status")
async def get_pipeline_status():
    """
    Get current pipeline capabilities and system status
    """
    try:
        # Check database status
        summary = crypto_db.get_data_summary()
        
        # Check available tickers
        yf_tickers = crypto_db.get_all_tickers('yfinance')
        bn_tickers = crypto_db.get_all_tickers('binance')
        
        # Check available models
        model_files = []
        if os.path.exists(MODEL_DIR):
            model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith('.pkl')]
        
        status = {
            "system_status": "operational",
            "database": {
                "total_records": len(summary),
                "yfinance_tickers": len(yf_tickers),
                "binance_tickers": len(bn_tickers),
                "sample_tickers": (yf_tickers + bn_tickers)[:10]
            },
            "models": {
                "trained_models": len(model_files),
                "model_types": ["arima", "sarima", "random_forest", "xgboost"],
                "sample_models": model_files[:10]
            },
            "pipeline_capabilities": {
                "full_pipeline": "Download → Preprocess → EDA → Clustering → Model Training",
                "download_and_train": "Download → Model Training",
                "preprocessing_only": "Preprocess existing data",
                "supported_sources": ["yfinance", "binance"],
                "supported_intervals": ["1m", "5m", "15m", "30m", "1h", "1d"],
                "model_types": ["ARIMA", "SARIMA", "Random Forest", "XGBoost"]
            }
        }
        
        return JSONResponse(content=status)
        
    except Exception as e:
        app_logger.error(f"Error getting pipeline status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ===== END PIPELINE ENDPOINTS =====

@app.get("/model_status/{ticker}")
async def model_status(ticker: str):
    """
    Check which models already exist for the given ticker.
    Returns a JSON object like:
      {
        "arima": true,
        "sarima": false,
        "rf": true,
        "xgb": false,
        "lstm": true
      }
    """
    types = ["arima", "sarima", "rf", "xgb", "lstm"]
    status = {
        t: os.path.exists(os.path.join(MODEL_DIR, f"{ticker}_{t}.pkl"))
        for t in types
    }
    return JSONResponse(content=status)