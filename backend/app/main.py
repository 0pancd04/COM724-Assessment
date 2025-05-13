import os
import logging
from logging.handlers import RotatingFileHandler
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from fastapi.responses import JSONResponse
import pandas as pd
import plotly.io as pio

from .logger import setup_logger
from .data_preprocessing import preprocess_data
from .data_downloader import download_data_yfinance, get_top_30_coins, flatten_ticker_data
from .download_binance_data import download_binance_ohlcv
from .grouping_analysis import perform_dimensionality_reduction, perform_clustering_analysis
from .correlation_analysis import perform_correlation_analysis
from .eda_analysis import perform_eda_analysis

LOG_DIR = "logs"

app_logger = setup_logger("app_logger", "app.log")

# Optionally, capture uvicorn server logs as well
uvicorn_logger = logging.getLogger("uvicorn")
uvicorn_handler = RotatingFileHandler(os.path.join(LOG_DIR, "uvicorn.log"), maxBytes=5 * 1024 * 1024, backupCount=3)
uvicorn_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
uvicorn_handler.setFormatter(uvicorn_formatter)
uvicorn_logger.addHandler(uvicorn_handler)

# --- FastAPI Application ---
app = FastAPI()

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
    period: str = Query("5y", description="Time period (e.g. '5y' for 5 years)"),
    interval: str = Query("1d", description="Data interval (e.g. '1d' for daily)"),
    datasource: str = Query("yfinance", description="yfinance or binance")
):
    """
    Download data for the top 30 crypto tickers, flatten each ticker's data into a single row,
    and store the complete dataset in a CSV file under the 'data' folder.
    
    The CSV file name is constructed as: yfinance_<interval>_<period>.csv
    """
    tickers = get_top_30_coins()
    if not tickers:
        raise HTTPException(status_code=404, detail="Could not retrieve top tickers.")
    
    flattened_data = {}
    for ticker in tickers:
        try:
            if datasource == "yfinance":
                df = download_data_yfinance(ticker, period=period, interval=interval)
                if not df.empty:
                    # Flatten the data into a single row
                    flat_series = flatten_ticker_data(df)
                    flattened_data[ticker] = flat_series
                    app_logger.info(f"Downloaded and flattened data for {ticker}")
                else:
                    app_logger.warning(f"No data found for {ticker}")
            elif datasource == "binance":
                df = download_binance_ohlcv(ticker, period, interval=interval)
        except Exception as e:
            app_logger.error(f"Error processing data for {ticker}: {e}")
    
    if not flattened_data:
        raise HTTPException(status_code=500, detail="No data was downloaded.")
    
    # Combine all ticker series into a DataFrame (each row represents a ticker)
    combined_df = pd.DataFrame.from_dict(flattened_data, orient="index")
    
    # Ensure the data directory exists
    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    # Construct file name: source + interval + period
    file_name = f"yfinance_{interval}_{period}.csv"
    file_path = os.path.join(data_dir, file_name)
    
    # Save the combined DataFrame to CSV
    combined_df.to_csv(file_path, index=True)
    app_logger.info(f"Saved combined data for {len(flattened_data)} tickers to {file_path}")
    
    return JSONResponse(content={"message": "Data downloaded and stored", "file": file_name})


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