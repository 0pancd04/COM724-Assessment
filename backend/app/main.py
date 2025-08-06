import os
import logging
from datetime import datetime, timedelta
from logging.handlers import RotatingFileHandler
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse, JSONResponse
import pandas as pd
import numpy as np
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
from .pipeline_execution_tracker import pipeline_tracker

async def enhance_pipeline_results(pipeline_results):
    """
    Enhance pipeline results with frontend-compatible data structure
    """
    try:
        app_logger.info("Enhancing pipeline results for frontend compatibility", "enhance_pipeline_results")
        
        # Get database summary
        summary_data = crypto_db.get_database_summary()
        
        # Get available tickers
        tickers_data = crypto_db.get_all_tickers()
        
        # Get recent EDA results (sample from available tickers)
        eda_results = []
        sample_tickers = tickers_data[:5] if tickers_data else []
        
        for ticker in sample_tickers:
            try:
                # Try both sources
                for source in ['yfinance', 'binance']:
                    try:
                        # Get EDA data
                        normalized_ticker = unified_handler.normalize_ticker(ticker, source)
                        stored_eda = analysis_storage.get_eda_results(normalized_ticker)
                        
                        if stored_eda:
                            # Get raw data for statistics
                            raw_data = crypto_db.get_ohlcv_data(normalized_ticker, source=source)
                            
                            eda_entry = {
                                "ticker": ticker,
                                "source": source,
                                "data": {
                                    "success": True,
                                    "charts_count": len(stored_eda),
                                    "statistics": {
                                        "total_records": len(raw_data) if not raw_data.empty else 0,
                                        "date_range": {
                                            "start": str(raw_data.index.min()) if not raw_data.empty else None,
                                            "end": str(raw_data.index.max()) if not raw_data.empty else None
                                        } if not raw_data.empty else None
                                    },
                                    "report": {
                                        "num_records": len(raw_data) if not raw_data.empty else 0
                                    }
                                }
                            }
                            eda_results.append(eda_entry)
                            break  # Found data for this ticker, move to next
                    except Exception as e:
                        continue
            except Exception as e:
                continue
        
        # Get model comparison results
        model_results = []
        for ticker in sample_tickers[:3]:  # Limit to 3 for models
            try:
                model_data = model_comparison.get_model_metrics(ticker)
                if model_data:
                    model_results.append({
                        "ticker": ticker,
                        "metrics": model_data
                    })
            except Exception as e:
                continue
        
        enhanced_data = {
            "summary": {
                "data": summary_data
            },
            "tickers": {
                "count": len(tickers_data),
                "tickers": tickers_data
            },
            "edaResults": eda_results,
            "modelResults": model_results,
            "timestamp": datetime.now().isoformat(),
            "pipeline_status": {
                "completed": True,
                "steps_completed": pipeline_results.get('summary', {}).get('completed', 0),
                "total_steps": pipeline_results.get('summary', {}).get('total_steps', 0),
                "success_rate": pipeline_results.get('summary', {}).get('completed', 0) / max(pipeline_results.get('summary', {}).get('total_steps', 1), 1)
            }
        }
        
        app_logger.info(f"Enhanced results: {len(eda_results)} EDA results, {len(model_results)} model results", "enhance_pipeline_results")
        return enhanced_data
        
    except Exception as e:
        app_logger.error(f"Error enhancing pipeline results: {e}", "enhance_pipeline_results")
        return {
            "summary": {"data": []},
            "tickers": {"count": 0, "tickers": []},
            "edaResults": [],
            "modelResults": [],
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }

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
    source: str = Query("yfinance", description="Data source: yfinance or binance"),
    max_days: int = Query(90, description="Maximum days to use"),
    n_clusters: int = Query(4, description="Number of clusters"),
    algorithm: str = Query("kmeans", description="Clustering algorithm: kmeans, hierarchical, or dbscan"),
    feature: str = Query("Close", description="Feature for analysis")
):
    """
    Performs clustering analysis on the data, comparing different algorithms and selecting the best one.
    Returns cluster assignments, statistics, and visualization.
    """
    try:
        app_logger.info(f"Starting clustering analysis with params: source={source}, max_days={max_days}, n_clusters={n_clusters}, algorithm={algorithm}, feature={feature}")
        
        # First perform dimensionality reduction
        _, dim_report, _, _ = perform_dimensionality_reduction(
            use_database=True,
            source=source
        )
        
        if "error" in dim_report:
            raise ValueError(f"Dimensionality reduction failed: {dim_report['error']}")
        
        # Then perform clustering
        output_file = "data/clustering_result.csv"
        chart_file = "data/clustering_chart.json"
        
        cluster_df, report, fig = perform_clustering_analysis(
            output_file=output_file,
            chart_file=chart_file,
            use_database=True,
            source=source
        )
        
        # Add cluster assignments to report
        cluster_assignments = {}
        for cluster in cluster_df["Cluster"].unique():
            tickers = cluster_df[cluster_df["Cluster"] == cluster].index.tolist()
            cluster_assignments[str(cluster)] = tickers
        
        # Calculate cluster characteristics
        cluster_characteristics = {}
        from .unified_data_handler import unified_handler
        
        app_logger.info(f"Calculating characteristics for {len(cluster_df['Cluster'].unique())} clusters")
        
        for cluster in cluster_df["Cluster"].unique():
            tickers = cluster_df[cluster_df["Cluster"] == cluster].index.tolist()
            cluster_data = []
            app_logger.info(f"Processing cluster {cluster} with {len(tickers)} tickers: {tickers}")
            
            for ticker in tickers:
                try:
                    # Clean ticker name (remove source suffix if present)
                    clean_ticker = ticker.replace('_yf', '').replace('_bn', '')
                    base_ticker = clean_ticker.replace('-USD', '').replace('USD', '').replace('USDT', '')
                    app_logger.info(f"Processing ticker: {ticker} -> cleaned: {clean_ticker} -> base: {base_ticker}")
                    
                    # Try to get data from database with different ticker formats
                    df = pd.DataFrame()
                    ticker_variants = [
                        ticker,                    # Original ticker (e.g., ADA_yf)
                        clean_ticker,             # Without source suffix (e.g., ADA)
                        f"{base_ticker}-USD",     # YFinance format (e.g., ADA-USD)
                        f"{base_ticker}USD",      # Alternative format (e.g., ADAUSD)
                        f"{base_ticker}USDT",     # Binance format (e.g., ADAUSDT)
                        base_ticker               # Just the base (e.g., ADA)
                    ]
                    
                    for ticker_variant in ticker_variants:
                        try:
                            df = crypto_db.get_ohlcv_data(ticker_variant, source)
                            if not df.empty:
                                app_logger.info(f"✅ Found data for {ticker} using variant: {ticker_variant}")
                                break
                        except Exception as e:
                            app_logger.debug(f"Failed to get data for {ticker_variant}: {e}")
                            continue
                    
                    if df.empty:
                        # If no data in DB, try to download it
                        app_logger.info(f"No data found for {ticker} in DB, attempting download...")
                        df = unified_handler.download_and_store_data(clean_ticker, source, "90d", "1d")
                    
                    if not df.empty and feature in df.columns:
                        returns = df[feature].pct_change().dropna()
                        if len(returns) > 1:  # Need at least 2 points for meaningful stats
                            volatility = returns.std() * (252 ** 0.5)  # Annualized
                            avg_return = returns.mean() * 252  # Annualized
                            
                            # Sanitize values
                            if not (pd.isna(volatility) or pd.isna(avg_return) or 
                                   np.isinf(volatility) or np.isinf(avg_return)):
                                cluster_data.append({
                                    'ticker': ticker,
                                    'return': float(avg_return),
                                    'volatility': float(volatility),
                                    'data_points': len(returns)
                                })
                                app_logger.info(f"Added data for {ticker}: return={avg_return:.4f}, volatility={volatility:.4f}")
                            else:
                                app_logger.warning(f"Invalid data for {ticker}: return={avg_return}, volatility={volatility}")
                        else:
                            app_logger.warning(f"Insufficient data points for {ticker}: {len(returns)}")
                    else:
                        app_logger.warning(f"No valid data or missing feature '{feature}' for {ticker}")
                        
                except Exception as e:
                    app_logger.warning(f"Error processing {ticker}: {e}")
                    continue
            
            # Calculate cluster statistics
            if cluster_data:
                avg_return = sum(d['return'] for d in cluster_data) / len(cluster_data)
                avg_volatility = sum(d['volatility'] for d in cluster_data) / len(cluster_data)
                
                # Determine risk level
                if avg_volatility > 0.5:  # 50% volatility
                    risk_level = "High"
                elif avg_volatility > 0.3:  # 30% volatility
                    risk_level = "Medium"
                else:
                    risk_level = "Low"
                
                cluster_characteristics[str(cluster)] = {
                    'size': len(tickers),
                    'tickers_with_data': len(cluster_data),
                    'avg_return': float(avg_return),
                    'avg_volatility': float(avg_volatility),
                    'risk_level': risk_level,
                    'ticker_details': cluster_data  # Include individual ticker stats
                }
                
                app_logger.info(f"Cluster {cluster} characteristics: size={len(tickers)}, "
                               f"with_data={len(cluster_data)}, return={avg_return:.4f}, "
                               f"volatility={avg_volatility:.4f}, risk={risk_level}")
            else:
                # Create default characteristics even if no data
                cluster_characteristics[str(cluster)] = {
                    'size': len(tickers),
                    'tickers_with_data': 0,
                    'avg_return': 0.0,
                    'avg_volatility': 0.0,
                    'risk_level': 'Unknown',
                    'ticker_details': []
                }
                app_logger.warning(f"No valid data found for cluster {cluster} with tickers: {tickers}")
        
        # Enhance report with cluster information
        report.update({
            'cluster_assignments': cluster_assignments,
            'cluster_characteristics': cluster_characteristics,
            'n_clusters': len(cluster_assignments),
            'algorithm': algorithm,
            'source': source,
            'feature': feature
        })
        
        return JSONResponse(content={
            "message": "Clustering analysis completed",
            "report": report,
            "chart_file_json": report.get("chart_file_json"),
            "chart_file_html": report.get("chart_file_html")
        })
    except ValueError as e:
        app_logger.error(f"Validation error in clustering analysis: {e}")
        raise HTTPException(status_code=400, detail=str(e))
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
    feature: str = Query("Close", description="Feature for correlation analysis, e.g. 'Close'"),
    source: str = Query("yfinance", description="Data source: yfinance or binance")
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
        
        output_file = "data/correlation_matrix.csv"
        chart_file = "data/correlation_chart.json"
        
        # Normalize feature name
        feature = feature.capitalize()  # Ensure first letter is capital (Close, Open, High, Low, Volume)
        app_logger.info(f"Running correlation analysis for tickers: {tickers}, feature: {feature}, source: {source}")
        
        corr_df, report, fig = perform_correlation_analysis(
            preprocessed_file=None,
            selected_tickers=tickers, 
            feature=feature, 
            output_file=output_file, 
            chart_file=chart_file,
            use_database=True,
            source=source
        )
        
        # Sanitize the report data for JSON serialization
        from .correlation_analysis import sanitize_dict
        
        sanitized_report = sanitize_dict(report)
        
        return JSONResponse(content={
            "message": "Correlation analysis completed",
            "report": sanitized_report
        })
    except ValueError as e:
        app_logger.error(f"Validation error in correlation analysis: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        app_logger.error(f"Unexpected error in correlation analysis: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during correlation analysis")
    
    
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
                
                # Convert column names to lowercase for frontend compatibility
                data_df.columns = [col.lower() if col != 'timestamp' else col for col in data_df.columns]
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
                        charts_data[chart_type] = {
                            'analysis_type': 'eda',
                            'created_at': datetime.now().isoformat()
                        }
                    
                    if chart_name.endswith('_json'):
                        try:
                            with open(chart_path, 'r', encoding='utf-8') as f:
                                chart_json = f.read()
                            charts_data[chart_type]['data'] = chart_json
                            app_logger.info(f"Successfully loaded chart JSON for {chart_type}", "main.get_eda_results")
                        except Exception as e:
                            app_logger.warning(f"Could not read chart JSON {chart_path}: {e}", "main.get_eda_results")
                    elif chart_name.endswith('_html'):
                        try:
                            with open(chart_path, 'r', encoding='utf-8') as f:
                                chart_html = f.read()
                            charts_data[chart_type]['html'] = chart_html
                            app_logger.info(f"Successfully loaded chart HTML for {chart_type}", "main.get_eda_results")
                        except Exception as e:
                            app_logger.warning(f"Could not read chart HTML {chart_path}: {e}", "main.get_eda_results")
                
                app_logger.info(f"Processed {len(charts_data)} chart types from {len(charts)} chart files", "main.get_eda_results")
                
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
        
        # Use the report from fresh EDA analysis if available
        if 'report' in locals() and report:
            report_data = report
            app_logger.info(f"Using fresh EDA report with {len(report)} statistics", "main.get_eda_results")
        
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
                    "end": str(ticker_data[-1]['timestamp']) if ticker_data else None,
                    "days": len(ticker_data) if ticker_data else 0
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
@app.get("/train/{ticker}")
async def train_models(
    ticker: str,
    feature: str = Query("Close", description="Feature column"),
    test_size: float = Query(0.2, description="Test split fraction"),
    source: str = Query(None, description="Data source: 'yfinance' or 'binance'"),
    history_length: str = Query("90d", description="History length for training: 90d, 180d, 365d, 730d, 1825d")
):
    """Train and compare all models using the unified model comparison framework"""
    try:
        app_logger.info(f"Training models for {ticker} with feature={feature}, test_size={test_size}, source={source}, history_length={history_length}")
        
        # Use the model comparison framework for comprehensive training
        results = model_comparison.train_all_models(ticker, feature, test_size, source, history_length)
        
        if 'error' in results:
            app_logger.error(f"Model training failed for {ticker}: {results['error']}")
            raise HTTPException(status_code=500, detail=results['error'])
        
        app_logger.info(f"Successfully trained models for {ticker}")
        
        # Extract metrics for frontend compatibility
        metrics = {}
        if 'models' in results:
            for model_name, model_data in results['models'].items():
                if 'metrics' in model_data:
                    metrics[model_name] = model_data['metrics']
                else:
                    metrics[model_name] = {'error': model_data.get('error', 'Unknown error')}
        
        return JSONResponse(content={
            "message": f"Models trained for {ticker}",
            "ticker": ticker,
            "feature": feature,
            "test_size": test_size,
            "source": source,
            "history_length": history_length,
            "results": results,
            "metrics": metrics,  # Add metrics for frontend compatibility
            "best_model": results.get('best_model'),
            "data_info": results.get('data_info')
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
    model_type: str = Query("arima", description="arima|sarima|random_forest|xgboost|lstm"),
    periods: int = Query(7, description="Number of periods to forecast"),
    source: str = Query("yfinance", description="Data source: yfinance or binance")
):
    """Return forecast for next `periods` days starting from tomorrow"""
    try:
        app_logger.info(f"Generating forecast for {ticker} using {model_type} model, periods={periods}, source={source}")
        
        # Get data from database using the same logic as model training
        df_ticker = pd.DataFrame()
        
        # Clean ticker name and try multiple variants
        clean_ticker = ticker.replace('_yf', '').replace('_bn', '')
        base_ticker = clean_ticker.replace('-USD', '').replace('USD', '').replace('USDT', '')
        
        ticker_variants = [
            ticker,                    # Original ticker
            clean_ticker,             # Without source suffix
            f"{base_ticker}-USD",     # YFinance format
            f"{base_ticker}USD",      # Alternative format
            f"{base_ticker}USDT",     # Binance format
            base_ticker               # Just the base
        ]
        
        for ticker_variant in ticker_variants:
            try:
                df_ticker = crypto_db.get_ohlcv_data(ticker_variant, source)
                if not df_ticker.empty:
                    app_logger.info(f"Found data for {ticker} using variant: {ticker_variant}")
                    break
            except Exception as e:
                continue
        
        if df_ticker.empty:
            # Try to download the data if not available
            app_logger.info(f"Data not found for {ticker}, attempting to download...")
            df_ticker = unified_handler.download_and_store_data(
                ticker=clean_ticker,
                source=source,
                period="90d",
                interval="1d"
            )
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
            model_comparison.train_all_models(ticker, feature="Close", test_size=0.2, source=source)
            pkl = load_model(ticker, model_type)

        if model_type == "arima":
            fc = forecast_arima(pkl, periods)
        elif model_type == "sarima":
            fc = forecast_sarima(pkl, periods)
        elif model_type == "random_forest":
            fc = forecast_random_forest(pkl, last_series, periods)
        elif model_type == "xgboost":
            fc = forecast_xgboost(pkl, last_series, periods)
        elif model_type == "lstm":
            fc = forecast_lstm(pkl, last_series, periods)
        else:
            raise HTTPException(400, f"Unknown model_type: {model_type}")

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
    model_type: str = Query("arima", description="arima|sarima|random_forest|xgboost|lstm"),
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
            
            # Get data for cached response enhancement
            df_ticker = pd.DataFrame()
            clean_ticker = ticker.replace('_yf', '').replace('_bn', '')
            base_ticker = clean_ticker.replace('-USD', '').replace('USD', '').replace('USDT', '')
            
            ticker_variants = [
                ticker, clean_ticker, f"{base_ticker}-USD", f"{base_ticker}USD", f"{base_ticker}USDT", base_ticker
            ]
            
            for ticker_variant in ticker_variants:
                try:
                    df_ticker = crypto_db.get_ohlcv_data(ticker_variant, source)
                    if not df_ticker.empty:
                        break
                except Exception:
                    continue
            
            if not df_ticker.empty:
                # Generate enhanced cached response with actual data
                from .predictive_modeling import compute_rsi, compute_macd
                
                latest_prices = df_ticker["Close"].tail(50)
                rsi = compute_rsi(latest_prices)
                macd_data = compute_macd(latest_prices)
                
                current_rsi = float(rsi.iloc[-1]) if not rsi.empty and not pd.isna(rsi.iloc[-1]) else 50.0
                current_macd = macd_data.iloc[-1] if not macd_data.empty else pd.Series({"MACD": 0, "Signal": 0, "Hist": 0})
                
                ma20 = float(latest_prices.rolling(20).mean().iloc[-1]) if len(latest_prices) >= 20 else float(latest_prices.iloc[-1])
                current_price = float(latest_prices.iloc[-1])
                
                # Determine current signal from raw_signals
                latest_date = max(signals_out.keys()) if signals_out else None
                current_signal = signals_out.get(latest_date, "HOLD") if latest_date else "HOLD"
                
                # Calculate confidence and returns
                confidence = 0.8 if current_signal != "HOLD" else 0.6
                expected_return = 0.03 if current_signal == "BUY" else -0.02 if current_signal == "SELL" else 0.0
                
                # Prepare price data for chart - adjust range based on periods
                # Show more historical context for longer prediction periods
                historical_days = max(30, periods * 2)  # At least 30 days, or 2x prediction period
                price_data = []
                chart_data = df_ticker.tail(historical_days)
                for idx, row in chart_data.iterrows():
                    price_data.append({
                        "date": idx.strftime("%Y-%m-%d"),
                        "price": float(row["Close"])
                    })
                
                # Prepare cached signals list for processing
                cached_signals_list = list(sorted(signals_out.items()))
                
                # Convert raw_signals to signals array for chart - match requested periods
                signals_array = []
                signal_mapping = {"BUY": 1, "SELL": -1, "HOLD": 0}
                
                # Generate signals array for the requested number of periods
                for i in range(periods):
                    if i < len(cached_signals_list):
                        signal = cached_signals_list[i][1]
                    else:
                        signal = "HOLD"
                    signals_array.append(signal_mapping.get(signal, 0))
                
                # Generate predictions based on requested periods (not just cached signals)
                predictions = []
                
                # Generate predictions for the requested number of periods
                start_date = datetime.utcnow().date() + timedelta(days=1)
                
                for i in range(periods):
                    pred_date = start_date + timedelta(days=i)
                    pred_date_str = pred_date.strftime("%Y-%m-%d")
                    
                    # Use cached signal if available, otherwise default to HOLD
                    if i < len(cached_signals_list):
                        signal = cached_signals_list[i][1]
                    else:
                        signal = "HOLD"
                    
                    predicted_price = current_price * (1 + (i * 0.01))  # Simple price progression
                    change_percent = ((predicted_price - current_price) / current_price) * 100
                    predictions.append({
                        "period": f"Day {i+1}",
                        "date": pred_date_str,
                        "price": predicted_price,
                        "change": change_percent,
                        "signal": signal,
                        "confidence": max(0.1, confidence - (i * 0.05))
                    })
                
                return JSONResponse({
                    "current_signal": current_signal,
                    "confidence": confidence,
                    "expected_return": expected_return,
                    "target_price": current_price * (1 + expected_return),
                    "stop_loss": current_price * 0.95 if current_signal == "BUY" else current_price * 1.05,
                    "risk_level": "Medium",
                    "risk_score": 0.5,
                    "indicators": {
                        "rsi": current_rsi,
                        "macd": {"signal": float(current_macd.get("MACD", 0)) if hasattr(current_macd, 'get') else 0.0},
                        "ma_trend": "Bullish" if current_price > ma20 else "Bearish",
                        "ma20": ma20,
                        "volume_trend": "Normal",
                        "volume_change": 0.0
                    },
                    "price_data": price_data,
                    "signals": signals_array,
                    "predictions": predictions,
                    "raw_signals": signals_out,
                    "pnl": pnl_out,
                    "from_cache": True
                })
            else:
                # Fallback for when no data is available
                return JSONResponse({
                    "current_signal": "HOLD",
                    "confidence": 0.5,
                    "expected_return": 0.0,
                    "target_price": 0.0,
                    "stop_loss": 0.0,
                    "risk_level": "Medium",
                    "risk_score": 0.5,
                    "indicators": {
                        "rsi": 50.0,
                        "macd": {"signal": 0.0},
                        "ma_trend": "Neutral",
                        "ma20": 0.0,
                        "volume_trend": "Normal",
                        "volume_change": 0.0
                    },
                    "price_data": [],
                    "signals": [],
                    "predictions": [],
                    "raw_signals": signals_out,
                    "pnl": pnl_out,
                    "from_cache": True
                })
        
        # Get data from database using the same logic as model training
        df_ticker = pd.DataFrame()
        
        # Clean ticker name and try multiple variants
        clean_ticker = ticker.replace('_yf', '').replace('_bn', '')
        base_ticker = clean_ticker.replace('-USD', '').replace('USD', '').replace('USDT', '')
        
        ticker_variants = [
            ticker,                    # Original ticker
            clean_ticker,             # Without source suffix
            f"{base_ticker}-USD",     # YFinance format
            f"{base_ticker}USD",      # Alternative format
            f"{base_ticker}USDT",     # Binance format
            base_ticker               # Just the base
        ]
        
        for ticker_variant in ticker_variants:
            try:
                df_ticker = crypto_db.get_ohlcv_data(ticker_variant, source)
                if not df_ticker.empty:
                    app_logger.info(f"Found data for {ticker} using variant: {ticker_variant}")
                    break
            except Exception as e:
                continue
        
        if df_ticker.empty:
            # Try to download the data if not available
            app_logger.info(f"Data not found for {ticker}, attempting to download...")
            df_ticker = unified_handler.download_and_store_data(
                ticker=clean_ticker,
                source=source,
                period="90d",
                interval="1d"
            )
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
        elif model_type == "random_forest":
            fc_df = forecast_random_forest(pkl, last_series, periods)
        elif model_type == "xgboost":
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

        # Generate comprehensive signals response
        from .predictive_modeling import compute_rsi, compute_macd
        
        # Get latest price data for indicators
        latest_prices = df_ticker["Close"].tail(50)  # Get last 50 days for indicators
        
        # Calculate technical indicators
        rsi = compute_rsi(latest_prices)
        macd_data = compute_macd(latest_prices)
        
        # Get current values
        current_rsi = float(rsi.iloc[-1]) if not rsi.empty and not pd.isna(rsi.iloc[-1]) else 50.0
        current_macd = macd_data.iloc[-1] if not macd_data.empty else pd.Series({"MACD": 0, "Signal": 0, "Hist": 0})
        
        # Calculate moving averages
        ma20 = float(latest_prices.rolling(20).mean().iloc[-1]) if len(latest_prices) >= 20 else float(latest_prices.iloc[-1])
        ma50 = float(latest_prices.rolling(50).mean().iloc[-1]) if len(latest_prices) >= 50 else float(latest_prices.iloc[-1])
        
        # Determine current signal (most recent)
        current_signal = "HOLD"
        latest_signal_value = signals_df["signal"].iloc[-1] if not signals_df.empty else 0
        if latest_signal_value > 0:
            current_signal = "BUY"
        elif latest_signal_value < 0:
            current_signal = "SELL"
            
        # Calculate confidence based on signal strength and indicators
        confidence = 0.7  # Base confidence
        if current_rsi < 30 and current_signal == "BUY":
            confidence += 0.2
        elif current_rsi > 70 and current_signal == "SELL":
            confidence += 0.2
        confidence = min(confidence, 1.0)
        
        # Calculate expected return (simplified)
        current_price = float(latest_prices.iloc[-1])
        expected_return = 0.05 if current_signal == "BUY" else -0.03 if current_signal == "SELL" else 0
        
        # Risk assessment
        volatility = float(latest_prices.pct_change().std())
        risk_score = min(volatility * 10, 1.0)  # Normalize to 0-1
        risk_level = "High" if risk_score > 0.7 else "Medium" if risk_score > 0.4 else "Low"
        
        # Price targets
        target_price = current_price * (1 + expected_return)
        stop_loss = current_price * 0.95 if current_signal == "BUY" else current_price * 1.05
        
        # Prepare price data for chart - adjust range based on periods
        historical_days = max(30, periods * 2)  # At least 30 days, or 2x prediction period
        price_data = []
        for idx, row in df_ticker.tail(historical_days).iterrows():
            price_data.append({
                "date": idx.strftime("%Y-%m-%d"),
                "price": float(row["Close"])
            })
        
        # Prepare signals array for chart
        signals_array = []
        for idx, signal in signals_df["signal"].items():
            signals_array.append(int(signal))
        
        # Generate predictions (simplified forecast)
        try:
            forecast_response = await get_forecast(ticker, model_type, periods, source)
            forecast_data = forecast_response.body.decode() if hasattr(forecast_response, 'body') else '{}'
            predictions = []
            try:
                import json
                forecast_json = json.loads(forecast_data) if isinstance(forecast_data, str) else forecast_data
                if 'forecast' in forecast_json:
                    for date, data in forecast_json['forecast'].items():
                        predictions.append({
                            "date": date,
                            "price": data.get('forecast', current_price)
                        })
            except:
                pass
        except:
            predictions = []

        return JSONResponse({
            "current_signal": current_signal,
            "confidence": confidence,
            "expected_return": expected_return,
            "target_price": target_price,
            "stop_loss": stop_loss,
            "risk_level": risk_level,
            "risk_score": risk_score,
            "indicators": {
                "rsi": current_rsi,
                "macd": {
                    "signal": float(current_macd.get("MACD", 0)) if hasattr(current_macd, 'get') else 0.0
                },
                "ma_trend": "Bullish" if float(current_price) > float(ma20) else "Bearish",
                "ma20": ma20,
                "volume_trend": "Normal",
                "volume_change": 0.0
            },
            "price_data": price_data,
            "signals": signals_array,
            "predictions": predictions,
            "raw_signals": signals_out.to_dict(),
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
        # Get data from database using the same logic as model training
        df_ticker = pd.DataFrame()
        
        # Clean ticker name and try multiple variants
        clean_ticker = ticker.replace('_yf', '').replace('_bn', '')
        base_ticker = clean_ticker.replace('-USD', '').replace('USD', '').replace('USDT', '')
        
        ticker_variants = [
            ticker, clean_ticker, f"{base_ticker}-USD", f"{base_ticker}USD", f"{base_ticker}USDT", base_ticker
        ]
        
        for ticker_variant in ticker_variants:
            try:
                df_ticker = crypto_db.get_ohlcv_data(ticker_variant, source)
                if not df_ticker.empty:
                    app_logger.info(f"Found data for {ticker} using variant: {ticker_variant}")
                    break
            except Exception as e:
                continue
        
        if df_ticker.empty:
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
        app_logger.info(f"Generating forecast outputs for {ticker} using {model_type} model, source={source}")
        
        # Get data from database using the same logic as model training
        df = pd.DataFrame()
        
        # Clean ticker name and try multiple variants
        clean_ticker = ticker.replace('_yf', '').replace('_bn', '')
        base_ticker = clean_ticker.replace('-USD', '').replace('USD', '').replace('USDT', '')
        
        ticker_variants = [
            ticker,                    # Original ticker
            clean_ticker,             # Without source suffix
            f"{base_ticker}-USD",     # YFinance format
            f"{base_ticker}USD",      # Alternative format
            f"{base_ticker}USDT",     # Binance format
            base_ticker               # Just the base
        ]
        
        for ticker_variant in ticker_variants:
            try:
                df = crypto_db.get_ohlcv_data(ticker_variant, source)
                if not df.empty:
                    app_logger.info(f"Found data for {ticker} using variant: {ticker_variant}")
                    break
            except Exception as e:
                continue
        
        if df.empty:
            # Try to download the data if not available
            app_logger.info(f"Data not found for {ticker}, attempting to download...")
            df = unified_handler.download_and_store_data(
                ticker=clean_ticker,
                source=source,
                period="90d",
                interval="1d"
            )
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
            model_comparison.train_all_models(ticker, feature="Close", test_size=0.2, source=source)
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
        # Get data from database using the same logic as model training
        df = pd.DataFrame()
        
        # Clean ticker name and try multiple variants
        clean_ticker = ticker.replace('_yf', '').replace('_bn', '')
        base_ticker = clean_ticker.replace('-USD', '').replace('USD', '').replace('USDT', '')
        
        ticker_variants = [
            ticker, clean_ticker, f"{base_ticker}-USD", f"{base_ticker}USD", f"{base_ticker}USDT", base_ticker
        ]
        
        for ticker_variant in ticker_variants:
            try:
                df = crypto_db.get_ohlcv_data(ticker_variant, source)
                if not df.empty:
                    app_logger.info(f"Found data for {ticker} using variant: {ticker_variant}")
                    break
            except Exception as e:
                continue
        
        if df.empty:
            # Try to download the data if not available
            app_logger.info(f"Data not found for {ticker}, attempting to download...")
            df = unified_handler.download_and_store_data(
                ticker=clean_ticker,
                source=source,
                period="90d",
                interval="1d"
            )
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
            model_comparison.train_all_models(ticker, feature="Close", test_size=0.2, source=source)
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

@app.get("/pipeline/executions")
async def get_pipeline_executions(
    limit: int = Query(50, description="Maximum number of executions to return"),
    status: str = Query(None, description="Filter by status (running, completed, failed)"),
    pipeline_type: str = Query(None, description="Filter by pipeline type")
):
    """Get list of pipeline executions with filtering options"""
    try:
        executions = pipeline_tracker.get_pipeline_executions(
            limit=limit,
            status=status,
            pipeline_type=pipeline_type
        )
        
        statistics = pipeline_tracker.get_execution_statistics()
        
        return JSONResponse(content={
            "success": True,
            "executions": executions,
            "statistics": statistics,
            "filters": {
                "limit": limit,
                "status": status,
                "pipeline_type": pipeline_type
            },
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        app_logger.error(f"Error retrieving pipeline executions: {e}", "get_pipeline_executions")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve pipeline executions: {str(e)}")

@app.get("/pipeline/execution/{trace_id}")
async def get_pipeline_execution_details(trace_id: str):
    """Get detailed information about a specific pipeline execution"""
    try:
        execution_details = pipeline_tracker.get_pipeline_execution(trace_id)
        
        if not execution_details:
            raise HTTPException(status_code=404, detail=f"Pipeline execution {trace_id} not found")
        
        return JSONResponse(content={
            "success": True,
            "execution": execution_details,
            "timestamp": datetime.now().isoformat()
        })
        
    except HTTPException:
        raise
    except Exception as e:
        app_logger.error(f"Error retrieving pipeline execution {trace_id}: {e}", "get_pipeline_execution_details")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve pipeline execution details: {str(e)}")

@app.get("/pipeline/statistics")
async def get_pipeline_statistics():
    """Get overall pipeline execution statistics"""
    try:
        statistics = pipeline_tracker.get_execution_statistics()
        
        # Get recent executions for trends
        recent_executions = pipeline_tracker.get_pipeline_executions(limit=10)
        
        # Calculate additional metrics
        success_trend = []
        duration_trend = []
        
        for execution in recent_executions:
            success_trend.append({
                "timestamp": execution["start_time"],
                "success_rate": execution["success_rate"]
            })
            if execution["duration_seconds"]:
                duration_trend.append({
                    "timestamp": execution["start_time"],
                    "duration": execution["duration_seconds"]
                })
        
        return JSONResponse(content={
            "success": True,
            "statistics": statistics,
            "trends": {
                "success_rate": success_trend,
                "duration": duration_trend
            },
            "recent_executions": recent_executions,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        app_logger.error(f"Error retrieving pipeline statistics: {e}", "get_pipeline_statistics")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve pipeline statistics: {str(e)}")

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
    include_clustering: bool = Query(True, description="Include clustering analysis"),
    request: Request = None
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
        
        # Start pipeline execution tracking
        user_agent = request.headers.get("User-Agent") if request else None
        client_host = request.client.host if request and hasattr(request, 'client') else None
        
        trace_id = pipeline_tracker.start_pipeline_execution(
            pipeline_type="full",
            tickers=ticker_list,
            sources=source_list,
            period=period,
            interval=interval,
            max_days=max_days,
            feature=feature,
            test_size=test_size,
            include_eda=include_eda,
            include_clustering=include_clustering,
            user_agent=user_agent,
            ip_address=client_host
        )
        
        app_logger.info(f"Pipeline execution started with trace_id: {trace_id}", "main.run_full_pipeline")
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
            include_clustering=include_clustering,
            trace_id=trace_id
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
        
        # Enhance the response with frontend-compatible data structure
        enhanced_results = await enhance_pipeline_results(results)
        
        return JSONResponse(content={
            "message": "Full cryptocurrency analysis pipeline completed",
            "pipeline_results": results,
            "enhanced_results": enhanced_results,  # Add enhanced data for frontend
            "trace_id": trace_id,  # Include trace ID for tracking
            "success": True,
            "timestamp": datetime.now().isoformat()
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
        "random_forest": true,
        "xgboost": false,
        "lstm": true
      }
    """
    types = ["arima", "sarima", "random_forest", "xgboost", "lstm"]
    status = {
        t: os.path.exists(os.path.join(MODEL_DIR, f"{ticker}_{t}.pkl"))
        for t in types
    }
    return JSONResponse(content=status)