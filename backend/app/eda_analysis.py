import pandas as pd
import os
import logging
import plotly.express as px
import plotly.graph_objects as go
from .logger import setup_enhanced_logger

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
logger = setup_enhanced_logger("eda_analysis", os.path.join(LOG_DIR, "eda_analysis.log"))

def unflatten_ticker_data(ticker: str, preprocessed_file: str = None, use_database: bool = True, source: str = 'yfinance') -> pd.DataFrame:
    """
    Reads the preprocessed CSV file and extracts the row for the given ticker,
    then unflattens it into a DataFrame with Date as index and columns: Open, High, Low, Close, Volume.
    For database mode, gets OHLCV data directly.
    """
    logger.info(f"Getting ticker data for {ticker}, use_database: {use_database}, source: {source}", "unflatten_ticker_data")
    
    if use_database:
        from .database import crypto_db
        try:
            # Get data directly from database using OHLCV method
            df = crypto_db.get_ohlcv_data(ticker, source=source)
            if df.empty:
                logger.warning(f"No OHLCV data found for {ticker} with source {source}, trying without source", "unflatten_ticker_data")
                # Try without source specification
                df = crypto_db.get_ohlcv_data(ticker)
                if df.empty:
                    raise ValueError(f"No data found for {ticker} in database")
            
            logger.info(f"Successfully retrieved {len(df)} records for {ticker} from database", "unflatten_ticker_data")
            
            # Ensure proper datetime index
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            
            # Ensure we have the expected columns
            expected_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_cols = [col for col in expected_cols if col not in df.columns]
            if missing_cols:
                logger.warning(f"Missing expected columns for {ticker}: {missing_cols}. Available: {list(df.columns)}", "unflatten_ticker_data")
            
            return df.sort_index()
            
        except Exception as e:
            logger.error(f"Error fetching data from database for {ticker}: {e}", "unflatten_ticker_data")
            raise ValueError(f"No data found for {ticker} in database: {str(e)}")
    else:
        # File-based processing (legacy)
        if not preprocessed_file:
            raise ValueError("preprocessed_file required when use_database=False")
        
        logger.info(f"Loading preprocessed file: {preprocessed_file}", "unflatten_ticker_data")
        
        # Load with explicit UTF-8 encoding
        df = pd.read_csv(preprocessed_file, index_col=0, encoding='utf-8')
        if ticker not in df.index:
            raise ValueError(f"Ticker {ticker} not found in preprocessed data.")
        row = df.loc[ticker]
        
        # Unflatten the data
        data = {}
        for col, value in row.items():
            try:
                date_str, metric = col.rsplit("_", 1)
            except Exception:
                continue
            if date_str not in data:
                data[date_str] = {}
            data[date_str][metric] = value
        
        unflattened_df = pd.DataFrame.from_dict(data, orient='index')
        unflattened_df.index = pd.to_datetime(unflattened_df.index)
        unflattened_df.sort_index(inplace=True)
        return unflattened_df

def perform_eda_analysis(ticker: str, preprocessed_file: str = None, output_dir: str = None, use_database: bool = True, source: str = 'yfinance') -> (dict, dict):
    """
    Performs EDA for the given ticker.
    
    Args:
        ticker: Cryptocurrency ticker symbol
        preprocessed_file: Optional path to preprocessed data file
        output_dir: Optional directory for saving charts
        use_database: Whether to use database (default True)
        source: Data source (default 'yfinance')
        
    Returns:
        tuple: (report dict, charts dict)
    """
    logger.info(f"Starting EDA analysis for ticker: {ticker}, source: {source}, use_database: {use_database}", "perform_eda_analysis")
    
    from .unified_data_handler import unified_handler
    from .database import crypto_db
    
    def serialize_timestamp(obj):
        """Helper to serialize timestamps"""
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        return str(obj)
    
    def check_existing_eda(ticker: str, analysis_type: str):
        """Check if EDA results exist in database - placeholder function"""
        # For now, always return None to force fresh analysis
        return None
    
    def store_eda_to_db(ticker: str, analysis_type: str, data: dict, chart_type: str, fig):
        """Store EDA results to database - placeholder function"""
        try:
            from .analysis_storage import analysis_storage
            analysis_storage.store_eda_results(
                ticker=ticker,
                analysis_type=analysis_type,
                data=data,
                chart_type=chart_type,
                chart_html=fig.to_html(full_html=False) if fig else None
            )
        except Exception as e:
            logger.warning(f"Could not store EDA to database: {e}")
            pass
    report = {}
    charts = {}
    if output_dir is None:
        output_dir = os.path.join(BASE_DIR, "..", "..","data", "eda", ticker)
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if EDA already exists in database
    if use_database:
        logger.info(f"Checking for existing EDA results for {ticker}", "perform_eda_analysis")
        existing = check_existing_eda(ticker, 'full_eda')
        if existing:
            logger.info(f"Using existing EDA for {ticker} from database", "perform_eda_analysis")
            return existing['data'].get('report', {}), existing['data'].get('charts', {})
        else:
            logger.info(f"No existing EDA found for {ticker}, generating new analysis", "perform_eda_analysis")
    
    # Reconstruct the time series data
    logger.info(f"Getting data for ticker: {ticker}", "perform_eda_analysis")
    try:
        df_ticker = unflatten_ticker_data(ticker, preprocessed_file, use_database, source)
        logger.info(f"Successfully loaded data for {ticker}: {df_ticker.shape[0]} records, columns: {list(df_ticker.columns)}", "perform_eda_analysis")
    except Exception as e:
        logger.error(f"Error getting data for {ticker}: {e}", "perform_eda_analysis")
        raise

    report["num_records"] = df_ticker.shape[0]
    
    # 1. Temporal Line Chart
    try:
        fig_line = go.Figure()
        for metric in ["Open", "High", "Low", "Close", "Volume"]:
            if metric in df_ticker.columns:
                fig_line.add_trace(go.Scatter(
                    x=df_ticker.index, y=df_ticker[metric], mode="lines", name=metric
                ))
        fig_line.update_layout(
            title=f"Temporal Trends for {ticker}", xaxis_title="Date", yaxis_title="Scaled Value"
        )
        line_json = os.path.join(output_dir, "temporal_line_chart.json")
        line_html = os.path.join(output_dir, "temporal_line_chart.html")
        with open(line_json, "w", encoding='utf-8') as f:
            f.write(fig_line.to_json())
        with open(line_html, "w", encoding='utf-8') as f:
            f.write(fig_line.to_html(full_html=True))
        charts["temporal_line_chart_json"] = line_json
        charts["temporal_line_chart_html"] = line_html
    except Exception as e:
        logger.error(f"EDA Temporal Line Chart for {ticker}: {e}")
        raise
    
    # 2. Histograms
    try:
        df_melt = df_ticker.reset_index().melt(
            id_vars='index', var_name="Metric", value_name="Value"
        )
        fig_hist = px.histogram(
            df_melt, x="Value", facet_col="Metric",
            title=f"Distribution Histograms for {ticker}",
            labels={"Value": "Scaled Value", "Metric": "Metric"}
        )
        hist_json = os.path.join(output_dir, "histograms.json")
        hist_html = os.path.join(output_dir, "histograms.html")
        with open(hist_json, "w", encoding='utf-8') as f:
            f.write(fig_hist.to_json())
        with open(hist_html, "w", encoding='utf-8') as f:
            f.write(fig_hist.to_html(full_html=True))
        charts["histograms_json"] = hist_json
        charts["histograms_html"] = hist_html
        
        # Store to database if 'use_database' is in locals/vars
        if 'use_database' in locals() and use_database:
            store_eda_to_db(ticker, 'distribution', 
                          {'metrics': list(df_ticker.columns)},
                          'histograms', fig_hist)
    except Exception as e:
        logger.error(f"EDA Histograms for {ticker}: {e}")
        raise
    
    # 3. Box Plots
    try:
        fig_box = px.box(
            df_melt, x="Metric", y="Value",
            title=f"Box Plots for {ticker}",
            labels={"Value": "Scaled Value", "Metric": "Metric"}
        )
        box_json = os.path.join(output_dir, "box_plots.json")
        box_html = os.path.join(output_dir, "box_plots.html")
        with open(box_json, "w", encoding='utf-8') as f:
            f.write(fig_box.to_json())
        with open(box_html, "w", encoding='utf-8') as f:
            f.write(fig_box.to_html(full_html=True))
        charts["box_plots_json"] = box_json
        charts["box_plots_html"] = box_html
    except Exception as e:
        logger.error(f"EDA Box Plots for {ticker}: {e}")
        raise
    
    # 4. Rolling Average
    try:
        df_rolling = df_ticker.rolling(window=7).mean()
        fig_roll = go.Figure()
        for metric in ["Open", "High", "Low", "Close", "Volume"]:
            if metric in df_rolling.columns:
                fig_roll.add_trace(go.Scatter(
                    x=df_rolling.index, y=df_rolling[metric], mode="lines",
                    name=f"{metric} (7-day MA)"
                ))
        fig_roll.update_layout(
            title=f"7-Day Rolling Average for {ticker}",
            xaxis_title="Date", yaxis_title="Scaled Value"
        )
        roll_json = os.path.join(output_dir, "rolling_average_chart.json")
        roll_html = os.path.join(output_dir, "rolling_average_chart.html")
        with open(roll_json, "w", encoding='utf-8') as f:
            f.write(fig_roll.to_json())
        with open(roll_html, "w", encoding='utf-8') as f:
            f.write(fig_roll.to_html(full_html=True))
        charts["rolling_average_chart_json"] = roll_json
        charts["rolling_average_chart_html"] = roll_html
    except Exception as e:
        logger.error(f"EDA Rolling Average for {ticker}: {e}")
        raise
    
    # 5. Candlestick
    try:
        if all(m in df_ticker.columns for m in ["Open","High","Low","Close"]):
            fig_candle = go.Figure(data=[go.Candlestick(
                x=df_ticker.index,
                open=df_ticker["Open"], high=df_ticker["High"],
                low=df_ticker["Low"], close=df_ticker["Close"]
            )])
            fig_candle.update_layout(
                title=f"Candlestick Chart for {ticker}",
                xaxis_title="Date", yaxis_title="Price"
            )
            candle_json = os.path.join(output_dir, "candlestick_chart.json")
            candle_html = os.path.join(output_dir, "candlestick_chart.html")
            with open(candle_json, "w", encoding='utf-8') as f:
                f.write(fig_candle.to_json())
            with open(candle_html, "w", encoding='utf-8') as f:
                f.write(fig_candle.to_html(full_html=True))
            charts["candlestick_chart_json"] = candle_json
            charts["candlestick_chart_html"] = candle_html
        else:
            logger.warning(f"Not all OHLC data for candlestick of {ticker}")
    except Exception as e:
        logger.error(f"EDA Candlestick for {ticker}: {e}")
        raise
    
    # 6. Rolling Volatility
    try:
        if "Close" in df_ticker.columns:
            df_returns = df_ticker["Close"].pct_change()
            df_vol = df_returns.rolling(window=14).std()
            fig_vol = go.Figure()
            fig_vol.add_trace(go.Scatter(
                x=df_vol.index, y=df_vol, mode="lines", name="14-day Volatility"
            ))
            fig_vol.update_layout(
                title=f"14-Day Rolling Volatility for {ticker}",
                xaxis_title="Date", yaxis_title="Volatility"
            )
            vol_json = os.path.join(output_dir, "rolling_volatility_chart.json")
            vol_html = os.path.join(output_dir, "rolling_volatility_chart.html")
            with open(vol_json, "w", encoding='utf-8') as f:
                f.write(fig_vol.to_json())
            with open(vol_html, "w", encoding='utf-8') as f:
                f.write(fig_vol.to_html(full_html=True))
            charts["rolling_volatility_chart_json"] = vol_json
            charts["rolling_volatility_chart_html"] = vol_html
        else:
            logger.warning(f"Close data missing for volatility of {ticker}")
    except Exception as e:
        logger.error(f"EDA Volatility for {ticker}: {e}")
        raise
    
    logger.info(f"EDA analysis completed for {ticker}. Generated {len(charts)} charts", "perform_eda_analysis")
    logger.info(f"Charts generated: {list(charts.keys())}", "perform_eda_analysis")
    
    # Store all EDA results to database if using database
    if use_database:
        try:
            logger.info(f"Storing EDA results to database for {ticker}", "perform_eda_analysis")
            store_eda_to_db(ticker, 'full_eda', {'report': report, 'charts': charts}, 'complete_eda', None)
        except Exception as e:
            logger.warning(f"Failed to store EDA results to database: {e}", "perform_eda_analysis")
    
    return report, charts
