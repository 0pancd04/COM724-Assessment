import pandas as pd
import os
import logging
import plotly.express as px
from .logger import setup_logger
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
logger = setup_logger("correlation_analysis", os.path.join(LOG_DIR, "correlation_analysis.log"))


def sanitize_float(value):
    """Convert non-JSON-compliant float values to None"""
    import numpy as np
    import math
    
    if value is None:
        return None
    if isinstance(value, (int, float)):
        if math.isnan(value) or math.isinf(value):
            return None
        return float(value)
    return value

def sanitize_dict(d):
    """Recursively sanitize dictionary values"""
    if isinstance(d, dict):
        return {k: sanitize_dict(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [sanitize_dict(item) for item in d]
    elif isinstance(d, (np.floating, float)):
        return sanitize_float(d)
    return d

def perform_correlation_analysis(
    preprocessed_file: str = None,
    selected_tickers: list = None,
    feature: str = "Close",
    output_file: str = None,
    chart_file: str = None,
    use_database: bool = True,
    source: str = 'yfinance'
) -> (pd.DataFrame, dict, object):
    """
    Performs correlation analysis for the selected cryptocurrencies on a specified feature.

    Returns:
      correlation_df (pd.DataFrame): The computed correlation matrix.
      report (dict): Report with top positive/negative pairs, file paths, & CIs.
      fig: Plotly figure object of the heatmap.
    """
    report = {}
    
    if use_database:
        logger.info(f"Loading data from database for tickers: {selected_tickers}")
        from .database import crypto_db
        
        # Get data for each ticker and create a correlation matrix
        ticker_data = {}
        logger.info(f"Starting to fetch data for tickers: {selected_tickers} with feature: {feature} from source: {source}")
        
        from .unified_data_handler import unified_handler
        
        for ticker in selected_tickers:
            try:
                # First try to get data from database
                normalized_ticker = unified_handler.normalize_ticker(ticker, source)
                logger.info(f"Fetching data for normalized ticker: {normalized_ticker}")
                
                data = crypto_db.get_ohlcv_data(normalized_ticker, source=source)
                if data.empty:
                    # Try downloading if not available
                    logger.info(f"No data found for {normalized_ticker}, attempting to download...")
                    downloaded_data = unified_handler.download_and_store_data(normalized_ticker, source, "90d", "1d")
                    if not downloaded_data.empty:
                        data = downloaded_data
                        logger.info(f"Successfully downloaded data for {normalized_ticker}")
                
                if not data.empty:
                    # Convert column names to lowercase
                    data.columns = [col.lower() for col in data.columns]
                    feature_lower = feature.lower()
                    
                    if feature_lower in data.columns:
                        ticker_data[ticker] = data[feature_lower]
                        logger.info(f"Successfully got {len(data[feature_lower])} {feature} points for {ticker}")
                    else:
                        available_cols = list(data.columns)
                        logger.warning(f"Feature {feature} not found in columns: {available_cols} for {ticker}")
                else:
                    logger.warning(f"No data found for {ticker} after download attempt")
            except Exception as e:
                logger.error(f"Error getting data for {ticker}: {str(e)}")
        
        logger.info(f"Collected data for {len(ticker_data)} tickers: {list(ticker_data.keys())}")
        
        if len(ticker_data) < 2:
            error_msg = f"Need at least 2 tickers with data, only got {len(ticker_data)}. Available tickers: {list(ticker_data.keys())}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Create DataFrame for correlation
        df_feat = pd.DataFrame(ticker_data)
        
        # Log data availability
        logger.info(f"Data shape before cleaning: {df_feat.shape}")
        logger.info(f"Missing values before cleaning: {df_feat.isnull().sum().to_dict()}")
        
        # Ensure index is datetime
        if not isinstance(df_feat.index, pd.DatetimeIndex):
            df_feat.index = pd.to_datetime(df_feat.index)
        
        # Sort index to ensure proper alignment
        df_feat = df_feat.sort_index()
        
        # Find common date range
        start_dates = []
        end_dates = []
        for col in df_feat.columns:
            series = df_feat[col].dropna()
            if not series.empty:
                start_dates.append(series.index[0])
                end_dates.append(series.index[-1])
        
        if start_dates and end_dates:
            start_date = max(start_dates)
            end_date = min(end_dates)
            logger.info(f"Common date range found: {start_date} to {end_date}")
        else:
            raise ValueError("Could not determine valid date range for any series")
        
        if start_date is None or end_date is None:
            raise ValueError("Could not determine common date range for correlation")
        
        # Slice to common date range
        df_feat = df_feat.loc[start_date:end_date]
        
        # Drop any remaining NaN values
        df_feat = df_feat.dropna()
        
        # Log data after cleaning
        logger.info(f"Data shape after cleaning: {df_feat.shape}")
        logger.info(f"Date range: {df_feat.index.min()} to {df_feat.index.max()}")
        
        if df_feat.empty:
            raise ValueError("No overlapping data points found between tickers after cleaning")
        
        if len(df_feat) < 2:
            raise ValueError(f"Insufficient data points for correlation. Found only {len(df_feat)} points after cleaning")
        
    else:
        logger.info(f"Loading preprocessed data from {preprocessed_file}")
        # Load with explicit UTF-8 encoding
        df = pd.read_csv(preprocessed_file, index_col=0, encoding='utf-8')

        # Validate tickers
        missing = [t for t in selected_tickers if t not in df.index]
        if missing:
            msg = f"Tickers not found: {missing}"
            logger.error(msg)
            raise ValueError(msg)

        df_sel = df.loc[selected_tickers].copy()
        suffix = f"_{feature}"
        cols = [c for c in df_sel.columns if c.endswith(suffix)]
        if not cols:
            msg = f"No feature columns ending with '{suffix}'"
            logger.error(msg)
            raise ValueError(msg)
        df_feat = df_sel[cols].T

    # Calculate correlation with error handling
    try:
        corr_df = df_feat.corr(method='pearson', min_periods=1)
        logger.info(f"Correlation matrix shape: {corr_df.shape}")
        
        # Handle inf values first
        corr_df = corr_df.replace([np.inf, -np.inf], np.nan)
        
        # Convert to native Python types and handle NaN
        corr_matrix = {}
        for col in corr_df.columns:
            corr_matrix[col] = {}
            for idx in corr_df.index:
                val = corr_df.loc[idx, col]
                if pd.isna(val):
                    corr_matrix[col][idx] = None
                else:
                    corr_matrix[col][idx] = round(float(val), 4)
        
        # Create new DataFrame with cleaned values
        corr_df = pd.DataFrame(corr_matrix)
        
        # Get off-diagonal pairs
        pairs = []
        idx = corr_df.index.tolist()
        for i in range(len(idx)):
            for j in range(i+1, len(idx)):
                corr_value = corr_df.iloc[i,j]
                if corr_value is not None:  # Only include valid correlations
                    pairs.append({
                        "pair": (idx[i], idx[j]),
                        "correlation": round(float(corr_value), 4)
                    })
        
        pairs_df = pd.DataFrame(pairs)
        if not pairs_df.empty:
            report["top_positive_pairs"] = pairs_df.nlargest(4, "correlation").to_dict(orient="records")
            report["top_negative_pairs"] = pairs_df.nsmallest(4, "correlation").to_dict(orient="records")
            
            # Add correlation statistics
            valid_corrs = [p["correlation"] for p in pairs if p["correlation"] is not None]
            if valid_corrs:
                report["statistics"] = {
                    "avg_correlation": round(float(np.mean(valid_corrs)), 4),
                    "max_correlation": round(float(np.max(valid_corrs)), 4),
                    "min_correlation": round(float(np.min(valid_corrs)), 4),
                    "std_correlation": round(float(np.std(valid_corrs)), 4)
                }
        else:
            logger.warning("No valid correlation pairs found")
            report["top_positive_pairs"] = []
            report["top_negative_pairs"] = []
            report["statistics"] = {
                "avg_correlation": None,
                "max_correlation": None,
                "min_correlation": None,
                "std_correlation": None
            }
    except Exception as e:
        logger.error(f"Error calculating correlations: {e}")
        raise ValueError(f"Failed to calculate correlations: {str(e)}")

    # Save correlation matrix
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        corr_df.to_csv(output_file, encoding='utf-8')
        logger.info(f"Saved correlation matrix to {output_file}")
        report["correlation_output_file"] = output_file

    # Heatmap
    fig = px.imshow(
        corr_df,
        text_auto=True,
        title=f"Correlation Matrix ({feature})",
        labels={"color": "Correlation"}
    )

    # Save chart with UTF-8
    if chart_file:
        os.makedirs(os.path.dirname(chart_file), exist_ok=True)
        with open(chart_file, "w", encoding='utf-8') as f:
            f.write(fig.to_json())
        logger.info(f"Saved correlation chart JSON to {chart_file}")
        report["chart_file_json"] = chart_file
        html_file = chart_file.replace(".json", ".html")
        with open(html_file, "w", encoding='utf-8') as f:
            f.write(fig.to_html(full_html=True))
        logger.info(f"Saved correlation chart HTML to {html_file}")
        report["chart_file_html"] = html_file

    # Store to database if using database mode
    if use_database and selected_tickers:
        store_correlation_to_db(corr_df, report, selected_tickers, feature, fig)
    
    return corr_df, report, fig

def store_correlation_to_db(corr_df, report, selected_tickers, feature, fig):
    """Store correlation results to database"""
    try:
        from .analysis_storage import analysis_storage
        
        # Convert correlation matrix to dict and sanitize values
        correlation_matrix = sanitize_dict(corr_df.to_dict())
        
        # Store correlation results
        analysis_storage.store_correlation_results(
            tickers=selected_tickers,
            correlation_matrix=correlation_matrix,
            top_positive=report.get("top_positive_pairs", []),
            top_negative=report.get("top_negative_pairs", [])
        )
        
        logger.info(f"Stored correlation results for {selected_tickers} in database")
        
    except Exception as e:
        logger.error(f"Error storing correlation to database: {e}")
        return False
