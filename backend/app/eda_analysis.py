import pandas as pd
import os
import logging
import plotly.express as px
import plotly.graph_objects as go
from .logger import setup_logger

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
logger = setup_logger("eda_analysis", os.path.join(LOG_DIR, "eda_analysis.log"))

def unflatten_ticker_data(ticker: str, preprocessed_file: str) -> pd.DataFrame:
    """
    Reads the preprocessed CSV file and extracts the row for the given ticker,
    then unflattens it into a DataFrame with Date as index and columns: Open, High, Low, Close, Volume.
    Assumes column names are in the format "YYYY-MM-DD_<Metric>".
    """
    # Load with explicit UTF-8 encoding
    df = pd.read_csv(preprocessed_file, index_col=0, encoding='utf-8')
    if ticker not in df.index:
        raise ValueError(f"Ticker {ticker} not found in preprocessed data.")
    row = df.loc[ticker]
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

def perform_eda_analysis(ticker: str, preprocessed_file: str, output_dir: str = None) -> (dict, dict):
    """
    Performs EDA for the given ticker:
      - Reconstructs the time series for Open, High, Low, Close, and Volume.
      - Generates interactive charts and saves them in JSON/HTML formats with UTF-8.
    Returns:
      report (dict): Summary including number of records and file paths.
      charts (dict): File paths for each saved chart.
    """
    report = {}
    charts = {}
    if output_dir is None:
        output_dir = os.path.join(BASE_DIR, "..", "..","data", "eda", ticker)
    os.makedirs(output_dir, exist_ok=True)
    
    # Reconstruct the time series data
    try:
        df_ticker = unflatten_ticker_data(ticker, preprocessed_file)
    except Exception as e:
        logger.error(f"Error unflattening data for {ticker}: {e}")
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
    
    return report, charts
