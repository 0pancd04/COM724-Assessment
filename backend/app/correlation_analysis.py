import pandas as pd
import os
import logging
import plotly.express as px
from .logger import setup_logger

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
logger = setup_logger("correlation_analysis", os.path.join(LOG_DIR, "correlation_analysis.log"))


def perform_correlation_analysis(
    preprocessed_file: str,
    selected_tickers: list,
    feature: str = "Close",
    output_file: str = None,
    chart_file: str = None
) -> (pd.DataFrame, dict, object):
    """
    Performs correlation analysis for the selected cryptocurrencies on a specified feature.

    Returns:
      correlation_df (pd.DataFrame): The computed correlation matrix.
      report (dict): Report with top positive/negative pairs, file paths, & CIs.
      fig: Plotly figure object of the heatmap.
    """
    report = {}
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
    df_feat = df_sel[cols]

    corr_df = df_feat.T.corr()
    # off-diagonal pairs
    pairs = []
    idx = corr_df.index.tolist()
    for i in range(len(idx)):
        for j in range(i+1, len(idx)):
            pairs.append({"pair": (idx[i], idx[j]), "correlation": corr_df.iloc[i,j]})
    pairs_df = pd.DataFrame(pairs)
    report["top_positive_pairs"] = pairs_df.nlargest(4, "correlation").to_dict(orient="records")
    report["top_negative_pairs"] = pairs_df.nsmallest(4, "correlation").to_dict(orient="records")

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
