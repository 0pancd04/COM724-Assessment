import pandas as pd
import os
import logging
import plotly.express as px
from .logger import setup_logger

logger = setup_logger("grouping_analysis", "grouping_analysis.log")

def perform_correlation_analysis(preprocessed_file: str, selected_tickers: list, feature: str = "Close", output_file: str = None, chart_file: str = None) -> (pd.DataFrame, dict, object):
    """
    Performs correlation analysis for the selected cryptocurrencies on a specified feature.
    
    Assumptions:
      - The preprocessed CSV file is in flattened format with rows as tickers and columns as "YYYY-MM-DD_<feature>".
      - The function will compute the Pearson correlation between the selected coins using the chosen feature.
    
    Steps:
      1. Load the preprocessed data.
      2. Subset the rows corresponding to the selected tickers.
      3. Extract columns corresponding to the given feature.
      4. Compute the correlation matrix.
      5. Extract off-diagonal correlation pairs and sort to identify the top 4 positive and top 4 negative pairs.
      6. Generate an interactive heatmap of the correlation matrix.
      7. Optionally store the correlation matrix and interactive chart in both JSON and HTML formats.
    
    Returns:
      correlation_df (pd.DataFrame): The computed correlation matrix.
      report (dict): A report including top positive and negative correlation pairs and file paths if saved.
      fig: A Plotly figure object visualizing the correlation matrix.
    """
    report = {}
    
    logger.info(f"Loading preprocessed data from {preprocessed_file}")
    df = pd.read_csv(preprocessed_file, index_col=0)
    
    # Subset the data for the selected tickers
    missing_tickers = [ticker for ticker in selected_tickers if ticker not in df.index]
    if missing_tickers:
        msg = f"The following selected tickers were not found in the data: {missing_tickers}"
        logger.error(msg)
        raise ValueError(msg)
    
    df_selected = df.loc[selected_tickers].copy()
    
    # Extract columns for the specified feature (e.g., columns ending with "_Close")
    feature_suffix = f"_{feature}"
    feature_cols = [col for col in df_selected.columns if col.endswith(feature_suffix)]
    if not feature_cols:
        msg = f"No columns found for feature '{feature}'"
        logger.error(msg)
        raise ValueError(msg)
    
    df_feature = df_selected[feature_cols].copy()
    
    # Compute the correlation matrix (using the transposed data so that each coin's time series is a column)
    correlation_df = df_feature.T.corr()
    
    # Extract off-diagonal pairs from the correlation matrix
    correlations = []
    tickers = correlation_df.index.tolist()
    for i in range(len(tickers)):
        for j in range(i+1, len(tickers)):
            pair = (tickers[i], tickers[j])
            corr_value = correlation_df.iloc[i, j]
            correlations.append({"pair": pair, "correlation": corr_value})
    
    corr_pairs_df = pd.DataFrame(correlations)
    # Sort to get the top 4 positive correlations
    top_positive = corr_pairs_df.sort_values(by="correlation", ascending=False).head(4)
    # And the top 4 negative correlations (lowest correlations)
    top_negative = corr_pairs_df.sort_values(by="correlation", ascending=True).head(4)
    
    report["top_positive_pairs"] = top_positive.to_dict(orient="records")
    report["top_negative_pairs"] = top_negative.to_dict(orient="records")
    
    # Optionally store the correlation matrix to a CSV file
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        correlation_df.to_csv(output_file)
        logger.info(f"Saved correlation matrix to {output_file}")
        report["correlation_output_file"] = output_file
    
    # Generate interactive heatmap using Plotly
    fig = px.imshow(correlation_df, text_auto=True, 
                    title=f"Correlation Matrix for Selected Cryptocurrencies ({feature})",
                    labels={"color": "Correlation"})
    
    # Save the interactive chart in both JSON and HTML formats if chart_file is provided.
    if chart_file:
        os.makedirs(os.path.dirname(chart_file), exist_ok=True)
        # Save JSON version
        with open(chart_file, "w") as f:
            f.write(fig.to_json())
        logger.info(f"Saved correlation chart JSON to {chart_file}")
        report["chart_file_json"] = chart_file
        # Save HTML version (replace .json with .html)
        chart_html_file = chart_file.replace(".json", ".html")
        with open(chart_html_file, "w") as f:
            f.write(fig.to_html(full_html=True))
        logger.info(f"Saved correlation chart HTML to {chart_html_file}")
        report["chart_file_html"] = chart_html_file
    
    return correlation_df, report, fig