import pandas as pd
import numpy as np
import os
import logging
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score

from .logger import setup_enhanced_logger

# Setup logger
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
logger = setup_enhanced_logger("grouping_analysis", os.path.join(LOG_DIR, "grouping_analysis.log"))

def check_existing_clustering(algorithm: str, n_clusters: int = 4):
    """Check if clustering results already exist in database"""
    try:
        from .analysis_storage import analysis_storage
        # This is a placeholder - implement actual database check if needed
        return None
    except Exception as e:
        logger.warning(f"Could not check existing clustering: {e}")
        return None


def perform_dimensionality_reduction(input_file: str = None,
                                     output_file: str = None,
                                     chart_file: str = None,
                                     use_database: bool = True,  # Default to using database
                                     source: str = 'yfinance') -> (pd.DataFrame, dict, str, object):
    """
    Loads the preprocessed data, applies PCA and TSNE (each reducing the feature space to 2 dimensions),
    runs KMeans clustering (k=4) on both reduced datasets, computes silhouette scores, and selects
    the best performing reduction method based on the score.

    The chosen reduced data is saved to output_file (if provided) and an interactive scatter plot is generated.
    Optionally, the chart is saved to chart_file in both JSON and HTML formats.
    Returns:
        best_reduced_df (pd.DataFrame): The chosen reduced data (with 2 dimensions and cluster assignments).
        report (dict): A report containing silhouette scores for PCA and TSNE and chosen algorithm.
        best_algorithm (str): The chosen algorithm ('PCA' or 'TSNE').
        fig: A Plotly figure object of the reduced data colored by clusters.
    """
    # Handle database vs file input
    if use_database and input_file is None:
        logger.info(f"Loading preprocessed data from database for source: {source}")
        # Import here to avoid circular imports
        from .unified_data_handler import unified_handler
        try:
            df = unified_handler.prepare_data_for_preprocessing(source)
            if df.empty:
                logger.warning(f"No preprocessed data available from database for {source}")
                return pd.DataFrame(columns=["Dim1", "Dim2", "Cluster"]), {"error": "No data in database"}, "None", None
        except Exception as e:
            logger.error(f"Error loading data from database: {e}")
            return pd.DataFrame(columns=["Dim1", "Dim2", "Cluster"]), {"error": str(e)}, "None", None
    else:
        if input_file is None:
            logger.error("No input file provided and use_database is False")
            return pd.DataFrame(columns=["Dim1", "Dim2", "Cluster"]), {"error": "No input specified"}, "None", None
        
        logger.info(f"Loading preprocessed data from {input_file}")
        # Read with explicit UTF-8 encoding
        df = pd.read_csv(input_file, index_col=0, encoding='utf-8')
    report = {}
    
    logger.info(f"Loaded data shape: {df.shape}", "perform_dimensionality_reduction")
    logger.info(f"Data columns: {list(df.columns)}", "perform_dimensionality_reduction")
    
    # Handle NaN values - this is the main issue causing PCA/TSNE to fail
    original_shape = df.shape
    logger.info(f"Checking for NaN values in data: {df.isnull().sum().sum()} total NaN values", "perform_dimensionality_reduction")
    
    # Option 1: Drop rows/columns with too many NaNs
    # Drop columns that are more than 50% NaN
    nan_threshold = 0.5
    df_cleaned = df.dropna(axis=1, thresh=int(nan_threshold * len(df)))
    logger.info(f"After dropping columns with >{nan_threshold*100}% NaN: {df_cleaned.shape}", "perform_dimensionality_reduction")
    
    # Drop rows that are more than 50% NaN
    df_cleaned = df_cleaned.dropna(axis=0, thresh=int(nan_threshold * len(df_cleaned.columns)))
    logger.info(f"After dropping rows with >{nan_threshold*100}% NaN: {df_cleaned.shape}", "perform_dimensionality_reduction")
    
    # Option 2: Fill remaining NaN values with forward fill, then backward fill, then 0
    if df_cleaned.isnull().sum().sum() > 0:
        logger.info(f"Filling remaining {df_cleaned.isnull().sum().sum()} NaN values", "perform_dimensionality_reduction")
        df_cleaned = df_cleaned.fillna(method='ffill').fillna(method='bfill').fillna(0)
    
    # Verify no NaN values remain
    remaining_nan = df_cleaned.isnull().sum().sum()
    if remaining_nan > 0:
        logger.error(f"Still have {remaining_nan} NaN values after cleaning!", "perform_dimensionality_reduction")
        # Force fill with 0
        df_cleaned = df_cleaned.fillna(0)
    
    logger.info(f"Data cleaning complete: {original_shape} -> {df_cleaned.shape}, NaN values: {df_cleaned.isnull().sum().sum()}", "perform_dimensionality_reduction")
    
    # Use the cleaned data
    df = df_cleaned
    
    # Check if we have enough data
    if df.empty:
        logger.warning("Empty DataFrame provided for dimensionality reduction")
        # Return empty results with proper structure
        empty_df = pd.DataFrame(columns=["Dim1", "Dim2", "Cluster"])
        report = {
            "error": "Empty DataFrame provided",
            "pca_silhouette": 0,
            "tsne_silhouette": 0,
            "best_algorithm": "None"
        }
        return empty_df, report, "None", None
    
    if df.shape[0] < 4:
        logger.warning(f"Insufficient data for dimensionality reduction: {df.shape[0]} rows (minimum 4 required)")
        # Create minimal dummy results for consistency
        n_samples = max(df.shape[0], 2)  # At least 2 samples for clustering
        dummy_data = []
        for i in range(n_samples):
            dummy_data.append([i % 2, (i // 2) % 2, i % 2])
        
        dummy_df = pd.DataFrame(dummy_data, 
                               columns=["Dim1", "Dim2", "Cluster"],
                               index=df.index[:n_samples] if len(df.index) >= n_samples else [f"dummy_{i}" for i in range(n_samples)])
        report = {
            "error": f"Insufficient data: only {df.shape[0]} rows available",
            "pca_silhouette": 0,
            "tsne_silhouette": 0,
            "best_algorithm": "Dummy"
        }
        report["PCA_silhouette"] = 0.0
        report["TSNE_silhouette"] = 0.0
        report["chosen_algorithm"] = "None"
        
        if output_file:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            dummy_df.to_csv(output_file, index=True, encoding='utf-8')
        
        fig = px.scatter(
            dummy_df, x="Dim1", y="Dim2",
            title="Insufficient Data for Analysis"
        )
        return dummy_df, report, "None", fig

    # PCA reduction
    logger.info("Performing PCA reduction", "perform_dimensionality_reduction")
    try:
        # Additional validation before PCA
        if df.shape[1] < 2:
            raise ValueError(f"Need at least 2 features for PCA, got {df.shape[1]}")
        
        # Check for infinite values
        if np.isinf(df).sum().sum() > 0:
            logger.warning("Found infinite values, replacing with 0", "perform_dimensionality_reduction")
            df = df.replace([np.inf, -np.inf], 0)
        
        n_components = min(2, df.shape[0], df.shape[1])
        pca = PCA(n_components=n_components, random_state=42)
        pca_comps = pca.fit_transform(df)
        
        logger.info(f"PCA completed successfully: {pca_comps.shape}", "perform_dimensionality_reduction")
        
        # Ensure we have 2 dimensions
        if pca_comps.shape[1] < 2:
            pca_comps = np.column_stack([pca_comps, np.zeros(pca_comps.shape[0])])
        
        pca_df = pd.DataFrame(pca_comps, index=df.index, columns=["PC1", "PC2"])
        
        # Clustering with safety checks
        n_clusters = min(4, len(pca_df))
        if n_clusters > 1:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            pca_labels = kmeans.fit_predict(pca_df)
            pca_df["Cluster"] = pca_labels
            
            if len(set(pca_labels)) > 1:
                pca_score = silhouette_score(pca_df[["PC1","PC2"]], pca_labels)
            else:
                pca_score = 0.0
        else:
            pca_df["Cluster"] = 0
            pca_score = 0.0
            
    except Exception as e:
        logger.error(f"PCA failed: {e}", "perform_dimensionality_reduction")
        logger.error(f"DataFrame info - Shape: {df.shape}, NaN count: {df.isnull().sum().sum()}, Inf count: {np.isinf(df).sum().sum()}", "perform_dimensionality_reduction")
        pca_df = pd.DataFrame({"PC1": range(len(df)), "PC2": [0]*len(df), "Cluster": [0]*len(df)}, index=df.index)
        pca_score = 0.0
        
    report["PCA_silhouette"] = float(pca_score)

    # TSNE reduction
    logger.info("Performing TSNE reduction", "perform_dimensionality_reduction")
    try:
        # Additional validation for TSNE
        if df.shape[0] < 4:
            raise ValueError(f"TSNE needs at least 4 samples, got {df.shape[0]}")
        
        # Adjust perplexity based on sample size
        perplexity = min(5, max(2, len(df) // 4))
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        tsne_comps = tsne.fit_transform(df)
        
        logger.info(f"TSNE completed successfully: {tsne_comps.shape}", "perform_dimensionality_reduction")
        tsne_df = pd.DataFrame(tsne_comps, index=df.index, columns=["Dim1", "Dim2"])
        
        # Clustering with safety checks
        n_clusters = min(4, len(tsne_df))
        if n_clusters > 1:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            tsne_labels = kmeans.fit_predict(tsne_df)
            tsne_df["Cluster"] = tsne_labels
            
            if len(set(tsne_labels)) > 1:
                tsne_score = silhouette_score(tsne_df[["Dim1","Dim2"]], tsne_labels)
            else:
                tsne_score = 0.0
        else:
            tsne_df["Cluster"] = 0
            tsne_score = 0.0
            
    except Exception as e:
        logger.error(f"TSNE failed: {e}", "perform_dimensionality_reduction")
        tsne_df = pd.DataFrame({"Dim1": range(len(df)), "Dim2": [0]*len(df), "Cluster": [0]*len(df)}, index=df.index)
        tsne_score = 0.0
        
    report["TSNE_silhouette"] = float(tsne_score)

    # Choose best
    if pca_score >= tsne_score:
        best_algo = "PCA"
        best_df = pca_df.rename(columns={"PC1":"Dim1","PC2":"Dim2"}).copy()
    else:
        best_algo = "TSNE"
        best_df = tsne_df.copy()
    report["chosen_algorithm"] = best_algo

    # Save reduced data
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        best_df.to_csv(output_file, index=True, encoding='utf-8')
        logger.info(f"Saved reduced data to {output_file}")
        report["output_file"] = output_file

    # Plot
    fig = px.scatter(
        best_df, x="Dim1", y="Dim2", color=best_df["Cluster"].astype(str),
        title=f"Dimensionality Reduction ({best_algo}) with KMeans Clusters",
        hover_name=best_df.index
    )

    # Save chart JSON/HTML
    if chart_file:
        os.makedirs(os.path.dirname(chart_file), exist_ok=True)
        with open(chart_file, "w", encoding='utf-8') as f:
            f.write(fig.to_json())
        logger.info(f"Saved chart JSON to {chart_file}")
        report["chart_file_json"] = chart_file
        html_file = chart_file.replace('.json', '.html')
        with open(html_file, "w", encoding='utf-8') as f:
            f.write(fig.to_html(full_html=True))
        logger.info(f"Saved chart HTML to {html_file}")
        report["chart_file_html"] = html_file

    return best_df, report, best_algo, fig


def perform_clustering_analysis(reduced_data_file: str = None,
                                output_file: str = None,
                                chart_file: str = None,
                                use_database: bool = True,
                                source: str = 'yfinance') -> (pd.DataFrame, dict, object):
    """
    Loads the dimensionally reduced data, applies three clustering algorithms (KMeans, Agglomerative, DBSCAN),
    computes silhouette scores, and selects the best clustering method based on score.
    The clustering result is stored and an interactive scatter plot generated.
    Optionally, the chart is saved to chart_file in JSON and HTML formats.
    Returns:
        cluster_result_df (pd.DataFrame): DataFrame with reduced data + cluster assignments.
        report (dict): Silhouette scores and chosen method.
        fig: Plotly figure object visualizing clusters.
    """
    report = {}
    
    # Get data from database or file
    if use_database:
        logger.info(f"Loading data from database for clustering")
        # First get dimensionality reduction results
        existing = check_existing_clustering('dimensionality_reduction', n_clusters=4)
        if existing:
            df = pd.DataFrame(existing['data']['reduced_data'])
        else:
            # Run dimensionality reduction first
            df, _, _, _ = perform_dimensionality_reduction(use_database=True, source=source)
    else:
        if not reduced_data_file:
            raise ValueError("reduced_data_file required when use_database=False")
        logger.info(f"Loading reduced data from {reduced_data_file}")
        # Read with explicit UTF-8 encoding
        df = pd.read_csv(reduced_data_file, index_col=0, encoding='utf-8')
    
    X = df[["Dim1", "Dim2"]].values

    # Check if we have enough data points for clustering
    if len(X) < 4:
        logger.warning(f"Not enough data points for clustering: {len(X)} < 4")
        # Return dummy results
        df["Cluster"] = 0
        report["error"] = "Insufficient data points for clustering"
        report["KMeans_silhouette"] = 0.0
        report["Agglomerative_silhouette"] = 0.0
        report["DBSCAN_silhouette"] = 0.0
        report["chosen_clustering_method"] = "None"
        report["chosen_silhouette"] = 0.0
        
        if output_file:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            df.to_csv(output_file, index=True, encoding='utf-8')
        
        fig = px.scatter(
            df, x="Dim1", y="Dim2",
            title="Insufficient Data for Clustering",
            hover_name=df.index
        )
        return df, report, fig
    
    # KMeans
    try:
        kmeans = KMeans(n_clusters=min(4, len(X)), random_state=42)
        km_labels = kmeans.fit_predict(X)
        if len(set(km_labels)) > 1:
            km_score = silhouette_score(X, km_labels)
        else:
            km_score = 0.0
    except Exception as e:
        logger.warning(f"KMeans clustering failed: {e}")
        km_score = 0.0
        km_labels = [0] * len(X)
    report["KMeans_silhouette"] = float(km_score)

    # Agglomerative
    try:
        agg = AgglomerativeClustering(n_clusters=min(4, len(X)))
        agg_labels = agg.fit_predict(X)
        if len(set(agg_labels)) > 1:
            agg_score = silhouette_score(X, agg_labels)
        else:
            agg_score = 0.0
    except Exception as e:
        logger.warning(f"Agglomerative clustering failed: {e}")
        agg_score = 0.0
        agg_labels = [0] * len(X)
    report["Agglomerative_silhouette"] = float(agg_score)

    # DBSCAN
    try:
        dbscan = DBSCAN(eps=0.5, min_samples=2)
        db_labels = dbscan.fit_predict(X)
        if len(set(db_labels)) > 1 and -1 not in set(db_labels):
            db_score = silhouette_score(X, db_labels)
        else:
            db_score = -1
    except Exception as e:
        logger.warning(f"DBSCAN clustering failed: {e}")
        db_score = -1
        db_labels = [0] * len(X)
    report["DBSCAN_silhouette"] = float(db_score)

    # Choose best
    best_method, best_score, best_labels = max(
        ("KMeans", km_score, km_labels),
        ("Agglomerative", agg_score, agg_labels),
        ("DBSCAN", db_score, db_labels),
        key=lambda x: x[1]
    )
    report["chosen_clustering_method"] = best_method
    report["chosen_silhouette"] = float(best_score)

    # Append clusters
    cluster_result_df = df.copy()
    cluster_result_df["Cluster"] = best_labels

    # Save clustering results
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        cluster_result_df.to_csv(output_file, index=True, encoding='utf-8')
        logger.info(f"Saved clustering result to {output_file}")
        report["clustering_output_file"] = output_file

    # Plot clusters
    fig = px.scatter(
        cluster_result_df, x="Dim1", y="Dim2",
        color=cluster_result_df["Cluster"].astype(str),
        title=f"Clustering Analysis: {best_method} (Silhouette: {best_score:.3f})",
        hover_name=cluster_result_df.index
    )

    # Save chart JSON/HTML
    if chart_file:
        os.makedirs(os.path.dirname(chart_file), exist_ok=True)
        with open(chart_file, "w", encoding='utf-8') as f:
            f.write(fig.to_json())
        logger.info(f"Saved clustering chart JSON to {chart_file}")
        report["chart_file_json"] = chart_file
        html_file = chart_file.replace('.json', '.html')
        with open(html_file, "w", encoding='utf-8') as f:
            f.write(fig.to_html(full_html=True))
        logger.info(f"Saved clustering chart HTML to {html_file}")
        report["chart_file_html"] = html_file

    return cluster_result_df, report, fig
