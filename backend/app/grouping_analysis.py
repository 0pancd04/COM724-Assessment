import pandas as pd
import os
import logging
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score

from .logger import setup_logger

# Setup logger
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
logger = setup_logger("grouping_analysis", os.path.join(LOG_DIR, "grouping_analysis.log"))


def perform_dimensionality_reduction(input_file: str,
                                     output_file: str = None,
                                     chart_file: str = None) -> (pd.DataFrame, dict, str, object):
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
    logger.info(f"Loading preprocessed data from {input_file}")
    # Read with explicit UTF-8 encoding
    df = pd.read_csv(input_file, index_col=0, encoding='utf-8')
    report = {}

    # PCA reduction
    logger.info("Performing PCA reduction")
    pca = PCA(n_components=2, random_state=42)
    pca_comps = pca.fit_transform(df)
    pca_df = pd.DataFrame(pca_comps, index=df.index, columns=["PC1", "PC2"])
    kmeans = KMeans(n_clusters=4, random_state=42)
    pca_labels = kmeans.fit_predict(pca_df)
    pca_df["Cluster"] = pca_labels
    pca_score = silhouette_score(pca_df[["PC1","PC2"]], pca_labels)
    report["PCA_silhouette"] = float(pca_score)

    # TSNE reduction
    logger.info("Performing TSNE reduction")
    tsne = TSNE(n_components=2, random_state=42, perplexity=5)
    tsne_comps = tsne.fit_transform(df)
    tsne_df = pd.DataFrame(tsne_comps, index=df.index, columns=["Dim1", "Dim2"])
    tsne_labels = kmeans.fit_predict(tsne_df)
    tsne_df["Cluster"] = tsne_labels
    tsne_score = silhouette_score(tsne_df[["Dim1","Dim2"]], tsne_labels)
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


def perform_clustering_analysis(reduced_data_file: str,
                                output_file: str = None,
                                chart_file: str = None) -> (pd.DataFrame, dict, object):
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
    logger.info(f"Loading reduced data from {reduced_data_file}")
    # Read with explicit UTF-8 encoding
    df = pd.read_csv(reduced_data_file, index_col=0, encoding='utf-8')
    X = df[["Dim1", "Dim2"]].values
    report = {}

    # KMeans
    kmeans = KMeans(n_clusters=4, random_state=42)
    km_labels = kmeans.fit_predict(X)
    km_score = silhouette_score(X, km_labels)
    report["KMeans_silhouette"] = float(km_score)

    # Agglomerative
    agg = AgglomerativeClustering(n_clusters=4)
    agg_labels = agg.fit_predict(X)
    agg_score = silhouette_score(X, agg_labels)
    report["Agglomerative_silhouette"] = float(agg_score)

    # DBSCAN
    dbscan = DBSCAN(eps=0.5, min_samples=2)
    db_labels = dbscan.fit_predict(X)
    if len(set(db_labels)) > 1 and -1 not in set(db_labels):
        db_score = silhouette_score(X, db_labels)
    else:
        db_score = -1
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
