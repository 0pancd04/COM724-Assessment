import pandas as pd
import os
import logging
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score

from .logger import setup_logger

logger = setup_logger("grouping_analysis", "grouping_analysis.log")

def perform_dimensionality_reduction(input_file: str, output_file: str = None, chart_file: str = None) -> (pd.DataFrame, dict, str, object):
    """
    Loads the preprocessed data, applies PCA and TSNE (each reducing the feature space to 2 dimensions),
    runs KMeans clustering (k=4) on both reduced datasets, computes silhouette scores, and selects
    the best performing reduction method based on the score.

    The chosen reduced data is saved to output_file (if provided) and an interactive scatter plot is generated.
    Optionally, the chart is saved to chart_file in both JSON and HTML formats.

    Returns:
        best_reduced_df (pd.DataFrame): The chosen reduced data (with 2 dimensions and cluster assignments).
        report (dict): A report containing silhouette scores for both PCA and TSNE.
        best_algorithm (str): The chosen algorithm ('PCA' or 'TSNE').
        fig: A Plotly figure object of the reduced data colored by clusters.
    """
    # Load the preprocessed data
    logger.info(f"Loading preprocessed data from {input_file}")
    df = pd.read_csv(input_file, index_col=0)
    report = {}

    # ----- Dimensionality Reduction with PCA -----
    logger.info("Performing PCA reduction")
    pca = PCA(n_components=2, random_state=42)
    pca_components = pca.fit_transform(df)
    pca_df = pd.DataFrame(pca_components, index=df.index, columns=["PC1", "PC2"])
    
    # Cluster on PCA result using KMeans
    kmeans = KMeans(n_clusters=4, random_state=42)
    pca_clusters = kmeans.fit_predict(pca_df)
    pca_df["Cluster"] = pca_clusters
    pca_silhouette = silhouette_score(pca_df[["PC1", "PC2"]], pca_clusters)
    report["PCA_silhouette"] = float(pca_silhouette)

    # ----- Dimensionality Reduction with TSNE -----
    logger.info("Performing TSNE reduction")
    tsne = TSNE(n_components=2, random_state=42, perplexity=5)
    tsne_components = tsne.fit_transform(df)
    tsne_df = pd.DataFrame(tsne_components, index=df.index, columns=["Dim1", "Dim2"])
    
    # Cluster on TSNE result using KMeans
    tsne_clusters = kmeans.fit_predict(tsne_df)
    tsne_df["Cluster"] = tsne_clusters
    tsne_silhouette = silhouette_score(tsne_df[["Dim1", "Dim2"]], tsne_clusters)
    report["TSNE_silhouette"] = float(tsne_silhouette)

    # Choose the best reduction method
    if pca_silhouette >= tsne_silhouette:
        best_algorithm = "PCA"
        best_reduced_df = pca_df.rename(columns={"PC1": "Dim1", "PC2": "Dim2"}).copy()
    else:
        best_algorithm = "TSNE"
        best_reduced_df = tsne_df.copy()

    report["chosen_algorithm"] = best_algorithm

    # Save reduced data if output_file is provided
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        best_reduced_df.to_csv(output_file)
        logger.info(f"Saved reduced data to {output_file}")
        report["output_file"] = output_file

    # Generate interactive scatter plot using Plotly
    fig = px.scatter(best_reduced_df, x="Dim1", y="Dim2", color=best_reduced_df["Cluster"].astype(str),
                     title=f"Dimensionality Reduction ({best_algorithm}) with KMeans Clusters",
                     hover_name=best_reduced_df.index)
    
    # Save the chart in both JSON and HTML formats if chart_file is provided.
    if chart_file:
        os.makedirs(os.path.dirname(chart_file), exist_ok=True)
        # Save JSON version
        with open(chart_file, "w") as f:
            f.write(fig.to_json())
        logger.info(f"Saved interactive chart JSON to {chart_file}")
        report["chart_file_json"] = chart_file
        
        # Save HTML version by replacing .json extension with .html
        chart_html_file = chart_file.replace(".json", ".html")
        with open(chart_html_file, "w") as f:
            f.write(fig.to_html(full_html=True))
        logger.info(f"Saved interactive chart HTML to {chart_html_file}")
        report["chart_file_html"] = chart_html_file

    return best_reduced_df, report, best_algorithm, fig



def perform_clustering_analysis(reduced_data_file: str, output_file: str = None, chart_file: str = None) -> (pd.DataFrame, dict, object):
    """
    Loads the dimensionally reduced data, applies three clustering algorithms (KMeans, Agglomerative, DBSCAN),
    computes silhouette scores, and selects the best clustering method (based on silhouette score).
    The clustering result (i.e. cluster assignments) is stored and an interactive scatter plot is generated.
    Optionally, the chart is saved to chart_file in both JSON and HTML formats.

    Returns:
        cluster_result_df (pd.DataFrame): DataFrame with the reduced data and the chosen cluster assignments.
        report (dict): Report containing silhouette scores for each clustering method and the chosen method.
        fig: A Plotly figure object visualizing the clusters.
    """
    logger.info(f"Loading reduced data from {reduced_data_file}")
    df = pd.read_csv(reduced_data_file, index_col=0)
    X = df[["Dim1", "Dim2"]].values
    report = {}

    # ----- Clustering with KMeans -----
    kmeans = KMeans(n_clusters=4, random_state=42)
    kmeans_labels = kmeans.fit_predict(X)
    kmeans_score = silhouette_score(X, kmeans_labels)
    report["KMeans_silhouette"] = float(kmeans_score)

    # ----- Clustering with Agglomerative Clustering -----
    agg = AgglomerativeClustering(n_clusters=4)
    agg_labels = agg.fit_predict(X)
    agg_score = silhouette_score(X, agg_labels)
    report["Agglomerative_silhouette"] = float(agg_score)

    # ----- Clustering with DBSCAN -----
    dbscan = DBSCAN(eps=0.5, min_samples=2)
    dbscan_labels = dbscan.fit_predict(X)
    # DBSCAN may result in a single cluster or noise; check for valid clustering.
    if len(set(dbscan_labels)) > 1 and -1 not in set(dbscan_labels):
        dbscan_score = silhouette_score(X, dbscan_labels)
    else:
        dbscan_score = -1
    report["DBSCAN_silhouette"] = float(dbscan_score)

    # Determine best clustering method
    best_method = None
    best_score = -1
    best_labels = None
    for method, score, labels in [("KMeans", kmeans_score, kmeans_labels),
                                  ("Agglomerative", agg_score, agg_labels),
                                  ("DBSCAN", dbscan_score, dbscan_labels)]:
        if score > best_score:
            best_score = score
            best_method = method
            best_labels = labels
    report["chosen_clustering_method"] = best_method
    report["chosen_silhouette"] = float(best_score)

    # Append cluster assignments to the DataFrame
    cluster_result_df = df.copy()
    cluster_result_df["Cluster"] = best_labels

    # Save clustering results if output_file is provided
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        cluster_result_df.to_csv(output_file)
        logger.info(f"Saved clustering result to {output_file}")
        report["clustering_output_file"] = output_file

    # Generate interactive scatter plot using Plotly
    fig = px.scatter(cluster_result_df, x="Dim1", y="Dim2", color=cluster_result_df["Cluster"].astype(str),
                     title=f"Clustering Analysis: {best_method} (Silhouette: {best_score:.3f})",
                     hover_name=cluster_result_df.index)

    # Save the chart in both JSON and HTML formats if chart_file is provided.
    if chart_file:
        os.makedirs(os.path.dirname(chart_file), exist_ok=True)
        # Save JSON version
        with open(chart_file, "w") as f:
            f.write(fig.to_json())
        logger.info(f"Saved clustering chart JSON to {chart_file}")
        report["chart_file_json"] = chart_file
        
        # Save HTML version by replacing .json with .html in chart_file name
        chart_html_file = chart_file.replace(".json", ".html")
        with open(chart_html_file, "w") as f:
            f.write(fig.to_html(full_html=True))
        logger.info(f"Saved clustering chart HTML to {chart_html_file}")
        report["chart_file_html"] = chart_html_file

    return cluster_result_df, report, fig
