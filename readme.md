Below is an example of a comprehensive README file for your project:

---

# Crypto Prediction Platform

This project is a Dockerized FastAPI backend designed for a crypto prediction platform. It downloads historical cryptocurrency data, performs data preprocessing, dimensionality reduction, clustering, correlation analysis, and extensive exploratory data analysis (EDA). Interactive charts are generated for each step, which can be accessed via dedicated API endpoints.

## Project Structure

```plaintext
.
├── backend
│   ├── app
│   │   ├── main.py                # FastAPI application entry point
│   │   ├── data_downloader.py     # Module for downloading crypto data
│   │   ├── data_preprocessing.py  # Module for cleaning and preprocessing data
│   │   ├── grouping_analysis.py   # Module for dimensionality reduction, clustering & correlation analysis
│   │   ├── eda_analysis.py        # Module for Exploratory Data Analysis (EDA)
│   │   └── logger.py              # Logging configuration module
│   ├── Dockerfile                 # Dockerfile for backend service
│   └── requirements.txt           # Python dependencies
└── docker-compose.yml             # Docker Compose configuration
```

## Prerequisites

-   [Docker](https://docs.docker.com/get-docker/) installed on your machine.
-   [Docker Compose](https://docs.docker.com/compose/install/) installed.

## Installation and Setup

1. **Clone the Repository:**

    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

2. **Build the Docker Images:**

    Make sure you are in the root directory (where `docker-compose.yml` is located) and run:

    ```bash
    docker compose build
    ```

    This command will build the Docker image for the backend using the Dockerfile located in the `backend` folder. The image includes all required Python packages:

    - FastAPI, Uvicorn
    - yfinance, lxml, pandas, requests
    - scikit-learn, plotly

3. **Run the Application:**

    Start your backend service with:

    ```bash
    docker compose up
    ```

    Your FastAPI application will run on the container's port 80, mapped to a host port (e.g. 8000 as defined in your `docker-compose.yml`). You can access the API at [http://localhost:8000](http://localhost:8000).

## API Endpoints and Usage

### 1. Data Download and Preprocessing

-   **Data Download:**  
    Your application downloads historical data for the top 30 cryptocurrencies using data providers like Yahoo Finance.  
    _This is handled internally by the backend and stored in a CSV file (e.g., `data/yfinance_1d_5y.csv`)._

-   **Data Preprocessing:**  
    The `data_preprocessing.py` module cleans and scales the downloaded data and stores it in a new CSV file (e.g., `data/preprocessed_yfinance_1d_5y.csv`).  
    It also logs details about missing values, rows dropped, etc.

### 2. Dimensionality Reduction & Clustering

-   **Dimensionality Reduction:**  
    Endpoint: `/dim_reduction`  
    This endpoint compares PCA and TSNE for reducing the feature space (e.g., from 365 days of data) to 2 dimensions, applies KMeans clustering, and selects the best reduction method based on silhouette scores.  
    Reduced data and interactive charts (saved as JSON and HTML) are stored in the `data/` folder.

-   **Clustering Analysis:**  
    Endpoint: `/clustering_analysis`  
    This endpoint loads the reduced data, applies KMeans, Agglomerative, and DBSCAN clustering algorithms, computes silhouette scores, and selects the best clustering method.  
    It returns a report and saves an interactive clustering chart.

-   **Available Tickers:**  
    Endpoint: `/available_tickers`  
    Lists available tickers grouped by cluster from the clustering results. This helps you pick one ticker per cluster for further analysis.

-   **Correlation Analysis:**  
    Endpoint: `/correlation_analysis`  
    Accepts a comma-separated list of 4 tickers (one from each cluster) and computes the Pearson correlation matrix based on a chosen feature (e.g., Close price).  
    It identifies and reports the top 4 positive and top 4 negative correlation pairs, and saves an interactive heatmap chart.

### 3. Exploratory Data Analysis (EDA)

-   **EDA Analysis:**  
    Endpoint: `/eda_analysis`  
    Performs EDA for a selected cryptocurrency.  
    It "unflattens" the preprocessed data into a time series with columns for Open, High, Low, Close, and Volume.  
    It generates multiple interactive charts including:

    -   Temporal line chart
    -   Distribution histograms
    -   Box plots
    -   7-Day rolling average chart
    -   **Advanced Charts:**
        -   Candlestick chart (for OHLC data)
        -   Rolling volatility chart (14-day rolling standard deviation of daily returns)

-   **EDA Chart Endpoints:**  
    Dedicated endpoints serve each EDA chart as interactive HTML pages:

    -   `/eda_chart/temporal_line`
    -   `/eda_chart/histograms`
    -   `/eda_chart/box_plots`
    -   `/eda_chart/rolling_average`
    -   `/eda_chart/candlestick`
    -   `/eda_chart/rolling_volatility`

    For example, access [http://localhost:8000/eda_chart/temporal_line?ticker=BTC](http://localhost:8000/eda_chart/temporal_line?ticker=BTC) to view the temporal line chart for BTC.

### 4. Interactive Chart Endpoints

-   **Dimensionality Reduction Chart:**  
    Endpoint: `/dim_reduction_chart`  
    Serves the HTML version of the dimensionality reduction interactive chart.

-   **Clustering Chart:**  
    Endpoint: `/clustering_chart`  
    Serves the HTML version of the clustering analysis interactive chart.

-   **Correlation Chart:**  
    Endpoint: `/correlation_chart`  
    Serves the HTML version of the correlation analysis interactive heatmap.

## Running and Testing

1. **Start the Service:**

    ```bash
    docker compose up
    ```

2. **Access the API Documentation:**

    FastAPI automatically generates interactive documentation (Swagger UI) at:
    [http://localhost:8000/docs](http://localhost:8000/docs)

3. **Test Endpoints:**

    Use the Swagger UI to test endpoints such as `/dim_reduction`, `/clustering_analysis`, `/available_tickers`, `/correlation_analysis`, and `/eda_analysis`.  
    For EDA charts, use the dedicated endpoints to view interactive HTML charts.

## Logging

-   Logs for each module (data download, preprocessing, grouping analysis, EDA) are saved in the `backend/app/` folder with filenames like `data_preprocessing.log`, `grouping_analysis.log`, and `eda_analysis.log`.
-   Review these logs to monitor process steps and diagnose any issues.

## Future Enhancements

-   **Live Data Streaming:**  
    Integrate live data streaming for interactive chart updates.

-   **Frontend Integration:**  
    Build a frontend dashboard that consumes these API endpoints to provide a complete interactive experience.

-   **Additional Analysis:**  
    Expand EDA with more advanced statistical analyses and visualizations.

---

This README file serves as a complete guide to set up, run, and interact with your crypto prediction platform backend. Adjust file paths and configurations as needed for your environment. Enjoy exploring your crypto data!
