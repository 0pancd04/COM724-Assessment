import pandas as pd
from sklearn.preprocessing import StandardScaler
import os
from .logger import setup_logger

logger = setup_logger("data_preprocessing", "data_preprocessing.log")

def preprocess_data(file_path: str, max_days: int = 365, output_file: str = None) -> (pd.DataFrame, dict):
    """
    Preprocess the stored CSV data for clustering using all features (Open, High, Low, Close, Volume)
    and store the preprocessed data in a new CSV if output_file is provided.
    
    The function also returns a report dictionary describing the cleaning steps and updates.
    
    Steps:
      1. Load the CSV file with tickers as rows.
      2. Extract columns corresponding to the features.
      3. Sort these columns by date and then by feature order (Open, High, Low, Close, Volume).
      4. Restrict to the most recent 'max_days' if more than max_days are available.
      5. Convert values to numeric and handle missing values via interpolation, forward/backward filling.
      6. Scale the data using StandardScaler.
      7. Optionally store the preprocessed DataFrame to CSV if output_file is provided.
    
    Args:
        file_path (str): Path to the CSV file containing the flattened data.
        max_days (int): Maximum number of days to use (default 365).
        output_file (str): If provided, path to save the preprocessed CSV.
        
    Returns:
        df_scaled (pd.DataFrame): A cleaned and scaled DataFrame.
        report (dict): A dictionary with detailed steps and changes made during preprocessing.
    """
    report = {}

    logger.info(f"Loading data from {file_path}")
    df = pd.read_csv(file_path, index_col=0)
    report["initial_shape"] = df.shape
    report["initial_columns"] = list(df.columns)

    # Define the list of features we want to process
    features = ["Open", "High", "Low", "Close", "Volume"]

    # Filter columns that end with any of the desired features.
    feature_cols = [col for col in df.columns if any(col.endswith(f"_{feat}") for feat in features)]
    if not feature_cols:
        msg = "No columns found for features: " + ", ".join(features)
        logger.error(msg)
        raise ValueError(msg)

    logger.info(f"Found {len(feature_cols)} columns for features {features}")
    report["feature_columns_found"] = len(feature_cols)

    # Define a sort key that extracts the date and orders features in the order specified
    def sort_key(col):
        try:
            date_str, feat = col.rsplit("_", 1)
            d = pd.to_datetime(date_str)
            feat_order = features.index(feat) if feat in features else 999
            return (d, feat_order)
        except Exception as e:
            logger.error(f"Error parsing column name '{col}': {e}")
            return (pd.Timestamp.min, 999)

    # Sort the columns based on date and feature order
    sorted_cols = sorted(feature_cols, key=sort_key)
    
    # Get unique dates from the sorted columns
    unique_dates = sorted({col.rsplit("_", 1)[0] for col in sorted_cols}, key=lambda x: pd.to_datetime(x))
    report["unique_dates_before_restriction"] = len(unique_dates)
    
    # Restrict to most recent max_days if needed
    if len(unique_dates) > max_days:
        logger.info(f"Restricting data to the most recent {max_days} days (from {len(unique_dates)} unique dates)")
        unique_dates = unique_dates[-max_days:]
        sorted_cols = [col for col in sorted_cols if col.rsplit("_", 1)[0] in unique_dates]
    report["columns_after_restriction"] = len(sorted_cols)

    logger.info(f"Total columns after restriction: {len(sorted_cols)}")
    df_features = df[sorted_cols].copy()

    # Convert data to numeric and track missing values
    df_features = df_features.apply(pd.to_numeric, errors='coerce')
    missing_before = df_features.isnull().sum().sum()
    report["missing_values_before_cleaning"] = int(missing_before)
    
    logger.info("Interpolating missing values")
    df_features = df_features.interpolate(axis=1, limit_direction="both")
    missing_after_interp = df_features.isnull().sum().sum()
    report["missing_values_after_interpolation"] = int(missing_after_interp)
    
    logger.info("Filling missing values (forward fill and backward fill)")
    df_features.fillna(method="ffill", axis=1, inplace=True)
    df_features.fillna(method="bfill", axis=1, inplace=True)
    missing_after_fill = df_features.isnull().sum().sum()
    report["missing_values_after_fill"] = int(missing_after_fill)

    # Drop any rows that still contain missing values
    rows_before_drop = df_features.shape[0]
    df_features.dropna(inplace=True)
    rows_after_drop = df_features.shape[0]
    report["rows_dropped"] = rows_before_drop - rows_after_drop

    logger.info("Scaling the data with StandardScaler")
    scaler = StandardScaler()
    scaled_array = scaler.fit_transform(df_features)
    df_scaled = pd.DataFrame(scaled_array, index=df_features.index, columns=df_features.columns)
    report["final_shape"] = df_scaled.shape

    logger.info("Data preprocessing completed successfully")

    # If output_file is provided, save the preprocessed data to CSV
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        df_scaled.to_csv(output_file)
        logger.info(f"Preprocessed data saved to {output_file}")
        report["output_file"] = output_file

    return df_scaled, report
