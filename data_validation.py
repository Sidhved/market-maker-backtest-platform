import pandas as pd
from sklearn.ensemble import IsolationForest
from data_retrieval import get_cached_data
import redis
import pickle

# ------------------------------------Validate Data Function------------------------------------------ #

def validate_data(df):
    """
    Validate the input DataFrame for data integrity and quality.

    This function checks for missing values, duplicates, time gaps, and resamples the data to 1-minute intervals.
    It returns a DataFrame that has been cleaned and resampled.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data to be validated.

    Returns:
    pd.DataFrame: A cleaned and resampled DataFrame.
    """
    # Ensure timestamps are in datetime format
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    # Check for missing values
    missing = df.isnull().sum()
    print(f"Missing values:\n{missing}")
    
    # Check for duplicates and remove them
    duplicates = df.duplicated().sum()
    print(f"Duplicates: {duplicates}")
    if duplicates > 0:
        df.drop_duplicates(inplace=True)
        print(f"Dropped {duplicates} duplicate rows. New row count: {len(df)}")
    
    # Check for time gaps
    df["timestamp_diff"] = df["timestamp"].diff()
    gaps = df[df["timestamp_diff"] > pd.Timedelta("1s")]
    print(f"Time gaps (greater than 1 second): {len(gaps)}")
    if len(gaps) > 0:
        print("Sample of time gaps:")
        print(gaps[["timestamp", "timestamp_diff"]].head())
        gap_seconds = gaps["timestamp_diff"].dt.total_seconds()
        print("\nDistribution of time gaps (in seconds):")
        print(gap_seconds.describe())
    
    # Resample to 1-minute intervals
    print("\nResampling data to 1-minute intervals...")
    df.set_index("timestamp", inplace=True)
    df_resampled = df.groupby("instrument").resample("1min").last()
    df_resampled = df_resampled.drop(columns=["instrument"]).reset_index()
    df_resampled.dropna(subset=["bid_price", "ask_price"], inplace=True)
    df_resampled.set_index("timestamp", inplace=True)
    df_resampled.index = df_resampled.index.to_period("T").to_timestamp()
    df_resampled = df_resampled.reset_index()
    df_resampled["timestamp_diff"] = df_resampled.groupby("instrument")["timestamp"].diff()
    gaps_resampled = df_resampled[df_resampled["timestamp_diff"] > pd.Timedelta("1min")]
    print(f"Time gaps after resampling (greater than 1 minute): {len(gaps_resampled)}")
    if len(gaps_resampled) > 0:
        print("Sample of time gaps after resampling:")
        print(gaps_resampled[["timestamp", "instrument", "timestamp_diff"]].head())
    
    return df_resampled

# -----------------------------------Detect Anomalies Function--------------------------------------- #

def detect_anomalies(df):
    """
    Detect anomalies in the DataFrame using Isolation Forest.

    This function applies the Isolation Forest algorithm to identify anomalies in the data based on
    bid price, ask price, and volume. It returns a DataFrame containing the detected anomalies.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data to analyze.

    Returns:
    pd.DataFrame: A DataFrame containing the detected anomalies.
    """
    model = IsolationForest(contamination=0.01, random_state=42)
    df["anomaly"] = model.fit_predict(df[["bid_price", "ask_price", "volume"]])
    anomalies = df[df["anomaly"] == -1]
    print(f"Anomalies detected: {len(anomalies)}")
    if len(anomalies) > 0:
        print("Sample of anomalies:")
        print(anomalies[["timestamp", "instrument", "bid_price", "ask_price", "volume"]].head())
    
    return anomalies

# -----------------------------------Cache Cleaned Data Function------------------------------------- #

def cache_cleaned_data(df):
    """
    Cache the cleaned DataFrame in Redis.

    This function serializes the cleaned DataFrame and stores it in a Redis database.

    Parameters:
    df (pd.DataFrame): The DataFrame to be cached.

    Returns:
    None
    """
    r = redis.Redis(host="localhost", port=6379, db=0)
    r.set("tick_data_cleaned", pickle.dumps(df))
    print("Cleaned data cached in Redis.")

# ---------------------------------------------------------------------------------------------------- #

if __name__ == "__main__":
    print("Loading data...")
    df = get_cached_data()
    print(f"Data loaded. Total rows: {len(df)}")
    
    print("\n--- Validating Data ---")
    df = validate_data(df)
    print(f"Data after resampling. Total rows: {len(df)}")
    
    print("\n--- Detecting Anomalies ---")
    anomalies = detect_anomalies(df)

    # Cache the cleaned data
    cache_cleaned_data(df)