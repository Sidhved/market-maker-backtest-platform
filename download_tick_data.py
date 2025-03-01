import os
import time
import pandas as pd
import requests
from dotenv import load_dotenv
from datetime import datetime
import logging
from io import StringIO

# Set up basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("tick_downloader")

# ------------------------------------Load environment variables-------------------------------------- #

# Load environment variables
load_dotenv()

# Alpha Vantage configuration
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY")
BASE_URL = "https://www.alphavantage.co/query"

# Symbols to download
SYMBOLS = ["AAPL", "MSFT", "AMZN", "GOOG", "META"]

# ---------------------------------------------------------------------------------------------------- #

# -----------------------------------Fetch and format stock data--------------------------------------- #

def fetch_and_format_data(symbol):
    """
    Fetch data from Alpha Vantage and convert to desired format.

    This function retrieves intraday stock data for a given symbol, processes it, and returns a DataFrame
    with the relevant columns: 'timestamp', 'instrument', 'bid_price', 'ask_price', and 'volume'.

    Parameters:
    symbol (str): The stock symbol to fetch data for.

    Returns:
    pd.DataFrame: A DataFrame containing the processed stock data.
    """
    logger.info(f"Fetching data for {symbol}")
    
    params = {
        "function": "TIME_SERIES_INTRADAY",
        "symbol": symbol,
        "interval": "1min",
        "apikey": ALPHA_VANTAGE_API_KEY,
        "outputsize": "full",
        "datatype": "csv"
    }
    
    try:
        response = requests.get(BASE_URL, params=params)
        response.raise_for_status()
        
        # Debug: Print the first few lines of the response
        content = response.text
        logger.info(f"Response first 500 chars: {content[:500]}")
        
        # Check if the response is actually JSON (error message) instead of CSV
        if content.strip().startswith('{'):
            logger.error(f"Received JSON error response: {content}")
            return pd.DataFrame()
        
        # Parse CSV data
        df = pd.read_csv(StringIO(content))
        
        if df.empty:
            logger.warning(f"No data found for {symbol}")
            return pd.DataFrame()
        
        # Debug: Print column names
        logger.info(f"CSV columns: {df.columns.tolist()}")
        
        # Look for the timestamp column (might be named differently)
        time_col = None
        for col in df.columns:
            if 'time' in col.lower() or 'date' in col.lower():
                time_col = col
                break
        
        if not time_col:
            logger.error(f"Could not find timestamp column in: {df.columns.tolist()}")
            return pd.DataFrame()
        
        # Convert column names to match our schema
        df_renamed = df.rename(columns={
            time_col: 'timestamp'
        })
        
        # Check if we have price columns
        if 'close' not in df.columns and 'price' not in df.columns:
            price_col = None
            for col in df.columns:
                if 'close' in col.lower() or 'price' in col.lower():
                    price_col = col
                    break
            
            if not price_col:
                logger.error(f"Could not find price column in: {df.columns.tolist()}")
                return pd.DataFrame()
        else:
            price_col = 'close' if 'close' in df.columns else 'price'
        
        # Add instrument column
        df_renamed['instrument'] = symbol
        
        # Create bid/ask prices
        spread = df_renamed[price_col] * 0.0005  # 0.05% spread
        df_renamed['bid_price'] = (df_renamed[price_col] - spread).round(2)
        df_renamed['ask_price'] = (df_renamed[price_col] + spread).round(2)
        
        # Handle volume column
        volume_col = None
        for col in df.columns:
            if 'volume' in col.lower():
                volume_col = col
                break
        
        if not volume_col:
            logger.warning(f"Could not find volume column, using default values")
            df_renamed['volume'] = 100  # Default value
        else:
            df_renamed['volume'] = df_renamed[volume_col]
        
        # Convert timestamp to datetime
        df_renamed['timestamp'] = pd.to_datetime(df_renamed['timestamp'])
        
        # Select and reorder columns to match desired format
        result_df = df_renamed[['timestamp', 'instrument', 'bid_price', 'ask_price', 'volume']]
        
        logger.info(f"Processed {len(result_df)} records for {symbol}")
        return result_df
        
    except Exception as e:
        logger.error(f"Error processing {symbol}: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return pd.DataFrame()

def download_all_symbols():
    """
    Download and format data for all symbols.

    This function iterates over the predefined list of stock symbols, fetches their data using
    fetch_and_format_data, and concatenates the results into a single DataFrame.

    Returns:
    pd.DataFrame: A DataFrame containing the combined data for all symbols.
    """
    all_data = pd.DataFrame()
    
    for symbol in SYMBOLS:
        df = fetch_and_format_data(symbol)
        if not df.empty:
            all_data = pd.concat([all_data, df], ignore_index=True)
        
        # Respect Alpha Vantage API rate limits (5 calls per minute for free tier)
        logger.info(f"Waiting 12 seconds before next request...")
        time.sleep(12)
    
    return all_data

def save_data(df, format_type='csv'):
    """
    Save the data in the specified format.

    This function saves the provided DataFrame to a file in the specified format (CSV, Parquet, or JSON).
    It generates a filename based on the current timestamp.

    Parameters:
    df (pd.DataFrame): The DataFrame to save.
    format_type (str): The format to save the data in ('csv', 'parquet', or 'json').
    
    Returns:
    None
    """
    if df.empty:
        logger.warning("No data to save")
        return
    
    # Create timestamp for filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if format_type == 'csv':
        filename = f"tick_data_{timestamp}.csv"
        df.to_csv(filename, index=False)
        logger.info(f"Data saved to {filename}")
    
    elif format_type == 'parquet':
        filename = f"tick_data_{timestamp}.parquet"
        df.to_parquet(filename, index=False)
        logger.info(f"Data saved to {filename}")
    
    elif format_type == 'json':
        filename = f"tick_data_{timestamp}.json"
        df.to_json(filename, orient='records', date_format='iso')
        logger.info(f"Data saved to {filename}")

# ---------------------------------------------------------------------------------------------------- #

def main():
    """
    Main function to download and save data.

    This function orchestrates the data download process by calling download_all_symbols and
    saving the resulting data in the specified format.

    Returns:
    None
    """
    logger.info("Starting data download process")
    
    # Download and format the data
    data = download_all_symbols()
    
    if not data.empty:
        # Save in CSV format (most common)
        save_data(data, format_type='csv')
        
        # Uncomment below if you need other formats as well
        # save_data(data, format_type='parquet')
        # save_data(data, format_type='json')
        
        logger.info(f"Process complete. Downloaded {len(data)} records.")
    else:
        logger.warning("No data was downloaded.")

if __name__ == "__main__":
    main()