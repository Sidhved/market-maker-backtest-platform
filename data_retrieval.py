import pandas as pd
from supabase import create_client, Client
from dotenv import load_dotenv
import os
import redis
import pickle

# ------------------------------------Load environment variables-------------------------------------- #

load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_HOST_URL")
SUPABASE_KEY = os.getenv("SUPABASE_ANON_API_KEY")

# ---------------------------------------------------------------------------------------------------- #

# -----------------------------------Initialize Supabase client--------------------------------------- #

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# -----------------------------------Fetch data from database----------------------------------------- #

def get_data_from_db(page_size=1000):
    """
    Fetch data from the 'tick_data' table in Supabase.

    This function retrieves data in pages from the Supabase database, allowing for large datasets to be fetched
    without overwhelming memory. It returns a DataFrame containing all the fetched data.

    Parameters:
    page_size (int): The number of rows to fetch per request. Default is 1000.

    Returns:
    pd.DataFrame: A DataFrame containing all the data fetched from the database.
    """
    all_data = []
    offset = 0

    while True:
        response = supabase.table("tick_data").select("*").order("timestamp").range(offset, offset + page_size - 1).execute()
        data = response.data
        if not data:
            break
        all_data.extend(data)
        offset += page_size
        print(f"Fetched {len(all_data)} rows so far...")
    
    if not all_data:
        raise ValueError("No data found in tick_data table.")
    
    df = pd.DataFrame(all_data)
    return df

# -----------------------------------Cache data in Redis--------------------------------------------- #

def cache_data(df, key="tick_data"):
    """
    Cache the DataFrame in Redis.

    This function serializes the DataFrame and stores it in a Redis database under the specified key.

    Parameters:
    df (pd.DataFrame): The DataFrame to be cached.
    key (str): The key under which to store the cached data in Redis. Default is 'tick_data'.

    Returns:
    None
    """
    r = redis.Redis(host="localhost", port=6379, db=0)
    r.set(key, pickle.dumps(df))
    print(f"Data cached in Redis under key: {key}.")

# -----------------------------------Retrieve cached data from Redis---------------------------------- #

def get_cached_data(use_cleaned=True):
    """
    Retrieve cached data from Redis.

    This function checks if cleaned data is available in Redis. If not, it fetches raw data from the database,
    caches it, and returns the DataFrame.

    Parameters:
    use_cleaned (bool): If True, attempts to retrieve cleaned data. If False, retrieves raw data.

    Returns:
    pd.DataFrame: The DataFrame containing the cached or newly fetched data.
    """
    r = redis.Redis(host="localhost", port=6379, db=0)
    key = "tick_data_cleaned" if use_cleaned else "tick_data"
    cached = r.get(key)
    if cached:
        print(f"Retrieved data from Redis cache (key: {key}).")
        return pickle.loads(cached)
    else:
        if use_cleaned:
            raise ValueError("Cleaned data not found in Redis. Run data_validation.py first.")
        df = get_data_from_db()
        cache_data(df, key="tick_data")
        return df

# ---------------------------------------------------------------------------------------------------- #

if __name__ == "__main__":
    df = get_cached_data(use_cleaned=False)  # Use raw data for initial retrieval
    print(df.head())
    print(f"Total rows retrieved: {len(df)}")
