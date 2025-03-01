import os
import json
import redis
import pandas as pd
from dotenv import load_dotenv
from supabase import create_client, Client

# Load environment variables
load_dotenv()
redis_host = os.getenv("REDIS_HOST", "localhost")
redis_port = int(os.getenv("REDIS_PORT", 6379))
redis_password = os.getenv("REDIS_PASSWORD", None)
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")

# Initialize Redis client
# Only include the password parameter if a password is provided
redis_params = {
    "host": redis_host,
    "port": redis_port,
    "decode_responses": True
}
if redis_password and redis_password != "none":
    redis_params["password"] = redis_password
redis_client = redis.Redis(**redis_params)

# Initialize Supabase client
supabase: Client = create_client(supabase_url, supabase_key)

def get_cached_data(use_cleaned=True):
    cache_key = "tick_data_cleaned" if use_cleaned else "tick_data"
    
    # Check if data is in Redis cache
    cached_data = redis_client.get(cache_key)
    if cached_data:
        print(f"Retrieved data from Redis cache (key: {cache_key}).")
        return pd.read_json(cached_data, orient="records")
    
    # Fetch data from Supabase
    table_name = "tick_data_cleaned" if use_cleaned else "tick_data"
    response = supabase.table(table_name).select("*").execute()
    data = response.data
    if not data:
        raise ValueError(f"No data found in Supabase table {table_name}")
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    # Sort by timestamp to ensure chronological order
    df = df.sort_values(by="timestamp")
    
    # Keep only the last 5,000 rows (1,000 per instrument, 5 instruments)
    df = df.tail(5000)
    
    # Cache the data in Redis
    redis_client.set(cache_key, df.to_json(orient="records"))
    print(f"Cached data to Redis (key: {cache_key}).")
    
    return df
