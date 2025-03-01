import pandas as pd
from supabase import create_client, Client
from dotenv import load_dotenv
import os

# ------------------------------------Load environment variables-------------------------------------- #

load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_HOST_URL")
SUPABASE_KEY = os.getenv("SUPABASE_ANON_API_KEY")

# ---------------------------------------------------------------------------------------------------- #

# -----------------------------------Initialize Supabase client--------------------------------------- #

# 
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def ingest_data(csv_file):
    """
    Ingests data from a CSV file into the 'tick_data' table in Supabase.

    This function reads a CSV file, iterates over each row, and inserts the data into the 'tick_data' table in Supabase.
    The data is inserted with the following columns: 'timestamp', 'instrument', 'bid_price', 'ask_price', and 'volume'.
    After ingestion, it prints the number of rows ingested into the 'tick_data' table.

    Parameters:
    csv_file (str): The path to the CSV file to be ingested.

    Returns:
    None
    """
    df = pd.read_csv(csv_file)
    for _, row in df.iterrows():
        data = {
            "timestamp": row["timestamp"],
            "instrument": row["instrument"],
            "bid_price": float(row["bid_price"]),
            "ask_price": float(row["ask_price"]),
            "volume": int(row["volume"])
        }
        supabase.table("tick_data").insert(data).execute()
    print(f"Ingested {len(df)} rows into tick_data.")
# ---------------------------------------------------------------------------------------------------- #

if __name__ == "__main__":
    ingest_data("tick_data.csv")