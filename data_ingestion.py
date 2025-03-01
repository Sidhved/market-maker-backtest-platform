import pandas as pd
from supabase import create_client, Client
from dotenv import load_dotenv
import os
import time
from httpx import RemoteProtocolError
import pickle

# ------------------------------------Load environment variables-------------------------------------- #

load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_HOST_URL")
SUPABASE_KEY = os.getenv("SUPABASE_ANON_API_KEY")

# ---------------------------------------------------------------------------------------------------- #

# -----------------------------------Initialize Supabase client--------------------------------------- #

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# -----------------------------------Ingest data from CSV file--------------------------------------- #

def get_last_ingested_id():
    """Fetch the ID of the last row in tick_data to resume ingestion."""
    try:
        response = supabase.table("tick_data").select("id").order("id", desc=True).limit(1).execute()
        if response.data:
            return response.data[0]["id"]
        return 0
    except Exception as e:
        print(f"Error fetching last ID: {e}")
        return 0

def ingest_data(csv_file, batch_size=1000, max_retries=3):
    """
    Ingests data from a CSV file into the 'tick_data' table in Supabase.

    This function reads a CSV file, iterates over each row, and inserts the data into the 'tick_data' table in Supabase.
    The data is inserted with the following columns: 'timestamp', 'instrument', 'bid_price', 'ask_price', and 'volume'.
    After ingestion, it prints the number of rows ingested into the 'tick_data' table.

    Parameters:
    csv_file (str): The path to the CSV file to be ingested.
    batch_size (int): The number of rows to insert in each batch.
    max_retries (int): The maximum number of retries for each batch.

    Returns:
    None
    """
    # Load the CSV
    df = pd.read_csv(csv_file)
    total_rows = len(df)
    print(f"Total rows in CSV: {total_rows}")

    # Get the last ingested ID to resume
    last_id = get_last_ingested_id()
    start_index = last_id  # Since ID is 1-based, start from the next row
    print(f"Resuming from row {start_index + 1} (last ID: {last_id})")

    # Slice the DataFrame to start from the next row
    df = df.iloc[start_index:]
    if df.empty:
        print("No new rows to ingest.")
        return

    batch = []
    ingested_count = start_index
    retry_count = 0

    for idx, row in df.iterrows():
        data = {
            "timestamp": row["timestamp"],
            "instrument": row["instrument"],
            "bid_price": float(row["bid_price"]),
            "ask_price": float(row["ask_price"]),
            "volume": int(row["volume"])
        }
        batch.append(data)

        if len(batch) >= batch_size:
            while retry_count < max_retries:
                try:
                    supabase.table("tick_data").insert(batch).execute()
                    ingested_count += len(batch)
                    print(f"Ingested {ingested_count} of {total_rows} rows")
                    retry_count = 0  # Reset retry counter on success
                    break
                except RemoteProtocolError as e:
                    retry_count += 1
                    print(f"Connection error (attempt {retry_count}/{max_retries}): {e}")
                    if retry_count == max_retries:
                        raise e
                    time.sleep(2 ** retry_count)  # Exponential backoff
            batch = []

    # Insert remaining rows
    if batch:
        while retry_count < max_retries:
            try:
                supabase.table("tick_data").insert(batch).execute()
                ingested_count += len(batch)
                print(f"Ingested {ingested_count} of {total_rows} rows")
                break
            except RemoteProtocolError as e:
                retry_count += 1
                print(f"Connection error (attempt {retry_count}/{max_retries}): {e}")
                if retry_count == max_retries:
                    raise e
                time.sleep(2 ** retry_count)

    print(f"Finished ingesting {ingested_count} rows into tick_data.")

# ---------------------------------------------------------------------------------------------------- #

if __name__ == "__main__":
    ingest_data("tick_data.csv")