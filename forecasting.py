import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from data_retrieval import get_cached_data

def arima_forecast(df, steps=10):
    # Ensure the DataFrame has a datetime index
    df = df.set_index("timestamp")
    model = ARIMA(df["bid_price"], order=(5, 1, 0))
    arima_result = model.fit()
    forecast = arima_result.forecast(steps=steps)
    return forecast

def lstm_forecast(df, seq_length=50, epochs=20):
    # Normalize the bid_price data
    scaler = MinMaxScaler()
    data = scaler.fit_transform(df[["bid_price"]])
    
    # Create sequences for LSTM
    def create_sequences(data, seq_length):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i+seq_length])
            y.append(data[i+seq_length])
        return np.array(X), np.array(y)

    X, y = create_sequences(data, seq_length)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    # Build the LSTM model with an Input layer
    model = Sequential([
        Input(shape=(seq_length, 1)),
        LSTM(50),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(X, y, epochs=epochs, batch_size=32, verbose=1)

    # Predict the next step
    last_sequence = data[-seq_length:].reshape(1, seq_length, 1)
    prediction = model.predict(last_sequence)
    # Inverse transform the prediction
    prediction = scaler.inverse_transform(prediction)
    return prediction

if __name__ == "__main__":
    print("Loading data...")
    df = get_cached_data(use_cleaned=True)  # Use cleaned data
    print(f"Data loaded. Total rows: {len(df)}")

    # Filter for a single instrument (e.g., AAPL) to simplify forecasting
    df_aapl = df[df["instrument"] == "AAPL"].copy()
    print(f"AAPL data. Total rows: {len(df_aapl)}")

    # Sample the last 10,000 rows for faster processing
    df_subset = df_aapl.tail(10000).copy()
    print(f"Using subset of {len(df_subset)} rows for forecasting.")

    print("\n--- Running ARIMA Forecast ---")
    arima_pred = arima_forecast(df_subset)
    print(f"ARIMA Forecast (next 10 steps):\n{arima_pred}")

    print("\n--- Running LSTM Forecast ---")
    lstm_pred = lstm_forecast(df_subset)
    print(f"LSTM Forecast (next step): {lstm_pred}")