from dash import Dash, dcc, html
from dash.dependencies import Input, Output, State  # Explicitly import Dash's Input and Output
import plotly.express as px
import pandas as pd
import numpy as np
import random
from data_retrieval import get_cached_data
import os
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.layers import Input as KerasInput  # Alias Keras's Input to avoid conflict
from sklearn.preprocessing import MinMaxScaler

# ------------------------------------Global Variables-------------------------------------- #

# Global cache for forecasts
forecast_cache = {"ARIMA": {}, "LSTM": {}}

# ------------------------------------MarketMaker Class-------------------------------------- #

class MarketMaker:
    """
    A class to simulate a market maker's trading strategy.

    Attributes:
    base_spread (float): The base spread for quotes.
    forecast_adjustment_factor (float): Factor to adjust quotes based on forecast.
    inventory_limit (int): Maximum inventory limit.
    volatility_window (int): Window size for calculating volatility.
    trade_probability (float): Probability of executing a trade.
    inventory (int): Current inventory of the market maker.
    pnl (float): Profit and loss of the market maker.
    mid_prices (list): List of mid prices for volatility calculation.
    volatilities (list): List of calculated volatilities.
    """

    def __init__(self, base_spread=0.0005, forecast_adjustment_factor=0.1, inventory_limit=100, volatility_window=60, trade_probability=0.5):
        self.base_spread = base_spread
        self.forecast_adjustment_factor = forecast_adjustment_factor
        self.inventory_limit = inventory_limit
        self.volatility_window = volatility_window
        self.trade_probability = trade_probability
        self.inventory = 0
        self.pnl = 0
        self.mid_prices = []
        self.volatilities = []

    def update_volatility(self, mid_price):
        """
        Update the volatility based on the mid price.

        Parameters:
        mid_price (float): The current mid price.

        Returns:
        float: The updated volatility.
        """
        self.mid_prices.append(mid_price)
        if len(self.mid_prices) > self.volatility_window:
            self.mid_prices.pop(0)
        if len(self.mid_prices) > 1:
            returns = np.diff(self.mid_prices) / self.mid_prices[:-1]
            vol = np.std(returns) * np.sqrt(252 * 1440)
            self.volatilities.append(vol)
            return vol
        return 0.1

    def get_quotes(self, market_bid, market_ask, mid_price, forecast_change):
        """
        Generate quotes based on market conditions and forecast changes.

        Parameters:
        market_bid (float): Current market bid price.
        market_ask (float): Current market ask price.
        mid_price (float): Current mid price.
        forecast_change (float): Change in forecast.

        Returns:
        tuple: A tuple containing the bid and ask prices.
        """
        volatility = self.update_volatility(mid_price)
        spread = self.base_spread * (1 + volatility)
        adjustment = self.forecast_adjustment_factor * forecast_change
        spread = spread * (1 + adjustment)
        spread = max(spread, 0.0001)
        inventory_skew = -0.1 * (self.inventory / self.inventory_limit)
        market_spread = market_ask - market_bid
        mm_spread = spread * mid_price
        center_price = mid_price + (inventory_skew * market_spread)
        bid = market_bid + 0.01
        ask = market_ask - 0.01
        if bid >= ask:
            bid = market_bid
            ask = market_ask
        if (ask - bid) < mm_spread:
            bid = center_price - mm_spread / 2
            ask = center_price + mm_spread / 2
        bid = min(bid, market_bid + 0.05)
        ask = max(ask, market_ask - 0.05)
        return bid, ask

    def execute_trade(self, price, side, volume):
        """
        Execute a trade by updating inventory and P&L.

        Parameters:
        price (float): The price at which the trade is executed.
        side (str): The side of the trade ('buy' or 'sell').
        volume (int): The volume of the trade.

        Returns:
        None
        """
        if side == "buy":
            self.inventory += volume
            self.pnl -= price * volume
        elif side == "sell":
            self.inventory -= volume
            self.pnl += price * volume

# ------------------------------------Forecast Functions-------------------------------------- #

def arima_forecast(df, steps=10):
    """
    Compute ARIMA forecast for the given DataFrame's bid_price.

    Parameters:
    df (DataFrame): The DataFrame containing bid prices.
    steps (int): The number of steps to forecast.

    Returns:
    np.array: The forecasted values.
    """
    try:
        df = df.set_index("timestamp")
        # Ensure the index is a DatetimeIndex
        df.index = pd.to_datetime(df.index)
        # Resample to 1-minute intervals and fill gaps
        df = df.resample('min').last().ffill()
        model = ARIMA(df["bid_price"], order=(5, 1, 0))
        arima_result = model.fit()
        forecast = arima_result.forecast(steps=steps)
        if forecast.isna().any():
            print("ARIMA forecast contains NaN values")
            return np.array([])
        return forecast
    except Exception as e:
        print(f"ARIMA forecast failed: {e}")
        return np.array([])

def lstm_forecast(df, steps=10, seq_length=50, epochs=5):
    """
    Compute LSTM forecast for the given DataFrame's bid_price.

    Parameters:
    df (DataFrame): The DataFrame containing bid prices.
    steps (int): The number of steps to forecast.
    seq_length (int): The length of the input sequences.
    epochs (int): The number of training epochs.

    Returns:
    np.array: The forecasted values.
    """
    try:
        scaler = MinMaxScaler()
        data = scaler.fit_transform(df[["bid_price"]])
        
        def create_sequences(data, seq_length, steps):
            X, y = [], []
            for i in range(len(data) - seq_length - steps + 1):
                X.append(data[i:i+seq_length])
                y.append(data[i+seq_length:i+seq_length+steps])
            return np.array(X), np.array(y)

        X, y = create_sequences(data, seq_length, steps)
        if X.shape[0] == 0:
            print("LSTM: Not enough data for sequences")
            return np.array([])

        X = X.reshape((X.shape[0], X.shape[1], 1))

        model = Sequential([
            KerasInput(shape=(seq_length, 1)),
            LSTM(50),
            Dense(steps)  # Predict 'steps' future values
        ])
        model.compile(optimizer="adam", loss="mse")
        model.fit(X, y, epochs=epochs, batch_size=32, verbose=0)

        last_sequence = data[-seq_length:].reshape(1, seq_length, 1)
        prediction = model.predict(last_sequence, verbose=0)
        prediction = scaler.inverse_transform(prediction)
        if np.isnan(prediction).any():
            print("LSTM forecast contains NaN values")
            return np.array([])
        return prediction.flatten()
    except Exception as e:
        print(f"LSTM forecast failed: {e}")
        return np.array([])

# ------------------------------------Trade Simulation Function-------------------------------------- #

def simulate_trades(df, base_spread, forecast):
    """
    Simulate trades based on market data and forecasts.

    Parameters:
    df (DataFrame): The DataFrame containing market data.
    base_spread (float): The base spread for quotes.
    forecast (pd.Series): The forecasted prices.

    Returns:
    tuple: A tuple containing the P&L, inventory, trades, and P&L DataFrame.
    """
    mm = MarketMaker(base_spread=base_spread)
    df["mid_price"] = (df["bid_price"] + df["ask_price"]) / 2
    trades = []
    pnls = []
    timestamps = []
    forecast_change = (forecast[0] - df["mid_price"].iloc[-1]) / df["mid_price"].iloc[-1] if len(forecast) > 0 else 0
    
    for index, row in df.iterrows():
        bid, ask = mm.get_quotes(row["bid_price"], row["ask_price"], row["mid_price"], forecast_change)
        if bid > row["bid_price"] and ask < row["ask_price"]:
            if random.random() < mm.trade_probability:
                if random.random() < 0.5:
                    mm.execute_trade(bid, "buy", 1)
                    trades.append({"timestamp": row["timestamp"], "side": "buy", "price": bid, "volume": 1})
                else:
                    mm.execute_trade(ask, "sell", 1)
                    trades.append({"timestamp": row["timestamp"], "side": "sell", "price": ask, "volume": 1})
        timestamps.append(row["timestamp"])
        pnls.append(mm.pnl)
    
    return mm.pnl, mm.inventory, trades, pd.DataFrame({"timestamp": timestamps, "pnls": pnls})

# ------------------------------------Dash App Initialization-------------------------------------- #

# Initialize the Dash app
app = Dash(__name__)
server = app.server  # For Heroku deployment (to be updated later for new hosting)

# Load data
df = get_cached_data(use_cleaned=True)

# List of instruments (determined from data)
instruments = sorted(df["instrument"].unique())  # ['AAPL', 'AMZN', 'GOOG', 'META', 'MSFT']

# Precompute forecasts for all instruments
for instrument in instruments:
    df_filtered = df[df["instrument"] == instrument].tail(1000).copy()
    df_subset = df_filtered.tail(1000)  # Reduced to 1000 rows for faster computation
    
    # ARIMA forecast
    arima_pred = arima_forecast(df_subset, steps=10)
    arima_dates = pd.date_range(start=df_subset["timestamp"].iloc[-1], periods=11, freq="min")[1:] if not df_subset.empty else []
    arima_forecast_df = pd.DataFrame({"timestamp": arima_dates, "price": arima_pred, "type": "ARIMA"}) if len(arima_pred) > 0 else pd.DataFrame({"timestamp": [], "price": [], "type": "ARIMA"})
    forecast_cache["ARIMA"][instrument] = arima_forecast_df

    # LSTM forecast
    lstm_pred = lstm_forecast(df_subset, steps=10)
    lstm_forecast_df = pd.DataFrame({"timestamp": arima_dates, "price": lstm_pred, "type": "LSTM"}) if len(lstm_pred) > 0 else pd.DataFrame({"timestamp": [], "price": [], "type": "LSTM"})
    forecast_cache["LSTM"][instrument] = lstm_forecast_df

# ------------------------------------Initial Simulation-------------------------------------- #

# Initial simulation with default instrument (AAPL)
initial_instrument = "AAPL"
initial_spread = 0.0005
df_filtered = df[df["instrument"] == initial_instrument].tail(1000).copy()

# Get initial forecasts from cache
arima_forecast_df = forecast_cache["ARIMA"][initial_instrument]
lstm_forecast_df = forecast_cache["LSTM"][initial_instrument]
arima_dates = arima_forecast_df["timestamp"] if not arima_forecast_df.empty else []

# Initial simulation
forecast_value = (lstm_forecast_df["price"].iloc[0] if not lstm_forecast_df.empty else 
                 (arima_forecast_df["price"].iloc[0] if not arima_forecast_df.empty else 0))
pnl, inventory, trades, pnl_df = simulate_trades(df_filtered, base_spread=initial_spread, forecast=pd.Series([forecast_value]))

# Convert trades to DataFrame for plotting
trades_df = pd.DataFrame(trades)
if not trades_df.empty:
    trades_df["marker"] = trades_df["side"].map({"buy": "triangle-up", "sell": "triangle-down"})
    trades_df["color"] = trades_df["side"].map({"buy": "#00FF00", "sell": "#FF0000"})

# Initial plots with neon colors
df_filtered["mid_price"] = (df_filtered["bid_price"] + df_filtered["ask_price"]) / 2
fig_price = px.line(df_filtered, x="timestamp", y="mid_price", title=f"{initial_instrument} Mid Price Over Time")
fig_price.update_traces(line_color="#00BFFF", name="Mid Price")
if not trades_df.empty:
    fig_price.add_scatter(
        x=trades_df["timestamp"], y=trades_df["price"], mode="markers",
        marker=dict(size=10, symbol=trades_df["marker"], color=trades_df["color"]),
        name="Trades"
    )

# Debug: Print the forecast DataFrames
print("Initial ARIMA Forecast DF:", arima_forecast_df)
print("Initial LSTM Forecast DF:", lstm_forecast_df)

# Add forecasts to the price chart
if not arima_forecast_df.empty and not arima_forecast_df["price"].isna().all():
    fig_price.add_scatter(
        x=arima_forecast_df["timestamp"], y=arima_forecast_df["price"],
        mode="lines+markers", line=dict(color="#FFD700"), name="ARIMA Forecast"
    )
if not lstm_forecast_df.empty and not lstm_forecast_df["price"].isna().all():
    fig_price.add_scatter(
        x=lstm_forecast_df["timestamp"], y=lstm_forecast_df["price"],
        mode="lines+markers", line=dict(color="#FF00FF"), name="LSTM Forecast"
    )

fig_price.update_xaxes(range=[df_filtered["timestamp"].min(), arima_dates[-1] if len(arima_dates) > 0 else df_filtered["timestamp"].max()])

fig_pnl = px.line(pnl_df, x="timestamp", y="pnls", title="P&L Over Time")
fig_pnl.update_traces(line_color="#FF00FF")

# Layout with explanatory text and links
app.layout = html.Div(
    id="main-layout",
    children=[
        # Introduction Section
        html.H1("Market Maker Dashboard", style={"textAlign": "center"}),
        html.P(
            "This dashboard simulates a market making strategy for algorithmic trading. Select an instrument to view its price movements, forecasted prices, and the market maker's profit and loss (P&L) over time. Adjust the base spread to see how it affects trading.",
            style={"textAlign": "center", "fontStyle": "italic", "fontSize": "16px", "marginBottom": "20px"}
        ),

        # Dark Mode Toggle
        html.Div([
            html.Label("Dark Mode", id="dark-mode-label", style={"display": "inline-block", "marginRight": "10px", "fontSize": "16px"}),
            dcc.Checklist(
                id="dark-mode-toggle",
                options=[{"label": "", "value": "dark"}],
                value=["dark"],
                style={"display": "inline-block"}
            ),
            html.P(
                "Toggle between light and dark themes for better visibility.",
                style={"display": "inline-block", "fontStyle": "italic", "fontSize": "14px", "marginLeft": "10px"}
            ),
        ], style={"textAlign": "center", "marginBottom": "20px"}),

        # Instrument Dropdown
        html.Label("Select Instrument", style={"fontSize": "18px", "fontWeight": "bold"}),
        html.P(
            f"Choose an instrument to analyze: {', '.join(instruments)}.",
            style={"fontStyle": "italic", "fontSize": "14px", "marginBottom": "10px"}
        ),
        dcc.Dropdown(
            id="instrument-dropdown",
            options=[{"label": inst, "value": inst} for inst in instruments],
            value=initial_instrument,
            style={"width": "50%", "marginBottom": "20px"}
        ),

        # Price Chart
        html.H4("Price Chart", style={"fontSize": "16px", "fontWeight": "bold"}),
        html.P(
            "Shows the mid-price (blue) over the last 1,000 timestamps, trades executed by the market maker (green for buys, red for sells), and forecasted bid prices for the next 10 minutes using ARIMA (gold) and LSTM (purple) models.",
            style={"fontStyle": "italic", "fontSize": "14px", "marginBottom": "10px"}
        ),
        dcc.Graph(id="price-chart", figure=fig_price),

        # P&L Chart
        html.H4("Profit and Loss (P&L) Chart", style={"fontSize": "16px", "fontWeight": "bold"}),
        html.P(
            "Displays the cumulative profit and loss (P&L) of the market maker over time (purple).",
            style={"fontStyle": "italic", "fontSize": "14px", "marginBottom": "10px"}
        ),
        dcc.Graph(id="pnl-chart", figure=fig_pnl),

        # Metrics
        html.H4("Performance Metrics", style={"fontSize": "16px", "fontWeight": "bold"}),
        html.P(
            "P&L: Total profit/loss in dollars. Inventory: Net position (positive for buys, negative for sells).",
            style={"fontStyle": "italic", "fontSize": "14px", "marginBottom": "10px"}
        ),
        html.H3(id="metrics", children=f"P&L: {pnl:.2f}, Inventory: {inventory}"),

        # Base Spread Slider
        html.Label("Base Spread", style={"fontSize": "18px", "fontWeight": "bold"}),
        html.P(
            "Adjust the base spread (as a percentage of the mid-price) to control the market maker's bid-ask spread. A smaller spread increases trade frequency but reduces profit per trade.",
            style={"fontStyle": "italic", "fontSize": "14px", "marginBottom": "10px"}
        ),
        dcc.Slider(
            id="spread-slider",
            min=0.0001,
            max=0.0010,
            step=0.0001,
            value=initial_spread,
            marks={i: f"{i:.4f}" for i in np.arange(0.0001, 0.0011, 0.0003)}
        ),

        # Links Section
        html.H4("Additional Resources", style={"fontSize": "16px", "fontWeight": "bold", "marginTop": "20px"}),
        html.Div([
            # GitHub Link
            html.A(
                [
                    html.Img(
                        src="https://github.githubassets.com/favicon.ico",
                        style={"height": "20px", "marginRight": "5px", "verticalAlign": "middle"}
                    ),
                    "GitHub Repository"
                ],
                href="https://github.com/Sidhved/market-maker-backtest-platform",  # Replace with your GitHub URL
                target="_blank",
                style={"marginRight": "20px", "textDecoration": "none", "color": "#00BFFF"}
            ),
            # Jupyter Notebook Link
            html.A(
                [
                    html.Img(
                        src="https://jupyter.org/favicon.ico",
                        style={"height": "20px", "marginRight": "5px", "verticalAlign": "middle"}
                    ),
                    "EDA Notebook"
                ],
                href="https://github.com/Sidhved/market-maker-backtest-platform/blob/main/EDA_and_Model_Evaluation.ipynb",  # Replace with your notebook URL
                target="_blank",
                style={"textDecoration": "none", "color": "#00BFFF"}
            ),
        ], style={"textAlign": "center", "marginTop": "10px", "marginBottom": "20px"})
    ],
    style={"padding": "20px"}
)

# Custom CSS for the toggle switch
app.css.append_css({
    "external_url": "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css"
})

app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            /* Switch container */
            .switch {
                position: relative;
                display: inline-block;
                width: 60px;
                height: 34px;
            }

            /* Hide default checkbox */
            .switch input {
                opacity: 0;
                width: 0;
                height: 0;
            }

            /* The slider */
            .slider {
                position: absolute;
                cursor: pointer;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background-color: #ccc;
                transition: .4s;
                border-radius: 34px;
            }

            /* The slider circle */
            .slider:before {
                position: absolute;
                content: "";
                height: 26px;
                width: 26px;
                left: 4px;
                bottom: 4px;
                background-color: white;
                transition: .4s;
                border-radius: 50%;
            }

            /* When checked (dark mode on) */
            input:checked + .slider {
                background-color: #00FF00; /* Neon green */
            }

            input:checked + .slider:before {
                transform: translateX(26px);
            }

            /* Moon icon (light mode) */
            .slider .fas {
                position: absolute;
                top: 50%;
                transform: translateY(-50%);
                font-size: 20px;
                color: #333;
            }

            .slider .fa-moon {
                left: 10px;
            }

            /* Sun icon (dark mode) */
            .slider .fa-sun {
                right: 10px;
                display: none;
            }

            input:checked + .slider .fa-moon {
                display: none;
            }

            input:checked + .slider .fa-sun {
                display: block;
                color: #FFFFFF;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# ------------------------------------Callbacks-------------------------------------- #

# Callback for updating charts (spread and instrument changes)
@app.callback(
    [Output("price-chart", "figure"),
     Output("pnl-chart", "figure"),
     Output("metrics", "children")],
    [Input("spread-slider", "value"),
     Input("instrument-dropdown", "value")]
)
def update_charts(spread, selected_instrument):
    """
    Update the price and P&L charts based on selected instrument and spread.

    Parameters:
    spread (float): The selected spread value.
    selected_instrument (str): The selected instrument.

    Returns:
    tuple: Updated figures and metrics.
    """
    # Filter data for the selected instrument
    df_filtered = df[df["instrument"] == selected_instrument].tail(1000).copy()
    
    # Get forecasts from cache
    arima_forecast_df = forecast_cache["ARIMA"][selected_instrument]
    lstm_forecast_df = forecast_cache["LSTM"][selected_instrument]
    arima_dates = arima_forecast_df["timestamp"] if not arima_forecast_df.empty else []

    # Debug: Print the forecast DataFrames
    print("ARIMA Forecast DF:", arima_forecast_df)
    print("LSTM Forecast DF:", lstm_forecast_df)

    # Re-run simulation with new spread and instrument
    forecast_value = (lstm_forecast_df["price"].iloc[0] if not lstm_forecast_df.empty else 
                     (arima_forecast_df["price"].iloc[0] if not arima_forecast_df.empty else 0))
    pnl, inventory, trades, pnl_df = simulate_trades(df_filtered, spread, pd.Series([forecast_value]))
    
    # Update price chart
    trades_df = pd.DataFrame(trades)
    df_filtered["mid_price"] = (df_filtered["bid_price"] + df_filtered["ask_price"]) / 2
    fig_price = px.line(df_filtered, x="timestamp", y="mid_price", title=f"{selected_instrument} Mid Price Over Time")
    fig_price.update_traces(line_color="#00BFFF", name="Mid Price")
    if not trades_df.empty:
        trades_df["marker"] = trades_df["side"].map({"buy": "triangle-up", "sell": "triangle-down"})
        trades_df["color"] = trades_df["side"].map({"buy": "#00FF00", "sell": "#FF0000"})
        fig_price.add_scatter(
            x=trades_df["timestamp"], y=trades_df["price"], mode="markers",
            marker=dict(size=10, symbol=trades_df["marker"], color=trades_df["color"]),
            name="Trades"
        )
    # Add forecasts to the price chart
    if not arima_forecast_df.empty and not arima_forecast_df["price"].isna().all():
        fig_price.add_scatter(
            x=arima_forecast_df["timestamp"], y=arima_forecast_df["price"],
            mode="lines+markers", line=dict(color="#FFD700"), name="ARIMA Forecast"
        )
    if not lstm_forecast_df.empty and not lstm_forecast_df["price"].isna().all():
        fig_price.add_scatter(
            x=lstm_forecast_df["timestamp"], y=lstm_forecast_df["price"],
            mode="lines+markers", line=dict(color="#FF00FF"), name="LSTM Forecast"
        )
    
    fig_price.update_xaxes(range=[df_filtered["timestamp"].min(), arima_dates[-1] if len(arima_dates) > 0 else df_filtered["timestamp"].max()])

    # Update P&L chart
    fig_pnl = px.line(pnl_df, x="timestamp", y="pnls", title="P&L Over Time")
    fig_pnl.update_traces(line_color="#FF00FF")
    
    # Update metrics
    metrics = f"P&L: {pnl:.2f}, Inventory: {inventory}"
    
    return fig_price, fig_pnl, metrics

# Callback for updating styling (dark mode changes)
@app.callback(
    [Output("price-chart", "style"),
     Output("pnl-chart", "style"),
     Output("main-layout", "style"),
     Output("dark-mode-label", "style")],
    [Input("dark-mode-toggle", "value"),
     Input("price-chart", "figure"),
     Input("pnl-chart", "figure")]
)
def update_styles(dark_mode, price_fig, pnl_fig):
    """
    Update the styles of the charts and layout based on dark mode toggle.

    Parameters:
    dark_mode (list): The current state of the dark mode toggle.
    price_fig (dict): The current price chart figure.
    pnl_fig (dict): The current P&L chart figure.

    Returns:
    tuple: Updated styles for charts and layout.
    """
    # Convert figures back to Plotly objects
    fig_price = price_fig
    fig_pnl = pnl_fig

    chart_style = {}
    layout_style = {"padding": "20px"}
    label_style = {"display": "inline-block", "marginRight": "10px", "fontSize": "16px"}
    
    if "dark" in dark_mode:
        layout_style.update({
            "backgroundColor": "#1E1E1E",
            "color": "#FFFFFF"
        })
        chart_style.update({
            "backgroundColor": "#1E1E1E",
            "color": "#FFFFFF"
        })
        label_style.update({"color": "#FFFFFF"})
        fig_price["layout"].update({
            "plot_bgcolor": "#1E1E1E",
            "paper_bgcolor": "#1E1E1E",
            "font": {"color": "#FFFFFF"},
            "xaxis": {"gridcolor": "#333333"},
            "yaxis": {"gridcolor": "#333333"}
        })
        fig_pnl["layout"].update({
            "plot_bgcolor": "#1E1E1E",
            "paper_bgcolor": "#1E1E1E",
            "font": {"color": "#FFFFFF"},
            "xaxis": {"gridcolor": "#333333"},
            "yaxis": {"gridcolor": "#333333"}
        })
    else:
        layout_style.update({
            "backgroundColor": "white",
            "color": "black"
        })
        chart_style.update({
            "backgroundColor": "white",
            "color": "black"
        })
        label_style.update({"color": "black"})
        fig_price["layout"].update({
            "plot_bgcolor": "white",
            "paper_bgcolor": "white",
            "font": {"color": "black"},
            "xaxis": {"gridcolor": "lightgray"},
            "yaxis": {"gridcolor": "lightgray"}
        })
        fig_pnl["layout"].update({
            "plot_bgcolor": "white",
            "paper_bgcolor": "white",
            "font": {"color": "black"},
            "xaxis": {"gridcolor": "lightgray"},
            "yaxis": {"gridcolor": "lightgray"}
        })
    
    # Update the figures with the new styling
    return chart_style, chart_style, layout_style, label_style

# ------------------------------------Run the App-------------------------------------- #

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8050))
    app.run_server(debug=True, host="0.0.0.0", port=port)