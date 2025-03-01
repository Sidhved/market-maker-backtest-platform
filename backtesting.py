from data_retrieval import get_cached_data
import pandas as pd
import numpy as np
import random

# -----------------------------------Market Maker Class---------------------------------------------- #

class MarketMaker:
    def __init__(self, base_spread=0.0005, forecast_adjustment_factor=0.1, inventory_limit=100, volatility_window=60, trade_probability=0.5):
        """
        Initialize the MarketMaker instance.

        Parameters:
        base_spread (float): The base spread for market making.
        forecast_adjustment_factor (float): Factor to adjust the forecast.
        inventory_limit (int): Maximum inventory limit.
        volatility_window (int): Window size for calculating volatility.
        trade_probability (float): Probability of executing a trade when quotes are competitive.
        """
        self.base_spread = base_spread
        self.forecast_adjustment_factor = forecast_adjustment_factor
        self.inventory_limit = inventory_limit
        self.volatility_window = volatility_window
        self.trade_probability = trade_probability
        self.inventory = 0
        self.pnl = 0
        self.mid_prices = []
        self.volatilities = []  # Track volatility for logging

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
            vol = np.std(returns) * np.sqrt(252 * 1440)  # Annualized volatility (1-min data)
            self.volatilities.append(vol)
            return vol
        return 0.1  # Default volatility if not enough data

    def get_quotes(self, market_bid, market_ask, mid_price, forecast_change):
        """
        Generate quotes based on market conditions.

        Parameters:
        market_bid (float): The current market bid price.
        market_ask (float): The current market ask price.
        mid_price (float): The current mid price.
        forecast_change (float): The change in forecast.

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
        Execute a trade and update inventory and P&L.

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

# -----------------------------------Simulate Trades Function---------------------------------------- #

def simulate_trades(df, mm, forecast):
    """
    Simulate trades based on market data and a market maker.

    Parameters:
    df (pd.DataFrame): The DataFrame containing market data.
    mm (MarketMaker): The MarketMaker instance.
    forecast (pd.Series): The forecast data.

    Returns:
    tuple: A tuple containing the P&L, inventory, and list of trades executed.
    """
    df["mid_price"] = (df["bid_price"] + df["ask_price"]) / 2
    trades = []
    forecast_change = (forecast[0] - df["mid_price"].iloc[-1]) / df["mid_price"].iloc[-1]
    
    for index, row in df.iterrows():
        bid, ask = mm.get_quotes(row["bid_price"], row["ask_price"], row["mid_price"], forecast_change)
        # Log quotes and market prices for debugging
        if index % 1000 == 0:  # Log every 1000 rows
            print(f"Timestamp: {row['timestamp']}, Mid Price: {row['mid_price']:.2f}, "
                  f"Market Bid: {row['bid_price']:.2f}, Market Ask: {row['ask_price']:.2f}, "
                  f"MM Bid: {bid:.2f}, MM Ask: {ask:.2f}, Inventory: {mm.inventory}, "
                  f"Volatility: {mm.volatilities[-1] if mm.volatilities else 0.1:.4f}")
        # Simulate trades: if MM quotes are competitive (inside market spread), execute with probability
        if bid > row["bid_price"] and ask < row["ask_price"]:
            # Competitive quotes: simulate order flow
            if random.random() < mm.trade_probability:
                # Randomly decide to buy or sell
                if random.random() < 0.5:  # 50% chance to buy
                    mm.execute_trade(bid, "buy", 1)
                    trades.append({"timestamp": row["timestamp"], "side": "buy", "price": bid, "volume": 1})
                else:  # 50% chance to sell
                    mm.execute_trade(ask, "sell", 1)
                    trades.append({"timestamp": row["timestamp"], "side": "sell", "price": ask, "volume": 1})
    return mm.pnl, mm.inventory, trades

# -----------------------------------Find Spread Range Function-------------------------------------- #

def find_spread_range(df, forecast):
    """
    Find the range of spreads where trades occur.

    This function tests various spreads and counts the number of trades executed for each spread.

    Parameters:
    df (pd.DataFrame): The DataFrame containing market data.
    forecast (pd.Series): The forecast data.

    Returns:
    float: The maximum spread with trades.
    """
    spread_values = np.arange(0.0001, 0.0101, 0.0001)  # Test from 0.01% to 1% in 0.01% increments
    trade_counts = []
    
    for spread in spread_values:
        mm = MarketMaker(base_spread=spread)
        df["mid_price"] = (df["bid_price"] + df["ask_price"]) / 2
        trades = []
        forecast_change = (forecast[0] - df["mid_price"].iloc[-1]) / df["mid_price"].iloc[-1]
        
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
        trade_counts.append(len(trades))
        print(f"base_spread: {spread:.4f}, Trades: {len(trades)}")
    
    # Find the range where trades occur
    max_spread_with_trades = max([spread for spread, count in zip(spread_values, trade_counts) if count > 0], default=0.0001)
    return max_spread_with_trades

# -----------------------------------Main Execution Block------------------------------------------- #

if __name__ == "__main__":
    print("Loading data...")
    df = get_cached_data(use_cleaned=True)
    print(f"Data loaded. Total rows: {len(df)}")

    # Filter for a single instrument (e.g., AAPL)
    df_aapl = df[df["instrument"] == "AAPL"].copy()
    print(f"AAPL data. Total rows: {len(df_aapl)}")

    # Use the ARIMA forecast from previous step
    arima_forecast = pd.Series([237.291800])
    mm = MarketMaker(base_spread=0.0005, forecast_adjustment_factor=0.1, inventory_limit=100, trade_probability=0.5)
    print("\n--- Simulating Trades ---")
    pnl, inventory, trades = simulate_trades(df_aapl, mm, arima_forecast)
    print(f"P&L: {pnl:.2f}, Inventory: {inventory}")
    print(f"Total trades executed: {len(trades)}")
    if trades:
        print("Sample of trades:")
        print(trades[:5])