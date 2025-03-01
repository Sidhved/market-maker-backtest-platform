# Market Maker Simulation & Algorithmic Trading Backtesting Platform

## Overview
The Market Maker Simulation & Algorithmic Trading Backtesting Platform is a web-based application that simulates a market making strategy for algorithmic trading. The platform allows users to backtest a market making algorithm using historical tick data for five major tech stocks (AAPL, AMZN, GOOG, META, MSFT). It provides an interactive dashboard to visualize price movements, forecasted prices using ARIMA and LSTM models, and the market maker’s profit and loss (P&L) over time. Users can adjust the base spread to explore its impact on trading behavior.

The project includes data ingestion, retrieval, validation, forecasting, backtesting, and visualization components, all integrated into a Dash-based web dashboard.

## Features
- **Interactive Dashboard**: Visualize mid-price movements, trades, and P&L over time for selected instruments.
- **Instrument Selection**: Choose from five tech stocks: AAPL, AMZN, GOOG, META, MSFT.
- **Price Forecasting**: View 10-minute bid price forecasts using ARIMA (gold) and LSTM (purple) models.
- **Adjustable Base Spread**: Modify the market maker’s base spread (0.01% to 0.1%) to see its effect on trade frequency and profitability.
- **Dark Mode Toggle**: Switch between light and dark themes for better visibility.
- **Performance Metrics**: Monitor the market maker’s total P&L and inventory in real-time.
- **EDA and Model Evaluation**: A Jupyter notebook (`EDA_and_Model_Evaluation.ipynb`) provides detailed exploratory data analysis (EDA) and performance metrics (MAE, RMSE) for the ARIMA and LSTM models.

## Architecture
The project is structured as follows:

- **`dashboard.py`**: The main Dash application script that defines the dashboard layout, callbacks, and visualization logic.
- **`data_retrieval.py`**: Handles data retrieval from Supabase, with caching support using Redis.
- **`data_ingestion.py`**: Ingests historical tick data into Supabase (used during initial setup).
- **`data_validation.py`**: Validates and preprocesses the data (e.g., resampling to 1-minute intervals, removing duplicates).
- **`forecasting.py`**: Contains the ARIMA and LSTM forecasting logic (integrated into `dashboard.py`).
- **`backtesting.py`**: Implements the market making strategy and backtesting logic (integrated into `dashboard.py`).
- **`EDA_and_Model_Evaluation.ipynb`**: A Jupyter notebook for exploratory data analysis (EDA) and model evaluation (ARIMA and LSTM metrics).
- **`requirements.txt`**: Lists all Python dependencies required to run the project.
- **`Procfile`**: Defines the web process for deployment (e.g., on Heroku or another platform).
- **`.env`**: Stores environment variables (e.g., Supabase URL and key) for local development (not committed to Git).

## Prerequisites
To run this project locally, you’ll need the following:

- **Python 3.12+**: Ensure Python is installed on your system.
- **Supabase Account**: A Supabase project with a `tick_data` table containing the historical data.
- **Redis**: For caching data (optional but recommended for performance).
- **Jupyter Notebook**: To run the EDA notebook (optional).
- **Git**: For cloning the repository.

## Setup Instructions
Follow these steps to set up the project locally:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Sidhved/market-maker-backtest-platform.git
   cd market-maker
   ```
2. **Set Up a Virtual Environment**:
    ```bash
    python -m venv market_maker_env
    source market_maker_env/bin/activate  # On Windows: market_maker_env\Scripts\activate
    ```
3. **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
4. Configure Environment Variables:
- Create a `.env` file in the project root:
    ```bash
    touch .env
    ```
- Add your Supabase credentials (obtain these from your Supabase dashboard under Settings > API):
    ```bash
    SUPABASE_URL=https://your-supabase-url.supabase.co
    SUPABASE_KEY=your-supabase-anon-key
    ```
- If using Redis locally, add Redis configuration (optional):
    ```bash
    REDIS_URL=redis://localhost:6379/0
    ```
5. **Run Redis Locally (optional, for caching)**:
- Install Redis on your system (e.g., via Homebrew on macOS: `brew install redis`).
- Start the Redis server:
    ```bash
    redis-server
    ```
6. ***Run the Dashboard**:
    ```bash
    python dashboard.py
    ```
- Open your browser and navigate to `http://127.0.0.1:8050` to view the dashboard.
7. **Run the EDA Notebook (optional)**:
- Start Jupyter Notebook:
    ```bash
    jupyter notebook
    ```
- Open `EDA_and_Model_Evaluation.ipynb` and run the cells to perform EDA and evaluate the models.

## Usage
1. **Access the Dashboard**:
- Navigate to `http://127.0.0.1:8050` after running `dashboard.py`.
2. **Interact with the Dashboard**:
- **Select Instrument**: Choose a stock (AAPL, AMZN, GOOG, META, MSFT) from the dropdown to view its data.
- **View Price Chart**: See the mid-price (blue), trades (green for buys, red for sells), and 10-minute forecasts (gold for ARIMA, purple for LSTM).
- **View P&L Chart**: Monitor the market maker’s cumulative profit and loss over time (purple).
- **Check Metrics**: Review the total P&L and inventory (net position).
- **Adjust Base Spread**: Use the slider to change the base spread (0.01% to 0.1%), affecting trade frequency and profitability.
- **Toggle Dark Mode**: Switch between light and dark themes for better visibility.
3. **Run the EDA Notebook (optional)**:
- Open `EDA_and_Model_Evaluation.ipynb` in Jupyter Notebook.
- Run the cells to explore the dataset, evaluate ARIMA and LSTM models, and compare their performance using MAE and RMSE.

## Data Source
- **Source**: Historical tick data for five tech stocks (AAPL, AMZN, GOOG, META, MSFT) stored in a Supabase database.
- **Table**: `tick_data` with columns: `timestamp`, `instrument`, `bid_price`, `ask_price`, `volume`.
- **Preprocessing**: The data has been resampled to 1-minute intervals, with duplicates removed and gaps filled using forward fill (`data_validation.py`).
- **Access**: Retrieved via `data_retrieval.py`, with caching in Redis to improve performance.

## Model Details
The dashboard uses two models to forecast bid prices for the next 10 minutes:

- **ARIMA (AutoRegressive Integrated Moving Average)**:
    - Order: (5,1,0).
    - Fits a statistical model to the bid price time series.
    - Forecasts are displayed in gold on the price chart.
    - Computation: Performed on the last 1,000 rows of data for each instrument.
- **LSTM (Long Short-Term Memory)**:
    - Architecture: 50 LSTM units, 1 dense layer, 5 epochs.
    - Sequence Length: 50 minutes.
    - Forecasts are displayed in purple on the price chart.
    - Computation: Performed on the last 1,000 rows of data for each instrument, using scaled data (MinMaxScaler).
- **Evaluation**: The `EDA_and_Model_Evaluation.ipynb` notebook provides detailed metrics (MAE, RMSE) and visualizations comparing actual vs. forecasted bid prices for both models.

## Deployment
Deployment instructions will be added after switching hosting platforms (to be determined in the next step). Currently, the project can be run locally as described in the Setup Instructions.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Contact
For questions or feedback, please contact the project maintainers:

Email: sidhved.warik@gmail.com

---
*Built with ❤️ by Sidhved*