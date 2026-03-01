import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tkinter as tk
import os
import time

# set the working directory to the location of the script
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# TODO: Implement the following APIs:
# Stock Price API (https://finnhub.io/docs/api/websocket-trades):
# Provides access to real-time stock market prices for companies in every major exchange around the world.
# Price data is delayed by 15 minutes

# Historical Stock Price API (Yahoo Finance https://ranaroussi.github.io/yfinance/):
# Provides access to historical stock price data for companies in every major exchange around the world.
# Historical price data is available for up to 20 years.

# Dynamic Data Fetching:
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import logging
from data_scraper import configure_logging, create_session, fetch_sp500_tickers


def download_training_data(ticker: str, forecast_days: int):
    """
    Dynamically download enough data for training
    based on prediction horizon
    """
    history_days = calculate_training_range(forecast_days)

    end_date = datetime.today()
    start_date = end_date - timedelta(days=int(history_days * 1.5))
    # The 1.5 multiplier compensates for weekends/holidays in the US

    df = yf.download(
        ticker,
        start=start_date.strftime("%Y-%m-%d"),
        end = end_date.strftime("%Y-%m-%d"),
        interval="1d",
        progress=False
    )

    if df.empty:
        raise ValueError(f"No data found for {ticker}.")
    
    return df.tail(history_days)  # Return only the required training window

def calculate_training_range(forecast_days: int) ->  int:
    """
    Determine the required historical window size for training.
    252 trading days ≈ 1 trading year,
    8× forecast window gives model signal depth prevents underfitting on short prediction horizons.
    """
    return max(252, forecast_days * 8)  
# At least 1 year of data (252 trading days) or 8X the forecast horizon

def time_series_split(df, train_ratio=0.8):
    """
    Split the data into training and testing sets 
    while preserving time series order.
    """
    split_index = int(len(df) * train_ratio)
    train = df.iloc[:split_index]
    val = df.iloc[split_index:]
    return train, val

# Feature Engineering:



# ---------------------------------------------------------------
# Main Execution

configure_logging()

# Initialize a session with retry logic for robust web scraping.
# This session will be used to fetch the list of S&P 500 tickers from wikipedia.
create_session() 

tickers = fetch_sp500_tickers()
print(tickers[:10])  # Print the first 10 tickers to verify
