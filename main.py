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

# Collect historical training data from yfinance API
import yfinance as yf
# Fetching Microsoft ticker from yfinance 
msft = yf.Ticker('MSFT')
msft.info

print("Microsoft Info ", msft.info)
print("Microsoft Income statement ",pd.DataFrame(msft.income_stmt))

# A test visualization
msft.history(period='1mo').plot(y='Close', use_index=True, title='Microsoft Stock Price')
plt.show()

# Fetch historical stock price data for the S&P 500 index
sp500 = yf.Ticker("^GSPC")
sp500_history = sp500.history(period='3mo')
# Visualize the historical data for the S&P 500 index
sp500_history.plot(y='Close', use_index=True, title='S&P 500 Stock Price')
plt.show()

# Importing all S&P 500 tickers from Wikipedia using pandas. 
# This way we can get the most up-to-date list of S&P 500 constituents 
# without relying on an API that may have limitations or require authentication. 
# The Wikipedia page is regularly updated and provides a comprehensive list of the companies included in the S&P 500 index.

