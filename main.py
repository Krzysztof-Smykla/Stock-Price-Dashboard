import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tkinter as tk
import os

# set the working directory to the location of the script
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Sample dataset
data = pd.read_csv('stock_price.csv')

print(data.head())
data['date'] = pd.to_datetime(data['date'])
data.set_index('date', inplace=True)

aal_data = data[data['Name'] == 'AAL']
aal_data['close'].plot(figsize=(12, 6))
plt.title('AAL Stock Price Over Time')
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()




# TODO: Implement the following APIs:
# Stock Price API (https://finnhub.io/docs/api/websocket-trades):
# Provides access to real-time stock market prices for companies in every major exchange around the world.
# Price data is delayed by 15 minutes

# Historical Stock Price API (Yahoo Finance https://ranaroussi.github.io/yfinance/):
# Provides access to historical stock price data for companies in every major exchange around the world.
# Historical price data is available for up to 20 years.






