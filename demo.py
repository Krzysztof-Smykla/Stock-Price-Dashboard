import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
# import pytorch
import os
from datetime import datetime

import warnings
warnings.filterwarnings("ignore")

# set the working directory to the location of the script
os.chdir(os.path.dirname(os.path.abspath(__file__)))

data = pd.read_csv('stock_price.csv', delimiter=',', on_bad_lines='skip')
print(data.shape)
print(data.sample(7))

data.info()

# Rename data columns to lowercase
data.columns = map(str.lower, data.columns)

# converting the 'date' field to a datetime data type
data['date'] =pd.to_datetime(data['date'])
data.info()

# Data exploration
companies = ['AAPL', 'AMD', 'FB', 'GOOGL', 'AMZN', 'NVDA', 'BBT','CSCO', 'IBM']


# Figure 1: Open vs Close Price Plot
plt.figure(figsize=(15, 8))
for index, company in enumerate(companies, 1):
    plt.subplot(3, 3, index)
    c = data[data["name"] == company]
    plt.plot(c['date'], c['close'], c="r", label="close", marker="+")
    plt.plot(c['date'], c['open'], c="g", label="open", marker="^")
    plt.title(company)
    plt.legend()
    plt.tight_layout()

# Figure 2: Sale Volume Plot
plt.figure(figsize=(15, 8))
for index, company in enumerate(companies, 1):
    plt.subplot(3, 3, index)
    c = data[data['name'] == company]
    plt.plot(c['date'], c['volume'], c='purple', marker='*')
    plt.title(f"{company} volume")
    plt.tight_layout()
plt.show()

# Apple Stocks from 2013 to 2018
apple = data[data['name'] == "AAPL"]
prediction_range = apple[
    apple['date'].between(datetime(2013, 1, 1), datetime(2018, 1, 1))
]
plt.plot(apple['date'], apple['close'])
plt.xlabel("Date")
plt.ylabel("Close")
plt.title("Apple Stock Prices")
plt.show()

close_data = apple.filter(['close'])
dataset = close_data.values
training = int(np.ceil(len(dataset) * .95))
print("Training dataset:", training)

# Building The Regression Prediction Model:


