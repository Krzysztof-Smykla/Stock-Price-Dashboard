import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
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
training = int(np.ceil(len(dataset) * 0.8))
print("TRAINING DATASET LENGTH: ", training)

# Scaling the dataset to a fixed range (0, 1):
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(dataset)

train_data = scaled_data[0:int(training), :]
# prepare feature and labels
x_train = []
y_train = []

# Splitting the training subset equally into x_train and y_train 
for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])

x_train = np.array(x_train)
y_train = np.array(y_train)

print(f"X Train: {x_train.shape}, ", f"Y Train:{y_train.shape}")

# Building The Regression Prediction Model
# Ridge regression introduces a regularization term that shrinks the coefficients, stabilizing predictions.
# It is a version of linear regression that adds an L2 penalty to control large coefficient values. 
# https://www.geeksforgeeks.org/machine-learning/what-is-ridge-regression/# 

# Using the scaled training data to train a ridge regression model
from sklearn.linear_model import Ridge

# Define the regression model as a class
class RegressionModel:
    def __init__(self, model):
        self.model = model

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def coefficients(self):
        return self.model.coef_

    def intercept(self):
        return self.model.intercept_
    
ridge = Ridge(alpha=0.7)

# The Ridge model instance
m = RegressionModel(ridge)
m.fit(x_train, y_train)

print("Coefficients:", m.coefficients())
print("Intercept:", m.intercept())

# Create the testing dataset
test_data = scaled_data[training - 60:, :]

x_test = []
y_test = dataset[training:, :]  # NOT scaled (for final comparison)

for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])

x_test = np.array(x_test)

print("X Test:", x_test.shape)
print("Y Test:", y_test.shape)

# Price Prediction
predictions = m.predict(x_test)
predictions = predictions.reshape(-1, 1)
predictions = scaler.inverse_transform(predictions)

# Evaluation Metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

rmse = np.sqrt(mean_squared_error(y_test, predictions))
mae = mean_absolute_error(y_test, predictions)

# Prediction error
print("RMSE:", rmse)
print("MAE:", mae)

# Training error
train_preds = m.predict(x_train)
train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
train_mae = mean_absolute_error(y_train, train_preds)

print("Train RMSE:", train_rmse)
print("Train MAE:", train_mae)

# Prediction visualization

train = apple[:training]
valid = apple[training:].copy()
valid['Predictions'] = predictions

plt.figure(figsize=(14, 7))

plt.plot(train['date'], train['close'], label='Train')
plt.plot(valid['date'], valid['close'], label='Actual')
plt.plot(valid['date'], valid['Predictions'], label='Predicted')

plt.title("Apple Stock Price Prediction (Ridge Regression)")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.legend()
plt.show()
