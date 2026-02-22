# Stock-Price-Dashboard
A data-driven dashboard for tracking and predicting the value of the S&P 500 index and its listed companies.

## Project Description
The program will use historic data of the S&P 500 index as well as live data sourced via an API. It will then make a prediction of the future price using a regression model.
The user will be able to choose whether to predict an individual company or the entire index. A significant constraint of this project is the nature of the stock price data - it is highly volatile and influenced by many factors. Predictions made for the entire index might be more accurate since the index reflects the aggregated value of all listed companies and its value is more stable over time. 

## Project Overview

This project builds an interactive financial dashboard that:

- Retrieves historical stock market data  
- Fetches live market data via an API  
- Applies regression-based machine learning models  
- Predicts future price movements  

Users can choose to forecast:

- The full S&P 500 index  
- An individual company listed in the index  

---

## Objectives

- Visualize historical stock/index trends  
- Compare volatility between individual companies and the index  
- Generate short-term future price predictions  
- Provide an intuitive and interactive user interface  

---

## Core Features

### 1. Data Collection

- Historical price data (Open, High, Low, Close, Volume)  
- Live market data via financial APIs (e.g., Yahoo Finance, Alpha Vantage)

### 2. Data Processing

- Data cleaning and normalization  
- Feature engineering:
  - Moving averages  
  - Daily returns  
  - Volatility measures  
- Handling missing or inconsistent values  

### 3. Prediction Models

- Regression-based models:
  - Linear Regression  
  - Ridge / Lasso  
  - Random Forest  
  - (Optional) LSTM for time-series forecasting  

- Model evaluation metrics:
  - RMSE  
  - MAE  
  - R² score  

### 4. Interactive Dashboard

- Historical price visualization  
- Predicted vs actual price comparison  
- User inputs:
  - Company ticker selection  
  - Prediction horizon  
  - Model selection (optional advanced feature)  

---

## Key Challenge

Stock market data is:

- Highly volatile  
- Influenced by macroeconomic and geopolitical factors  
- Non-stationary over time  

Predictions for the S&P 500 index may be more stable than individual stocks because:

- The index reflects aggregated market performance  
- It smooths extreme fluctuations of single companies  
- It captures broader economic trends  

---

## Planned Tech Stack

### Backend / Data Processing
- Python  
- pandas  
- numpy  
- scikit-learn  
- statsmodels (optional)  
- TensorFlow / PyTorch (for LSTM models) *optional 

### Data APIs
- Yahoo Finance API  (yfinance)
- Alpha Vantage
- Finnhub

### Dashboard Framework
- Streamlit (recommended for simplicity)  
- Dash (Plotly)  
- Flask + React (advanced architecture)  

---

## Possible Extensions

- Technical indicators:
  - RSI  
  - MACD  
  - Bollinger Bands  

- Add macroeconomic variables:
  - Interest rates  
  - Inflation  
  - GDP indicators  

- Backtesting module  
- Portfolio simulation  
- Multi-company comparison  
- Model performance tracking  

---
