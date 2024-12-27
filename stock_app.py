import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
from helper import ExtractData
from regression_model import StockPredictor

# Default configuration constants

# Number of days for regression models (1 year of data)
num_days = 252
future_days = 1

# Historical data range for LSTM model (over 10 years)
start = dt.datetime(2011, 1, 1)
end = dt.datetime.now()

# Use the past 60 days as input for the LSTM model
prediction_days = 60

# List of stock tickers for user selection (can be extended as needed)
stock_tickers = ['AAPL', 'AMZN', 'FB', 'INTC', 'MSFT', 'HPQ', 'ZM', 'NFLX', 'TWTR', 'GS',
                 'MS', 'NVDA', 'MRNA', 'GOOG', 'HPQ', 'BA', 'AMC', 'DIS', 'SBUX', 'TSLA']

# Sidebar for user input: Stock selection
st.sidebar.header('User Input Features')
selected_ticker = st.sidebar.selectbox('Stock Ticker', stock_tickers)

# Sidebar for user input: Model selection
model_name = ['Linear', 'DecisionTree', 'SVM', 'LSTM']
selected_model = st.sidebar.multiselect('Model', model_name, model_name[0])

# Main interface heading and data source description
st.write(
    """
    # Stock Predictor for the Next Day

    * **Data source:** [Yahoo Finance](https://finance.yahoo.com/).

    ## **Stock Historical Closing Price**
    """
)

# Function to load historical stock data for the selected ticker
@st.cache_data
def load_data(ticker):
    """
    Fetch historical stock data for the selected ticker.

    Parameters:
        ticker (str): The stock ticker symbol.

    Returns:
        data (pd.DataFrame): Historical stock data with daily intervals.
    """
    data = ExtractData(ticker, start, end)
    return data

data = load_data(selected_ticker)

# Visualization of the stock's historical closing prices
st.line_chart(data.Close)

st.write(
    """
    ## **Model Prediction**

    Forecast stock price changes and provide trading suggestions using machine learning models.
    """
)

# Iterate through selected models and display predictions
for model_name in selected_model:
    if model_name == 'LSTM':
        st.text('This might take a few minutes!')
    model, prediction = StockPredictor(model_name, data)
