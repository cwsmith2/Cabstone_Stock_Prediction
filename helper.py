import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
from sklearn.preprocessing import MinMaxScaler

def ExtractData(ticker, start='2011-01-01', end=dt.datetime.now()):
    """
    Fetch historical stock data for a given ticker.

    Parameters:
        ticker (str): Stock ticker symbol.
        start (datetime): Start date for the data retrieval (default: '2011-01-01').
        end (datetime): End date for the data retrieval (default: current date).

    Returns:
        data (pd.DataFrame): Historical stock data with daily intervals.
    """
    ticker = yf.Ticker(ticker)
    data = ticker.history(start=start, end=end, interval="1d")

    return data

def ScaleData(data):
    """
    Scale numerical data to the range (0, 1).

    Parameters:
        data (np.ndarray): Array of numerical values to be normalized.

    Returns:
        scaler (MinMaxScaler): Fitted scaler instance for reverse scaling.
        scaled_data (np.ndarray): Normalized data in the range (0, 1).
    """
    data = data.reshape(-1, 1)  # Reshape data for compatibility with the scaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    return scaler, scaled_data

def InverseScale(scaler, data):
    """
    Reverse normalization to retrieve the original scale.

    Parameters:
        scaler (MinMaxScaler): The scaler instance used for initial normalization.
        data (np.ndarray): Normalized data to be converted back to original values.

    Returns:
        price (np.ndarray): Data transformed back to its original scale.
    """
    price = scaler.inverse_transform(data)
    return price
