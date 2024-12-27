import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import sklearn
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from helper import ScaleData, InverseScale

# LSTM model configuration parameters
lstm_size = 64
dropout = 0.2
dense_size = 10
epoch = 20
batch_size = 64

def StockPredictor(model_name, data, num_days=252, future_days=1, prediction_days=60):
    """
    Train the specified model and predict the stock price for the next day.

    Parameters:
        model_name (str): Name of the model to use ('Linear', 'DecisionTree', 'SVM', 'LSTM').
        data (pd.DataFrame): Historical stock price data.
        num_days (int): Number of days of historical data to use for regression models.
        future_days (int): Number of days ahead to predict.
        prediction_days (int): Number of days used as input for LSTM.

    Returns:
        model: Trained model instance.
        prediction (float): Predicted stock price for the next day.
    """
    data_close = data.Close.values.reshape(-1, 1)

    if model_name in ['Linear', 'DecisionTree', 'SVM']:
        # Prepare data for regression models
        data_reg = data_close[-num_days:]
        X = data_reg[:-future_days]
        y = data_reg[future_days:]
        x_next_day = data_reg[-future_days].reshape(-1, 1)

        # Split data into training and testing sets
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        # Train the specified regression model
        if model_name == 'Linear':
            model = LinearRegression().fit(x_train, y_train)
        elif model_name == 'DecisionTree':
            model = DecisionTreeRegressor(max_depth=8).fit(x_train, y_train)
        elif model_name == 'SVM':
            model = SVR(kernel='poly', C=0.001).fit(x_train, y_train.ravel())

        # Predict the next day's stock price
        prediction = model.predict(x_next_day).ravel()[0]

        # Evaluate model performance
        y_fit, y_pred = print_metrics(model_name, model, x_train, y_train, x_test, y_test)
        pred_last = y_pred.ravel()[-1]

    elif model_name == 'LSTM':
        # Normalize data
        scaler, scaled_data = ScaleData(data_close)
        x_train, x_test, y_train, y_test = train_test_split_lstm(scaled_data, prediction_days)
        x_next_day = np.array([scaled_data[len(scaled_data) - prediction_days:, 0]])
        x_next_day = np.reshape(x_next_day, (x_next_day.shape[0], x_next_day.shape[1], 1))

        # Build and train the LSTM model
        input_size = (x_train.shape[1], x_train.shape[2])
        model = create_model(input_size, lstm_size, dropout, dense_size)
        model.fit(x_train, y_train, epochs=epoch, batch_size=batch_size, validation_data=(x_test, y_test), verbose=0)

        # Predict the next day's stock price
        prediction = InverseScale(scaler, model.predict(x_next_day)).ravel()[0]

        # Evaluate model performance
        y_fit, y_pred = print_metrics(model_name, model, x_train, y_train, x_test, y_test, scaler)
        pred_last = y_pred.ravel()[-1]

    else:
        st.error("Invalid model name! Please choose a valid model.")
        return None, None

    # Provide trading suggestions
    Trade_suggest(pred_last, prediction)

    # Visualize recent trends
    Visualize_recent_trend(model_name, data, y_pred.ravel())

    return model, prediction

def train_test_split_lstm(data, prediction_days, train_size=0.9):
    """
    Split data for LSTM model training and testing in sequence order.

    Parameters:
        data (np.ndarray): Historical stock price data.
        prediction_days (int): Number of previous days used as input.
        train_size (float): Proportion of data for training.

    Returns:
        x_train, x_test, y_train, y_test: Training and testing datasets.
    """
    n_train = int(len(data) * train_size)
    x_train, y_train, x_test, y_test = [], [], [], []

    for i in range(prediction_days, n_train):
        x_train.append(data[i - prediction_days:i, 0])
        y_train.append(data[i, 0])

    for j in range(n_train, len(data)):
        x_test.append(data[j - prediction_days:j, 0])
        y_test.append(data[j, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_test, y_test = np.array(x_test), np.array(y_test)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    return x_train, x_test, y_train, y_test

def create_model(input_size, lstm_size=64, dropout=0.2, dense_size=10):
    """
    Define and compile an LSTM model with specified parameters.

    Parameters:
        input_size (tuple): Shape of the input data.
        lstm_size (int): Number of units in LSTM layers.
        dropout (float): Dropout rate for regularization.
        dense_size (int): Number of units in dense layers.

    Returns:
        model (Sequential): Compiled LSTM model.
    """
    model = Sequential()
    model.add(LSTM(units=lstm_size, return_sequences=True, input_shape=input_size))
    model.add(Dropout(dropout))
    model.add(LSTM(units=lstm_size, return_sequences=False))
    model.add(Dropout(dropout))
    model.add(Dense(units=dense_size))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def print_metrics(model_name, model, x_train, y_train, x_test, y_test, scaler=None):
    """
    Calculate and display model performance metrics.

    Parameters:
        model_name (str): Name of the model.
        model: Trained model instance.
        x_train, y_train: Training input and output.
        x_test, y_test: Testing input and output.
        scaler (MinMaxScaler, optional): Scaler for LSTM model.

    Returns:
        y_fit, y_pred: Predicted outputs for training and testing data.
    """
    y_fit = model.predict(x_train)
    y_pred = model.predict(x_test)

    if model_name == 'LSTM':
        y_fit = InverseScale(scaler, y_fit)
        y_pred = InverseScale(scaler, y_pred)
        y_train = InverseScale(scaler, y_train.reshape(-1, 1))
        y_test = InverseScale(scaler, y_test.reshape(-1, 1))

    st.text(f"{model_name} model performance:")
    st.text(f"Training: MSE = {mean_squared_error(y_train, y_fit):.4f}, R^2 = {r2_score(y_train, y_fit):.4f}")
    st.text(f"Testing: MSE = {mean_squared_error(y_test, y_pred):.4f}, R^2 = {r2_score(y_test, y_pred):.4f}")

    return y_fit, y_pred

def Trade_suggest(today, tomorrow):
    """
    Provide trading suggestions based on predicted price changes.

    Parameters:
        today (float): Predicted price for the current day.
        tomorrow (float): Predicted price for the next day.

    Returns:
        None
    """
    st.text(f"Predicted next day price: {tomorrow:.4f}")

    if today <= tomorrow:
        change_percent = (tomorrow - today) / today * 100.0
        st.text(f"The stock price is expected to increase by {change_percent:.2f}%. Recommendation: LONG")
    else:
        change_percent = (today - tomorrow) / today * 100.0
        st.text(f"The stock price is expected to decrease by {change_percent:.2f}%. Recommendation: SHORT")

def Visualize_recent_trend(model_name, data, predicted_data, days=60):
    """
    Display a comparison of actual and predicted stock prices over recent days.

    Parameters:
        model_name (str): Name of the model used for prediction.
        data (pd.DataFrame): Historical stock data.
        predicted_data (np.ndarray): Predicted stock prices for testing data.
        days (int): Number of recent days to plot.

    Returns:
        None
    """
    days = min(days, predicted_data.shape[0])
    data_plot = data[-days:].copy()
    data_plot['Prediction'] = predicted_data[-days:]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_title(f"Actual vs Predicted Stock Prices ({model_name})")
    ax.plot(data_plot['Close'], label='Actual Price')
    ax.plot(data_plot['Prediction'], '-.', label='Predicted Price')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price ($)')
    ax.legend()
    st.pyplot(fig)
