# Stock-Predictor

## Overview
The Stock-Predictor project leverages machine learning techniques to analyze historical stock price data and predict future stock prices. The project includes a web-based application that allows users to visualize stock data, choose from various prediction models, and receive actionable insights based on predictions.

---

## Table of Contents
1. [Installation](#installation)
2. [Project Purpose](#project-purpose)
3. [Project Details](#project-details)
4. [File Descriptions](#file-descriptions)
5. [How to Run](#how-to-run)
6. [Analysis Overview](#analysis-overview)
7. [Results](#results)
8. [Conclusion](#conclusion)
9. [Acknowledgements](#acknowledgements)

---

## Installation

To run the scripts and notebooks in this project, ensure you have the following Python libraries installed:
- `streamlit`
- `yfinance`
- `pandas`
- `numpy`
- `matplotlib`
- `tensorflow`
- `scikit-learn`

### Steps:
1. Create a virtual environment (recommended):
   ```bash
   python -m venv env
   source env/bin/activate   # On Windows: .\env\Scripts\activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Project Purpose

The purpose of this project is to explore the viability of predicting stock prices using historical data and machine learning techniques. Specifically, the project aims to:

- Analyze historical stock price trends.
- Compare different machine learning models for prediction accuracy.
- Build a user-friendly application for real-time predictions.

---

## Project Details

This project predicts stock prices using historical data and machine learning models. Key functionalities include:

- A web application built using Streamlit.
- Four models: Linear Regression, Decision Tree, Support Vector Machine (SVM), and Long Short-Term Memory (LSTM) networks.
- Model evaluation metrics include Mean Squared Error (MSE) and R² Score.

### Evaluation Metrics:
- **MSE**: Measures the average squared difference between predicted and actual values.
- **R² Score**: Represents the proportion of variance explained by the model.
- **Prediction Accuracy**: Percentage of predictions within a 5-10% range of the actual price.

---

## File Descriptions

- `Stock_predictor.ipynb`: Main notebook for data analysis, model training, and evaluation.
- `User_Interface.ipynb`: Notebook to interactively test predictions and adjust parameters.
- `helper.py`: Script for data extraction and preprocessing.
- `regression_model.py`: Implements machine learning models.
- `stock_app.py`: Streamlit app for user interaction.
- `requirements.txt`: List of required Python libraries.

---

## How to Run

1. Install required libraries (see [Installation](#installation)).
2. To run the web application:
   ```bash
   streamlit run stock_app.py
   ```
3. Use the sidebar to select stock tickers and models.
4. View predictions and stock trends in the browser.

---

## Analysis Overview

### Data Description
- Data is fetched using the Yahoo Finance API.
- Includes daily open, close, high, low prices, volume, dividends, and splits.
- Preprocessed to focus on the closing price.

### Methodology
1. **Regression Models**:
   - Linear Regression
   - Decision Tree Regression
   - Support Vector Machine (SVM)
2. **Deep Learning**:
   - LSTM Neural Networks to capture long-term dependencies.

### Key Steps:
- Preprocessing: Normalize closing prices to the range (0, 1).
- Split data into training and testing sets (80:20 split).
- Evaluate models using MSE, R² Score, and prediction accuracy.

---

## Results


Key Insights:
- Linear Regression performs consistently well for most stocks.
- Decision Trees are prone to overfitting, especially with high depth.
- SVM balances flexibility and robustness.
- LSTM models capture complex patterns but may overfit on training data.

---

## Conclusion

### Reflection
This project highlights the potential of machine learning for stock price prediction. While simpler models like Linear Regression work well for short-term trends, advanced models like LSTM are better suited for capturing intricate patterns over longer periods.

## Acknowledgements

- **Data Source**: [Yahoo Finance API](https://finance.yahoo.com/).

