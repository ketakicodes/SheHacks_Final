import streamlit as st
import requests
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt

# Function to get stock ticker from Alpha Vantage
def get_stock_ticker_alpha_vantage(company_name, api_key):
    base_url = "https://www.alphavantage.co/query"
    params = {
        "function": "SYMBOL_SEARCH",
        "keywords": company_name,
        "apikey": api_key
    }
    response = requests.get(base_url, params=params)
    data = response.json()
    
    if "bestMatches" in data and data["bestMatches"]:
        return data["bestMatches"][0]["1. symbol"]
    else:
        return None

# Function to calculate RSI
def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Streamlit app layout
st.title("Stock Price Prediction")
st.write("Enter a company name to predict its stock price.")

# User input for company name
company_name = st.text_input("Company Name")

# Your Alpha Vantage API key
api_key = "344LAH6N7S1K0357"

if st.button("Get Stock Price Prediction"):
    if company_name:
        ticker = get_stock_ticker_alpha_vantage(company_name, api_key)

        if ticker:
            st.write(f"Ticker symbol for {company_name} is {ticker}")
            
            # Create a Ticker object
            stock_info = yf.Ticker(ticker)

            # Fetch historical stock price data
            historical_data = stock_info.history(start='2022-01-01', end='2025-01-01')
            st.write("Historical Data:", historical_data.head())

            # Feature Engineering
            data = historical_data[['Close', 'Volume']].copy()
            data['SMA_20'] = data['Close'].rolling(window=20).mean()
            data['Volatility'] = data['Close'].rolling(window=20).std()
            data['RSI'] = calculate_rsi(data['Close'])
            data['Lag_1'] = data['Close'].shift(1)
            data['Lag_2'] = data['Close'].shift(2)  # Additional lag feature
            data.dropna(inplace=True)

            # Check if there is enough data
            if len(data) < 60:
                st.warning("Not enough data to create sequences. Please use a longer time range or reduce the time_step.")
            else:
                # Normalize the data
                scaler = MinMaxScaler(feature_range=(0, 1))
                scaled_data = scaler.fit_transform(data)

                # Create sequences
                def create_sequences(data, time_step=60):
                    X, y = [], []
                    for i in range(len(data) - time_step):
                        X.append(data[i:i + time_step])
                        y.append(data[i + time_step, 0])  # Predicting the 'Close' price
                    return np.array(X), np.array(y)

                time_step = 60
                X, y = create_sequences(scaled_data, time_step)

                # Split into training and testing sets
                train_size = int(len(X) * 0.8)
                X_train, X_test = X[:train_size], X[train_size:]
                y_train, y_test = y[:train_size], y[train_size:]

                # Build the LSTM model
                model = Sequential()
                model.add(LSTM(units=100, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
                model.add(Dropout(0.3))
                model.add(LSTM(units=100, return_sequences=False))
                model.add(Dropout(0.3))
                model.add(Dense(units=1))

                # Compile the model
                model.compile(optimizer='adam', loss='mean_squared_error')

                # Train the model
                model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.1)

                # Predict and inverse transform the predictions
                predictions = model.predict(X_test)
                predictions = scaler.inverse_transform(np.concatenate((predictions, np.zeros((predictions.shape[0], scaled_data.shape[1] - 1))), axis=1))[:, 0]
                actual_prices = scaler.inverse_transform(np.concatenate((y_test.reshape(-1, 1), np.zeros((y_test.shape[0], scaled_data.shape[1] - 1))), axis=1))[:, 0]

                # Evaluate the model
                rmse = np.sqrt(mean_squared_error(actual_prices, predictions))
                mae = mean_absolute_error(actual_prices, predictions)
                mape = np.mean(np.abs((actual_prices - predictions) / actual_prices)) * 100

                st.write(f"RMSE: {rmse}")
                st.write(f"MAE: {mae}")
                st.write(f"MAPE: {mape}%")

                # Visualization of Actual vs Predicted Prices
                fig, ax = plt.subplots(figsize=(14, 5))
                predicted_dates = historical_data.index[-len(y_test):]  # Get the last dates corresponding to the test set
                ax.plot(predicted_dates, actual_prices, color='blue', label='Actual Stock Price')
                ax.plot(predicted_dates, predictions, color='red', label='Predicted Stock Price')
                ax.set_title('Stock Price Prediction', fontsize=16, fontweight='bold')
                ax.set_xlabel('Date', fontsize=14)
                ax.set_ylabel('Stock Price', fontsize=14)
                ax.legend()
                ax.grid()
                st.pyplot(fig)

                # Additional Visualizations
                fig, ax = plt.subplots(figsize=(14, 5))
                ax.plot(data['Close'], label='Close Price')
                ax.plot(data['SMA_20'], label='20-Day SMA', linestyle='--')
                ax.set_title('Close Price and Moving Average', fontsize=16, fontweight='bold')
                ax.set_xlabel('Date', fontsize=14)
                ax.set_ylabel('Price', fontsize=14)
                ax.legend()
                ax.grid()
                st.pyplot(fig)

                fig, ax = plt.subplots(figsize=(14, 5))
                ax.plot(data['Volatility'], label='Volatility', color='orange')
                ax.set_title('Volatility Over Time', fontsize=16, fontweight='bold')
                ax.set_xlabel('Date', fontsize=14)
                ax.set_ylabel('Volatility', fontsize=14)
                ax.legend()
                ax.grid()
                st.pyplot(fig)

                fig, ax = plt.subplots(figsize=(14, 5))
                ax.plot(data['RSI'], label='RSI', color='purple')
                ax.axhline(70, linestyle='--', alpha=0.5, color='red')
                ax.axhline(30, linestyle='--', alpha=0.5, color='green')
                ax.set_title('Relative Strength Index (RSI)', fontsize=16, fontweight='bold')
                ax.set_xlabel('Date', fontsize=14)
                ax.set_ylabel('RSI', fontsize=14)
                ax.legend()
                ax.grid()
                st.pyplot(fig)

        else:
            st.error(f"Could not find a ticker symbol for {company_name}. Please try a different name.")
    else:
        st.warning("Please enter a company name.")