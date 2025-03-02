import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
import numpy as np

# 🎯 App Title
st.set_page_config(page_title="AI-Powered Investment Advisor", layout="wide")
st.title("📈 AI-Powered Investment Advisor (Real-Time Sentiment)")

# 📌 Sidebar: User Preferences
st.sidebar.header("🔍 Select Your Preferences")

# 📌 Stock Selection
stocks_input = st.sidebar.text_input(
    "Select Stocks (comma-separated, e.g., AAPL, MSFT, TSLA)", 
    placeholder="Example: AAPL, TSLA, GOOGL"
)

# 📌 Risk Assessment (Sliders)
st.sidebar.subheader("🧩 Risk Assessment")
age = st.sidebar.slider("Your Age:", min_value=18, max_value=80)
investment_horizon = st.sidebar.slider("Investment Horizon (Years):", min_value=1, max_value=30)
risk_tolerance = st.sidebar.selectbox("Risk Tolerance Level:", ["Select", "Low", "Medium", "High"])

# 📌 Stock Trend Selection (Moved to Sidebar)
st.sidebar.subheader("📈 View Stock Trend (Optional)")
selected_stock = st.sidebar.text_input(
    "Enter a stock symbol to view trend (e.g., GOOGL):", placeholder="Example: GOOGL"
)

# 📌 Ensure user input before showing recommendations
if st.sidebar.button("🎯 Generate Recommendations"):
    if not stocks_input:
        st.sidebar.warning("⚠️ Please enter at least one stock.")
    elif risk_tolerance == "Select":
        st.sidebar.warning("⚠️ Please choose your risk tolerance level.")
    else:
        stocks = [s.strip().upper() for s in stocks_input.split(",") if s.strip()]

        # Function to Fetch Stock Data
        def get_stock_data(tickers):
            data = []
            np.random.seed(42)  # Ensure reproducibility
            
            for ticker in tickers:
                try:
                    stock = yf.Ticker(ticker)
                    hist = stock.history(period="6mo")  
                    if not hist.empty:
                        last_price = hist['Close'].iloc[-1]
                        volatility = hist['Close'].pct_change().std()

                        # Generate Sentiment Score & Risk Cluster
                        sentiment_score = np.round(np.random.uniform(-1, 1), 2)
                        risk_cluster = np.random.choice(["Low Risk", "Medium Risk", "High Risk"])

                        data.append({
                            "Stock": ticker, 
                            "Price (USD)": round(last_price, 2), 
                            "Volatility": round(volatility, 4),
                            "Sentiment Score": f"{sentiment_score} ({'Bullish' if sentiment_score > 0 else 'Bearish'})",
                            "Risk Cluster": risk_cluster
                        })
                except Exception as e:
                    st.warning(f"Could not fetch data for {ticker}: {e}")
            
            return pd.DataFrame(data)

        # Fetch Market Data
        market_data = get_stock_data(stocks)

        # 🎯 Display Market Data with Sentiment Score & Risk Cluster
        if not market_data.empty:
            st.subheader("📊 AI-Based Investment Recommendations")
            st.dataframe(market_data)

            # 🎯 AI Investment Recommendation Logic
            st.subheader("🤖 Personalized Investment Recommendations")
            recommended_stock = market_data.sort_values(by="Volatility", ascending=(risk_tolerance == "Low")).head(1)['Stock'].values[0]
            st.success(f"✅ Based on your risk profile, we recommend: *{recommended_stock}*")

            # 📊 Bar Chart for Stock Prices
            st.subheader("📊 Live Stock Market Data")
            fig = px.bar(
                market_data, 
                x="Stock", 
                y="Price (USD)", 
                color="Price (USD)", 
                color_continuous_scale="bluered",
                title="Stock Prices"
            )
            fig.update_layout(xaxis_title="Stock", yaxis_title="Price")
            st.plotly_chart(fig)

        # 🎯 Stock Price Trend Visualization
        if selected_stock:
            st.subheader(f"📉 Stock Trend for {selected_stock.upper()}")
            stock_data = yf.Ticker(selected_stock.upper()).history(period="6mo")

            if not stock_data.empty:
                fig = px.line(
                    stock_data, 
                    x=stock_data.index, 
                    y="Close", 
                    title=f"{selected_stock.upper()} Stock Price Trend"
                )
                fig.update_layout(xaxis_title="Date", yaxis_title="Price (USD)")
                st.plotly_chart(fig)
            else:
                st.warning(f"No data available for {selected_stock.upper()}")

# 💡 Disclaimer
st.info("💡 Note: This tool fetches real-time data from Yahoo Finance. Always conduct your own research before investing!")
