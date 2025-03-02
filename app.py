import streamlit as st

# Set up the Streamlit app
st.set_page_config(
    page_title="Financial Dashboard",
    page_icon="📈",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.title("Welcome to the Financial Dashboard")
st.write("Navigate through the sections to explore different financial analyses.")

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Home", "My Advisor", "Sentiment Analysis", "Stock Prediction"]
)

# Store the company name in session state
company_name = st.text_input("Enter the company name for analysis (e.g., Apple): ")
if company_name:
    st.session_state['company_name'] = company_name

# Navigate to selected page
if page == "Home":
    st.write("""
    ## Features:
    - **My Advisor**: Get AI-powered investment advice.
    - **Sentiment Analysis**: Analyze market sentiment for a company.
    - **Stock Prediction**: Predict future stock prices.
    """)

elif page == "My Advisor":
    import Advisor

elif page == "Sentiment Analysis":
    import Sentiment_Analysis

elif page == "Stock Prediction":
    import Stock_prediction  # Ensure this file exists
