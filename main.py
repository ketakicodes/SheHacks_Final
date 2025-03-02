# main.py
import streamlit as st

st.set_page_config(
    page_title="Financial Dashboard",
    page_icon="📈",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.title("Welcome to the Financial Dashboard")
st.write("Navigate through the sections to explore different financial analyses.")

# Input for company name
company_name = st.text_input("Enter the company name for analysis (e.g., Apple): ")

# Store the company name in session state
if company_name:
    st.session_state['company_name'] = company_name

st.markdown("""
- **Advisor**: Get AI-powered investment advice.
- **Sentiment Analysis**: Analyze market sentiment for a company.
- **Stock Prediction**: Predict future stock prices.
""")