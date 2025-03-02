import streamlit as st
import praw
import requests
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
from bs4 import BeautifulSoup
import unicodedata

# Check if the company name is stored in session state
if 'company_name' in st.session_state:
  company_name = st.session_state['company_name']
  st.write(f"Analyzing sentiment for: {company_name}")

  # Fetch and analyze sentiment
  # (Add your sentiment analysis logic here)
else:
  st.error("No company name provided. Please go back to the main page and enter a company name.")

# Download NLTK data
nltk.download('stopwords')
nltk.download('wordnet')

# Set Up API Credentials
reddit = praw.Reddit(
    client_id="WFMf8tiQFeilUmjqTKmdQQ",
    client_secret="bCCgYmZYZTv7XEEtW0SSJr1ayd19cw",
    user_agent="finance_sent:1.0(by LopsidedMarketing166)"
)

alpha_vantage_api_key = "ISTOQJTRNIJSV9PG"

news_api_key = 'd5619f5224324eea8874a3c3e794334a'

# Load FinBERT Model & Tokenizer
tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
model = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone')

# Fetch Stock Ticker Symbol Using Alpha Vantage


def get_stock_ticker_alpha_vantage(company_name, api_key):
  try:
    base_url = "https://www.alphavantage.co/query"
    params = {
        "function": "SYMBOL_SEARCH",
        "keywords": company_name,
        "apikey": api_key
    }
    response = requests.get(base_url, params=params)
    response.raise_for_status()
    data = response.json()

    if "bestMatches" in data and data["bestMatches"]:
      return data["bestMatches"][0]["1. symbol"]
  except requests.exceptions.RequestException as e:
    st.error(f"Error fetching ticker for {company_name}: {e}")
  return None

# Fetch Financial News from Reddit


def fetch_reddit_posts_for_stock(ticker_symbol, limit=90):
  try:
    subreddit = reddit.subreddit("stocks")
    posts = []
    for post in subreddit.search(ticker_symbol, limit=limit):
      posts.append(post.title + ". " + post.selftext)
    return posts
  except Exception as e:
    st.error(f"Error fetching Reddit posts: {e}")
  return []


# Fetch Financial News Using News API
def fetch_financial_news(stock_symbol, api_key):
  try:
    url = 'https://newsapi.org/v2/everything'
    params = {
        'q': stock_symbol,
        'language': 'en',
        'sortBy': 'relevancy',
        'apiKey': api_key
    }
    response = requests.get(url, params=params)
    response.raise_for_status()
    articles = response.json().get('articles', [])
    return [article['title'] + ". " + article['description'] for article in articles if article['description']]
  except requests.exceptions.RequestException as e:
    st.error(f"Error fetching news: {e}")
  return []

# Preprocess and Analyze Sentiment Using FinBERT


def preprocess_text(text):
  text = BeautifulSoup(text, "html.parser").get_text()
  text = unicodedata.normalize("NFKD", text)
  text = text.lower()
  text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
  text = text.translate(str.maketrans('', '', string.punctuation))
  text = re.sub(r'[^\x00-\x7F]+', ' ', text)
  text = re.sub(r'\s+', ' ', text).strip()
  text = re.sub(r'\d+', '', text)
  stop_words = set(stopwords.words('english'))
  text = ' '.join([word for word in text.split() if word not in stop_words])
  lemmatizer = WordNetLemmatizer()
  text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])

  inputs = tokenizer(text, return_tensors='pt', max_length=512, truncation=True, padding=True)
  return inputs


def analyze_sentiment(news_articles):
  sentiments = []
  for article in news_articles:
    inputs = preprocess_text(article)
    with torch.no_grad():
      outputs = model(**inputs)
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    sentiment = torch.argmax(probabilities, dim=1).item()
    sentiment_label = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
    sentiments.append((article, sentiment_label[sentiment]))
  return sentiments

# Count Sentiment Articles


def count_sentiment_articles(sentiment_results):
  positive_count = sum(1 for _, sentiment in sentiment_results if sentiment == 'Positive')
  neutral_count = sum(1 for _, sentiment in sentiment_results if sentiment == 'Neutral')
  negative_count = sum(1 for _, sentiment in sentiment_results if sentiment == 'Negative')
  total_count = len(sentiment_results)
  return positive_count, neutral_count, negative_count, total_count


def determine_overall_mood(positive_count, neutral_count, negative_count):
  if positive_count > negative_count and positive_count > neutral_count:
    return "Positive"
  elif negative_count > positive_count and negative_count > neutral_count:
    return "Negative"
  else:
    return "Neutral"


def generate_trading_signal(overall_mood):
  if overall_mood == "Positive":
    return "Buy"
  elif overall_mood == "Negative":
    return "Sell"
  else:
    return "Hold"

# Main Function


def main():
  # Check if the company name is passed
  if 'company_name' in st.session_state:
    company_name = st.session_state['company_name']
    st.title(f"Sentiment Analysis for {company_name}")

    # Fetch stock symbol
    stock_symbol = get_stock_ticker_alpha_vantage(company_name, alpha_vantage_api_key)

    if stock_symbol:
      st.success(f"Stock Symbol for {company_name}: {stock_symbol}")

      # Fetch data from each source
      reddit_posts = fetch_reddit_posts_for_stock(stock_symbol, limit=90)
      news_articles = fetch_financial_news(stock_symbol, news_api_key)

      # Analyze sentiment for eacAh source separately
      if reddit_posts:
        reddit_sentiment_results = analyze_sentiment(reddit_posts)
        reddit_positive_count, reddit_neutral_count, reddit_negative_count, reddit_total_count = count_sentiment_articles(
            reddit_sentiment_results)
        reddit_overall_mood = determine_overall_mood(reddit_positive_count, reddit_neutral_count, reddit_negative_count)
        reddit_signal = generate_trading_signal(reddit_overall_mood)

        st.subheader("Reddit Sentiment Analysis")
        st.write(f"Positive Articles: {reddit_positive_count} out of {reddit_total_count}")
        st.write(f"Neutral Articles: {reddit_neutral_count} out of {reddit_total_count}")
        st.write(f"Negative Articles: {reddit_negative_count} out of {reddit_total_count}")
        st.write(f"Overall Sentiment Mood: {reddit_overall_mood}")
        st.write(f"Trading Signal: {reddit_signal}")
      else:
        st.error("No relevant Reddit posts found.")

      if news_articles:
        news_sentiment_results = analyze_sentiment(news_articles)
        news_positive_count, news_neutral_count, news_negative_count, news_total_count = count_sentiment_articles(
            news_sentiment_results)
        news_overall_mood = determine_overall_mood(news_positive_count, news_neutral_count, news_negative_count)
        news_signal = generate_trading_signal(news_overall_mood)

        st.subheader("News API Sentiment Analysis")
        st.write(f"Positive Articles: {news_positive_count} out of {news_total_count}")
        st.write(f"Neutral Articles: {news_neutral_count} out of {news_total_count}")
        st.write(f"Negative Articles: {news_negative_count} out of {news_total_count}")
        st.write(f"Overall Sentiment Mood: {news_overall_mood}")
        st.write(f"Trading Signal: {news_signal}")
      else:
        st.error("No relevant news articles found.")
    else:
      st.error("Company name not recognized or no ticker found.")
  else:
    st.error("No company name provided. Please go back to the main page and enter a company name.")


# Run the Main Function
if __name__ == "__main__":
  main()
