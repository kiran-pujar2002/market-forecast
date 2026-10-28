import os
import requests
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

DB_URL = os.getenv("DATABASE_URL")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
engine = create_engine(DB_URL, pool_pre_ping=True)

TICKERS = ["AAPL", "MSFT", "TSLA", "NVDA", "GOOGL"]

def fetch_news(ticker):
    print(f"Fetching news for {ticker}...")
    from_date = (datetime.now() - timedelta(days=5)).strftime('%Y-%m-%d')
    
    # We add 'language=en' and 'sortBy=relevancy' to get better results
    url = (
        f"https://newsapi.org/v2/everything?"
        f"q=\"{ticker}\" OR \"{ticker} stock\"&"
        f"from={from_date}&"
        f"language=en&"
        f"sortBy=relevancy&"
        f"apiKey={NEWS_API_KEY}"
    )
    
    response = requests.get(url)
    if response.status_code != 200:
        print(f"  Error fetching {ticker}: {response.status_code} - {response.text}")
        return
        
    articles = response.json().get("articles", [])
    if not articles:
        print(f"  No recent news for {ticker}")
        return

    # Blacklist of sources or phrases that give bad data on the free API
    BLACKLIST = ["pypi", "github", "reddit", "r/wallstreetbets"]
    
    news_rows = []
    for article in articles:
        headline = article.get("title", "")
        source = article.get("source", {}).get("name", "").lower()
        published_at_str = article.get("publishedAt")
        
        # Skip if empty
        if not headline or not published_at_str:
            continue
            
        # Skip if the source or headline is in our blacklist
        if any(bad_word in source for bad_word in BLACKLIST) or any(bad_word in headline.lower() for bad_word in BLACKLIST):
            continue
            
        # Basic check: If headline has more than 30% non-ascii characters (like Japanese), skip it.
        non_ascii = sum(1 for c in headline if ord(c) > 127)
        if non_ascii / len(headline) > 0.3:
            continue

        published_at = datetime.fromisoformat(published_at_str.replace("Z", "+00:00"))
        
        news_rows.append({
            "ticker": ticker,
            "published_at": published_at,
            "headline": headline,
            "source": article.get("source", {}).get("name")
        })

    # Insert into database
    if news_rows:
        with engine.begin() as conn:
            for row in news_rows:
                # Check if we already have this exact headline for this ticker
                conn.execute(
                    text("""
                        INSERT INTO news (ticker, published_at, headline, source)
                        SELECT :ticker, :published_at, :headline, :source
                        WHERE NOT EXISTS (
                            SELECT 1 FROM news 
                            WHERE ticker = :ticker AND headline = :headline
                        )
                    """),
                    row
                )
        print(f"  Inserted {len(news_rows)} valid articles for {ticker}")

if __name__ == "__main__":
    for ticker in TICKERS:
        fetch_news(ticker)
    print("Done fetching news.")