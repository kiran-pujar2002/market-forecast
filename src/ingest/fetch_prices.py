import os
from datetime import date
import yfinance as yf
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
import warnings

warnings.filterwarnings("ignore")
load_dotenv()

DB_URL = os.getenv("DATABASE_URL")
engine = create_engine(DB_URL, pool_pre_ping=True)

TICKERS = ["AAPL", "MSFT", "TSLA", "NVDA", "GOOGL"]
START = "2019-01-01"
END = str(date.today())

def fetch_and_store(ticker: str):
    print(f"Fetching {ticker}...")
    df = yf.download(ticker, start=START, end=END, multi_level_index=False)
    if df.empty:
        print(f"  No data for {ticker}")
        return

    df = df.reset_index()
    df.columns = [c.lower() for c in df.columns]
    df["ticker"] = ticker
    df["date"] = pd.to_datetime(df["date"]).dt.date

    rows = df[["ticker", "date", "open", "high", "low", "close", "volume"]]

    # Fast bulk insert using Pandas
    try:
        # We use 'append' to add to the existing table
        rows.to_sql("prices", engine, if_exists="append", index=False, method="multi")
        print(f"  Successfully inserted {len(rows)} rows for {ticker}")
    except Exception as e:
        print(f"  Failed to insert {ticker}. Error: {e}")

if __name__ == "__main__":
    for t in TICKERS:
        fetch_and_store(t)
    print("Done.")