import os
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from dotenv import load_dotenv
import warnings

warnings.filterwarnings("ignore")
load_dotenv()

DB_URL = os.getenv("DATABASE_URL")
engine = create_engine(DB_URL, pool_pre_ping=True)

def calculate_rsi(series, period=14):
    delta = series.diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def build_features_for_ticker(df):
    """Takes a raw dataframe for a SINGLE ticker and returns engineered features."""
    t_df = df.copy()
    
    # Ensure date is a column and is datetime
    if 'date' not in t_df.columns:
        if 'index' in t_df.columns:
            t_df.rename(columns={'index': 'date'}, inplace=True)
    
    t_df['date'] = pd.to_datetime(t_df['date'])
    t_df.set_index('date', inplace=True)
    
    t_df['sma_10'] = t_df['close'].rolling(window=10).mean()
    t_df['sma_50'] = t_df['close'].rolling(window=50).mean()
    t_df['rsi_14'] = calculate_rsi(t_df['close'], period=14)
    t_df['daily_return'] = t_df['close'].pct_change()
    t_df['volatility_10'] = t_df['daily_return'].rolling(window=10).std()
    t_df['volume_change'] = t_df['volume'].pct_change()
    
    t_df['target'] = (t_df['close'].shift(-1) > t_df['close']).astype(int)
    
    t_df.dropna(inplace=True)
    t_df.reset_index(inplace=True)
    return t_df

def build_features():
    """Original function to process all tickers and save to CSV."""
    print("Fetching raw prices from database...")
    query = "SELECT ticker, date, open, high, low, close, volume FROM prices ORDER BY ticker, date"
    df = pd.read_sql(query, engine)
    
    processed_dfs = []
    for ticker in df['ticker'].unique():
        t_df = df[df['ticker'] == ticker]
        processed = build_features_for_ticker(t_df)
        processed_dfs.append(processed)
        
    final_df = pd.concat(processed_dfs)
    
    os.makedirs("data/processed", exist_ok=True)
    output_path = "data/processed/features.csv"
    final_df.to_csv(output_path, index=False)
    print(f"Saved processed features to {output_path}")

if __name__ == "__main__":
    build_features()