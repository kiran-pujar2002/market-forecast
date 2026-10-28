import os
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
import warnings

warnings.filterwarnings("ignore")
load_dotenv()

DB_URL = os.getenv("DATABASE_URL")
engine = create_engine(DB_URL, pool_pre_ping=True)

def build_final_features():
    print("1. Loading price features...")
    df_price = pd.read_csv("data/processed/features.csv")
    df_price['date'] = pd.to_datetime(df_price['date']).dt.date
    
    print("2. Fetching news sentiment from database...")
    query = """
        SELECT ticker, DATE(published_at) as date, AVG(sentiment_score) as daily_sentiment
        FROM news
        WHERE sentiment_score IS NOT NULL
        GROUP BY ticker, DATE(published_at)
    """
    df_news = pd.read_sql(query, engine)
    df_news['date'] = pd.to_datetime(df_news['date']).dt.date
    
    print(f"Loaded sentiment for {len(df_news)} unique ticker-days.")
    
    print("3. Merging price and sentiment...")
    # Left join means we keep all price days. If there's no news that day, sentiment becomes NaN
    merged_df = pd.merge(df_price, df_news, on=['ticker', 'date'], how='left')
    
    # If there is no news on a given day, we assume sentiment is neutral (0.0)
    merged_df['daily_sentiment'] = merged_df['daily_sentiment'].fillna(0.0)
    
    print("4. Saving final dataset...")
    output_path = "data/processed/final_features.csv"
    merged_df.to_csv(output_path, index=False)
    print(f"Saved combined dataset to {output_path}")
    print(f"Final shape: {merged_df.shape}")

if __name__ == "__main__":
    build_final_features()