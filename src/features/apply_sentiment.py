import os
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from transformers import pipeline
import warnings

warnings.filterwarnings("ignore")
load_dotenv()

DB_URL = os.getenv("DATABASE_URL")
engine = create_engine(DB_URL, pool_pre_ping=True)

def apply_sentiment():
    print("Loading FinBERT model (this might take a minute the first time)...")
    # ProsusAI/finbert is the standard open-source Financial BERT
    nlp = pipeline("sentiment-analysis", model="ProsusAI/finbert")
    
    print("Fetching news from database...")
    query = "SELECT id, headline FROM news WHERE sentiment_score IS NULL"
    df = pd.read_sql(query, engine)
    
    if df.empty:
        print("No new news to process.")
        return
        
    print(f"Processing {len(df)} headlines...")
    
    with engine.begin() as conn:
        for index, row in df.iterrows():
            headline = row['headline']
            news_id = row['id']
            
            try:
                # FinBERT outputs something like [{'label': 'positive', 'score': 0.89}]
                result = nlp(headline[:512])[0] # Truncate to 512 tokens (BERT limit)
                label = result['label']
                score = result['score']
                
                # Convert label to a numeric score for the ML model
                # Positive = score, Negative = -score, Neutral = 0
                if label == 'positive':
                    numeric_score = score
                elif label == 'negative':
                    numeric_score = -score
                else:
                    numeric_score = 0.0
                    
                # Update the database
                conn.execute(
                    text("""
                        UPDATE news 
                        SET sentiment_score = :score, sentiment_label = :label
                        WHERE id = :id
                    """),
                    {"score": numeric_score, "label": label, "id": news_id}
                )
                
                # Print progress every 5 headlines
                if index % 5 == 0:
                    print(f"  [{index}/{len(df)}] {label.upper()} ({numeric_score:.2f}): {headline[:50]}...")
                    
            except Exception as e:
                print(f"  Error processing headline {news_id}: {e}")

    print("Sentiment analysis complete! Database updated.")

if __name__ == "__main__":
    apply_sentiment()