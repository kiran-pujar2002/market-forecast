import os
import joblib
import pandas as pd
import numpy as np
import yfinance as yf
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sqlalchemy import create_engine, text
from sqlalchemy.pool import NullPool
from dotenv import load_dotenv
import sys
import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from features.build_features import build_features_for_ticker
from serve.backtest import run_backtest

load_dotenv()
DB_URL = os.getenv("DATABASE_URL")
engine = create_engine(DB_URL, poolclass=NullPool)

app = FastAPI(title="Dynamic Market Forecast API")

class TickerRequest(BaseModel):
    ticker: str
    horizon: int = 1

def get_ticker_data_from_db(ticker: str):
    query = f"SELECT date, open, high, low, close, volume FROM prices WHERE ticker = '{ticker}' ORDER BY date"
    df = pd.read_sql(query, engine)
    if df.empty:
        return None
    df['ticker'] = ticker
    return df

def save_ticker_data_to_db(ticker: str, df: pd.DataFrame):
    print(f"Saving {ticker} to Neon database...")
    df_to_save = df[['ticker', 'date', 'open', 'high', 'low', 'close', 'volume']].copy()
    try:
        df_to_save.to_sql("prices", engine, if_exists="append", index=False, method="multi")
        print(f"Successfully cached {ticker}.")
    except Exception as e:
        print(f"WARNING: Could not cache {ticker}. Error: {e}")

def get_model_and_features(ticker: str, horizon: int):
    ticker = ticker.upper()
    model_path = f"models/xgb_{ticker}_h{horizon}_model.pkl"
    
    df = get_ticker_data_from_db(ticker)
    
    if df is not None and len(df) > 60:
        print(f"Loaded {ticker} from Neon database (Cache Hit).")
    else:
        print(f"Downloading {ticker} from Yahoo Finance (Cache Miss)...")
        df = yf.download(ticker, start="2015-01-01", multi_level_index=False)
        
        if df.empty or len(df) < 60:
            raise ValueError(f"Ticker {ticker} not found or has less than 60 days of trading data.")
            
        df = df.reset_index()
        df.columns = [str(c).lower() for c in df.columns]
        if 'date' not in df.columns:
            if 'index' in df.columns: df.rename(columns={'index': 'date'}, inplace=True)
            elif 'datetime' in df.columns: df.rename(columns={'datetime': 'date'}, inplace=True)
                
        df['ticker'] = ticker
        df['date'] = pd.to_datetime(df['date']).dt.date
        save_ticker_data_to_db(ticker, df)
    
    df['date'] = pd.to_datetime(df['date'])
    features_df = build_features_for_ticker(df)
    
    if horizon > 1:
        features_df['target'] = (features_df['close'].shift(-horizon) > features_df['close']).astype(int)
        features_df.dropna(inplace=True) 
    
    feature_cols = ['open', 'high', 'low', 'close', 'volume', 'sma_10', 'sma_50', 
                    'rsi_14', 'daily_return', 'volatility_10', 'volume_change']
    
    features_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    features_df.fillna(0, inplace=True)

    if os.path.exists(model_path):
        model = joblib.load(model_path)
    else:
        print(f"Training new model for {ticker} (Horizon: {horizon} days)...")
        model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42)
        model.fit(features_df[feature_cols], features_df['target'])
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, model_path)
        
    latest_features = features_df[feature_cols].iloc[-1:]
    return model, latest_features, features_df

@app.post("/predict")
def predict(request: TickerRequest):
    try:
        model, latest_features, features_df = get_model_and_features(request.ticker, request.horizon)
        
        prediction = model.predict(latest_features)[0]
        probability = model.predict_proba(latest_features)[0].max()
        
        latest_close = float(latest_features['close'].values[0])
        avg_daily_return = features_df['daily_return'].mean()
        direction_multiplier = 1 if prediction == 1 else -1
        
        expected_return_pct = (avg_daily_return * request.horizon) * probability * direction_multiplier
        target_price = latest_close * (1 + expected_return_pct)
        pct_string = f"{'+' if expected_return_pct > 0 else ''}{expected_return_pct * 100:.2f}%"
        
        future_dates = []
        future_prices = []
        
        current_date = features_df['date'].iloc[-1]
        if isinstance(current_date, str):
            current_date = datetime.datetime.strptime(current_date, '%Y-%m-%d').date()
        elif isinstance(current_date, pd.Timestamp):
            current_date = current_date.date()
            
        current_price = latest_close
        
        days_added = 0
        while days_added < request.horizon:
            current_date += datetime.timedelta(days=1)
            if current_date.weekday() < 5:
                current_price = current_price * (1 + (expected_return_pct / request.horizon))
                future_dates.append(current_date.strftime('%Y-%m-%d'))
                future_prices.append(current_price)
                days_added += 1

        chart_data = features_df.tail(60).copy()
        chart_data['date'] = chart_data['date'].astype(str)
        chart_cols = ['date', 'open', 'high', 'low', 'close', 'sma_10', 'sma_50', 'volume']
        chart_json = chart_data[chart_cols].to_dict(orient="records")
        
        return {
            "ticker": request.ticker.upper(),
            "direction": "UP" if prediction == 1 else "DOWN",
            "confidence": round(float(probability), 4),
            "latest_close": round(latest_close, 2),
            "target_price": round(target_price, 2),
            "expected_return_pct": pct_string,
            "historical_data": chart_json,
            "future_dates": future_dates,
            "future_prices": future_prices,
            "data_days": len(features_df)  # <-- Passes the data length to frontend
        }
    except Exception as e:
        print(f"ERROR: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/backtest/{ticker}")
def backtest_endpoint(ticker: str, days: int = 252):
    try:
        results = run_backtest(ticker.upper(), days)
        if "error" in results:
            raise HTTPException(status_code=400, detail=results["error"])
        return results
    except Exception as e:
        print(f"ERROR: {e}")
        raise HTTPException(status_code=400, detail=str(e))