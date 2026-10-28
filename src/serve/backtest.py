import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sqlalchemy import create_engine
from sqlalchemy.pool import NullPool
from dotenv import load_dotenv
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from features.build_features import build_features_for_ticker

load_dotenv()
DB_URL = os.getenv("DATABASE_URL")
engine = create_engine(DB_URL, poolclass=NullPool)

def run_backtest(ticker: str, days: int = 252, initial_capital: float = 10000.0):
    query = f"SELECT date, open, high, low, close, volume FROM prices WHERE ticker = '{ticker}' ORDER BY date"
    df = pd.read_sql(query, engine)
    
    if df.empty or len(df) < days:
        return {"error": f"Not enough data to backtest {ticker}. It needs at least {days} trading days."}

    df['date'] = pd.to_datetime(df['date'])
    features_df = build_features_for_ticker(df)
    
    feature_cols = ['open', 'high', 'low', 'close', 'volume', 'sma_10', 'sma_50', 
                    'rsi_14', 'daily_return', 'volatility_10', 'volume_change']
    
    features_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    features_df.fillna(0, inplace=True)

    backtest_df = features_df.tail(days + 1).copy()
    train_df = features_df.iloc[:-(days + 1)]
    
    model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42)
    model.fit(train_df[feature_cols], train_df['target'])
    
    dates_list = []
    strategy_equity = []
    buy_hold_equity = []
    
    cash = initial_capital
    hold = initial_capital
    
    for i in range(len(backtest_df) - 1):
        current_row = backtest_df.iloc[i]
        next_row = backtest_df.iloc[i+1]
        
        dates_list.append(next_row['date'])
        
        X_current = current_row[feature_cols].values.reshape(1, -1)
        prediction = model.predict(X_current)[0]
        
        daily_ret = next_row['close'] / current_row['close'] - 1
        
        if prediction == 1:
            cash = cash * (1 + daily_ret)
            
        hold = hold * (1 + daily_ret)
        
        strategy_equity.append(cash)
        buy_hold_equity.append(hold)
        
    return {
        "dates": [d.strftime('%Y-%m-%d') for d in dates_list],
        "strategy_equity": [round(x, 2) for x in strategy_equity],
        "buy_hold_equity": [round(x, 2) for x in buy_hold_equity],
        "final_strategy": round(strategy_equity[-1], 2),
        "final_buy_hold": round(buy_hold_equity[-1], 2),
        "strategy_return_pct": round(((strategy_equity[-1] - initial_capital) / initial_capital) * 100, 2),
        "buy_hold_return_pct": round(((buy_hold_equity[-1] - initial_capital) / initial_capital) * 100, 2)
    }