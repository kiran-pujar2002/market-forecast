import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import warnings
import joblib

warnings.filterwarnings("ignore")

def train_walk_forward():
    print("Loading engineered features...")
    df = pd.read_csv("data/processed/final_features.csv")
    
    # Sort strictly by date to ensure time-series integrity
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values('date', inplace=True)
    
    # Define Features (X) and Target (y)
    # We exclude 'date', 'ticker', and 'target' from the features
    feature_cols = [c for c in df.columns if c not in ['date', 'ticker', 'target']]
    
    # We will predict for each ticker independently
    tickers = df['ticker'].unique()
    
    results = {}
    
    for ticker in tickers:
        print(f"\n--- Training for {ticker} ---")
        t_df = df[df['ticker'] == ticker].copy()
        
        X = t_df[feature_cols]
        y = t_df['target']
        
        # Walk-Forward Parameters
        initial_train_size = int(len(t_df) * 0.6) # Start with 60% of data for training
        test_size = 30 # Predict the next 30 days, then roll forward
        step_size = 30
        
        predictions = []
        actuals = []
        
        # Walk-forward loop
        for start_idx in range(initial_train_size, len(t_df) - test_size, step_size):
            train_end = start_idx
            test_end = start_idx + test_size
            
            X_train = X.iloc[:train_end]
            y_train = y.iloc[:train_end]
            
            X_test = X.iloc[train_end:test_end]
            y_test = y.iloc[train_end:test_end]
            
            # Train XGBoost model
            model = xgb.XGBClassifier(
                n_estimators=100, 
                learning_rate=0.05, 
                max_depth=3, 
                random_state=42,
                eval_metric='logloss'
            )
            model.fit(X_train, y_train)
            
            # Predict
            preds = model.predict(X_test)
            predictions.extend(preds)
            actuals.extend(y_test)
            
        # Calculate overall metrics for this ticker
        acc = accuracy_score(actuals, predictions)
        auc = roc_auc_score(actuals, predictions)
        
        print(f"{ticker} Walk-Forward Accuracy: {acc:.4f}")
        print(f"{ticker} Walk-Forward ROC-AUC:  {auc:.4f}")
        
        results[ticker] = {'accuracy': acc, 'auc': auc}
        
    # Print average performance
    avg_acc = np.mean([r['accuracy'] for r in results.values()])
    avg_auc = np.mean([r['auc'] for r in results.values()])
    print(f"\n==============================")
    print(f"Average Accuracy: {avg_acc:.4f}")
    print(f"Average ROC-AUC:  {avg_auc:.4f}")
    print(f"==============================")
    
    # Save the last trained model (we'll just train one on all AAPL data to save for the API later)
    print("\nTraining final model on latest data to save...")
    final_X = df[df['ticker'] == 'AAPL'][feature_cols]
    final_y = df[df['ticker'] == 'AAPL']['target']
    
    final_model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42)
    final_model.fit(final_X, final_y)
    
    os.makedirs("models", exist_ok=True)
    joblib.dump(final_model, "models/xgb_aapl_model.pkl")
    print("Model saved to models/xgb_aapl_model.pkl")

if __name__ == "__main__":
    train_walk_forward()