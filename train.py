import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
import joblib
import os
from utils import load_and_merge_data

def create_features(df):
    df = df.copy()

    # Make sure Date column is datetime
    if not np.issubdtype(df["Date"].dtype, np.datetime64):
        df["Date"] = pd.to_datetime(df["Date"])

    df["Week"] = df["Date"].dt.isocalendar().week.astype(int)
    df["Month"] = df["Date"].dt.month
    df["Year"] = df["Date"].dt.year
    df["Lag1"] = df["Weekly_Sales"].shift(1)
    df["Lag2"] = df["Weekly_Sales"].shift(2)
    df["Rolling3"] = df["Weekly_Sales"].rolling(3).mean()
    df["Rolling6"] = df["Weekly_Sales"].rolling(6).mean()
    return df.dropna()

def train_model():
    print("📂 Loading and merging dataset...")
    _, weekly_sales = load_and_merge_data()

    print("⚙️ Creating features...")
    df = create_features(weekly_sales)

    X = df[["Week", "Month", "Year", "Lag1", "Lag2", "Rolling3", "Rolling6"]]
    y = df["Weekly_Sales"]

    print("🔄 Starting cross-validation...")
    tscv = TimeSeriesSplit(n_splits=5)
    preds, actuals = [], []

    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=5, random_state=42)
        model.fit(X_train, y_train)
        pred = model.predict(X_val)

        preds.extend(pred)
        actuals.extend(y_val)

    mae = mean_absolute_error(actuals, preds)
    print(f"✅ Cross-validated MAE: {mae:.2f}")

    # Final training on all data
    print("📊 Training final model on full dataset...")
    model.fit(X, y)

    # Save trained model
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/sales_forecast.pkl")
    print("💾 Model saved to models/sales_forecast.pkl")

    # Save OOF plot
    plt.figure(figsize=(10,5))
    plt.plot(df["Date"], y, label="Actual")
    plt.plot(df["Date"].iloc[len(df)-len(preds):], preds, label="Predicted")
    plt.legend()
    plt.title("Out-of-Fold Forecast")
    plt.xlabel("Date")
    plt.ylabel("Weekly Sales")
    plt.savefig("static/oof_plot.png")
    print("📈 OOF plot saved to static/oof_plot.png")

if __name__ == "__main__":
    train_model()
