import pandas as pd

def load_and_merge_data():
    train = pd.read_csv("data/train.csv")
    features = pd.read_csv("data/features.csv")
    stores = pd.read_csv("data/stores.csv")

    # merge datasets
    df = train.merge(features, on=["Store", "Date", "IsHoliday"], how="left")
    df = df.merge(stores, on="Store", how="left")

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")

    return df, df[["Date", "Weekly_Sales"]].groupby("Date").sum().reset_index()

def create_features(df):
    """
    df: DataFrame with columns ['Date','Weekly_Sales']
    """
    df = df.copy()
    df["Week"] = df["Date"].dt.isocalendar().week.astype(int)
    df["Month"] = df["Date"].dt.month
    df["Year"] = df["Date"].dt.year

    # lags
    df["Lag1"] = df["Weekly_Sales"].shift(1)
    df["Lag2"] = df["Weekly_Sales"].shift(2)

    # rolling
    df["Rolling3"] = df["Weekly_Sales"].shift(1).rolling(3).mean()
    df["Rolling6"] = df["Weekly_Sales"].shift(1).rolling(6).mean()

    df = df.dropna()
    return df
