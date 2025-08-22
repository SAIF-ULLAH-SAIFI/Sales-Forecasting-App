from flask import Flask, render_template, request
import pandas as pd
import joblib
from utils import load_and_merge_data, create_features

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/train")
def train():
    from train import train_model
    train_model()
    return render_template("result.html", msg="Model trained successfully! Check the graph below.")

@app.route("/predict", methods=["POST"])
def predict():
    horizon = int(request.form["horizon"])
    _, weekly_sales = load_and_merge_data()
    df = create_features(weekly_sales)

    # Load model
    model = joblib.load("models/sales_forecast.pkl")

    # Start from last date
    last_date = df["Date"].max()

    forecasts = []
    lag1 = df["Weekly_Sales"].iloc[-1]
    lag2 = df["Weekly_Sales"].iloc[-2]

    for i in range(1, horizon + 1):
        next_date = last_date + pd.Timedelta(weeks=i)

        # features for prediction
        features = {
            "Week": next_date.isocalendar().week,
            "Month": next_date.month,
            "Year": next_date.year,
            "Lag1": lag1,
            "Lag2": lag2,
            "Rolling3": df["Weekly_Sales"].iloc[-3:].mean(),
            "Rolling6": df["Weekly_Sales"].iloc[-6:].mean(),
        }

        X_pred = pd.DataFrame([features])
        y_pred = model.predict(X_pred)[0]

        forecasts.append((next_date, y_pred))

        # update lags
        lag2 = lag1
        lag1 = y_pred

        # append new row properly
        new_row = pd.DataFrame([{
            "Date": next_date,
            "Weekly_Sales": y_pred,
            "Week": features["Week"],
            "Month": features["Month"],
            "Year": features["Year"],
            "Lag1": lag1,
            "Lag2": lag2,
            "Rolling3": features["Rolling3"],
            "Rolling6": features["Rolling6"],
        }])
        df = pd.concat([df, new_row], ignore_index=True)

    # final forecast df
    forecast_df = pd.DataFrame(forecasts, columns=["Date", "Forecast"])
    forecast_df.to_csv("static/forecast.csv", index=False)

    return render_template("forecast.html", table=forecast_df.to_html(classes="table table-striped"))
if __name__ == "__main__":
    app.run(debug=True)
