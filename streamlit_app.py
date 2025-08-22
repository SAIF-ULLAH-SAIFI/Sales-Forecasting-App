import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("models/sales_forecast.pkl")

# Page Config
st.set_page_config(page_title="📊 Sales Forecast Dashboard", layout="wide")

# Sidebar
st.sidebar.title("⚙️ Controls")
horizon = st.sidebar.slider("🔮 Forecast Horizon (weeks)", 1, 52, 12)

# Title
st.title("📊 Sales Forecasting Dashboard")
st.markdown("Upload your sales dataset and get future sales predictions with an interactive dashboard.")

# File uploader
uploaded_file = st.file_uploader("📂 Upload CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("🔍 Uploaded Data Preview")
    st.dataframe(df.head())

    try:
        # Features from uploaded data (excluding target if needed)
        X = df.dropna()

        # Forecast using trained model
        predictions = model.predict(X.tail(horizon))

        st.subheader("📈 Forecast Results")
        st.line_chart(predictions)

        # Display forecast table
        st.write("📊 Future Sales Forecast")
        forecast_df = pd.DataFrame({
            "Week": range(1, horizon + 1),
            "Forecast": predictions
        })
        st.table(forecast_df)

    except Exception as e:
        st.error(f"⚠️ Error during prediction: {e}")
        st.info("Make sure your uploaded CSV has the same feature columns as the model was trained on.")
else:
    st.info("👆 Please upload a CSV file to continue.")
