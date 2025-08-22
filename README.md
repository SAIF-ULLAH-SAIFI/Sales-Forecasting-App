# Sales-Forecasting-App
📊 Sales Forecasting App

This project is a Sales Forecasting Web Application built with Streamlit and a trained Machine Learning model.
It predicts future sales based on past data and provides a simple interactive dashboard.

🚀 Features

Upload / input sales data

Forecast future sales using ML model (sales_forecast.pkl)

Interactive charts and clean UI with Streamlit

Easy deployment on Streamlit Cloud or HuggingFace Spaces

🛠️ Tech Stack

Python 3.9+

Streamlit

scikit-learn

pandas, numpy

matplotlib / seaborn

📂 Project Structure
sales_forecasting_flask/
│── data/                # Sample dataset
│── models/              # Trained ML model (sales_forecast.pkl)
│── static/              # CSS / Images
│── templates/           # HTML templates (for Flask version)
│── app.py               # Flask app (old version)
│── streamlit_app.py     # Streamlit app (new version)
│── train.py             # Model training script
│── utils.py             # Helper functions
│── requirements.txt     # Dependencies
│── README.md            # Project description

⚡ Installation & Usage
1️⃣ Clone Repository
git clone https://github.com/YourUsername/sales_forecasting_app.git
cd sales_forecasting_app

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Run Streamlit App
streamlit run streamlit_app.py


It will start a local server at:
👉 http://localhost:8501

🌍 Deployment

You can deploy this project easily on:

Streamlit Cloud

HuggingFace Spaces

Just upload your repo and select streamlit_app.py as the entrypoint.
 

🙌 Acknowledgements

Streamlit

scikit-learn

Special thanks to Elevvo Tech
 for guidance.
