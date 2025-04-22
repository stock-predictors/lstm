import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
import joblib
from tensorflow.keras.models import load_model

# --- Caching Resources ---
@st.cache_resource
def load_data_and_models():
    data = pd.read_csv("MVS.csv", parse_dates=["DatetimeIndex"], index_col="DatetimeIndex")

    model_M = load_model("mastercard_lstm_model.h5")
    model_V = load_model("visa_lstm_model.h5")

    scaler_M = joblib.load("scaler_mastercard.save")
    scaler_V = joblib.load("scaler_visa.save")

    return data, model_M, model_V, scaler_M, scaler_V

data, model_M, model_V, scaler_M, scaler_V = load_data_and_models()

# --- Feature Engineering ---
data['MA10_V'] = data['Close_V'].rolling(window=10).mean()
data['MA20_V'] = data['Close_V'].rolling(window=20).mean()
data['Volatility_V'] = data['Close_V'].rolling(window=10).std()
data.dropna(inplace=True)

# --- Streamlit UI Setup ---
st.set_page_config(page_title="Visa & Mastercard Stocks", layout="wide")
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "📈 Predict", "ℹ️ Company Info"])

# --- Utility Functions ---
def plot_historical_prices():
    st.subheader("Stock Prices: 2008–2024")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['Close_M'], mode='lines', name='Mastercard'))
    fig.add_trace(go.Scatter(x=data.index, y=data['Close_V'], mode='lines', name='Visa'))
    fig.update_layout(title="Historical Close Prices", xaxis_title="Date", yaxis_title="Price (USD)")
    st.plotly_chart(fig, use_container_width=True)

def plot_volumes():
    st.subheader("Volume Traded: 2008–2024")
    fig = go.Figure()
    fig.add_trace(go.Bar(x=data.index, y=data['Volume_M'], name='Mastercard'))
    fig.add_trace(go.Bar(x=data.index, y=data['Volume_V'], name='Visa'))
    fig.update_layout(barmode='overlay', xaxis_title="Date", yaxis_title="Volume")
    st.plotly_chart(fig, use_container_width=True)

def create_sequences_lstm(input_data, seq_len=60):
    sequences = []
    last_data = input_data[-seq_len:]
    sequences.append(last_data)
    return np.array(sequences)

# --- Optimized Future Prediction ---
def make_future_prediction(date):
    future_days = (date - data.index[-1]).days
    if future_days <= 0:
        st.error("Please select a date beyond the last available data point (after June 2024).")
        return

    if future_days > 90:
        st.warning("Limiting prediction to 90 days ahead for performance.")
        future_days = 90

    # --- Mastercard ---
    scaled_M = scaler_M.transform(data[['Close_M']])
    x_input_M = create_sequences_lstm(scaled_M)

    preds_M = []
    for _ in range(future_days):
        next_pred = model_M.predict(x_input_M, verbose=0)[0][0]
        preds_M.append(next_pred)
        x_input_M = np.append(x_input_M[:, 1:, :], [[[next_pred]]], axis=1)

    pred_price_M = scaler_M.inverse_transform([[preds_M[-1]]])[0][0]

    # --- Visa ---
    scaled_V = scaler_V.transform(data[['Close_V', 'MA10_V', 'MA20_V', 'Volatility_V']])
    x_input_V = create_sequences_lstm(scaled_V)

    preds_V = []
    for _ in range(future_days):
        next_pred_v = model_V.predict(x_input_V, verbose=0)[0][0]
        preds_V.append(next_pred_v)
        x_input_V = np.append(x_input_V[:, 1:, :], [[[next_pred_v, 0, 0, 0]]], axis=1)

    pred_price_V = scaler_V.inverse_transform([[preds_V[-1], 0, 0, 0]])[0][0]

    return pred_price_M, pred_price_V

# --- Pages ---
if page == "🏠 Home":
    st.title("🏦 Visa & Mastercard - Stock Market Overview")
    plot_historical_prices()
    plot_volumes()
    st.success("Use the sidebar to navigate to the prediction page or company info.")

elif page == "📈 Predict":
    st.title("🔮 Predict Future Stock Prices")
    user_date = st.date_input("Select a future date (after June 2024)", min_value=datetime(2024, 6, 29), value=datetime(2025, 6, 1))

    if st.button("Predict Stock Prices"):
        pred_M, pred_V = make_future_prediction(user_date)
        st.success(f"📅 Predicted Price on {user_date.strftime('%Y-%m-%d')}")
        st.write(f"💳 **Visa**: ${pred_V:.2f}")
        st.write(f"💰 **Mastercard**: ${pred_M:.2f}")

        # Advice
        st.subheader("💡 Investment Advice")
        advice_M = "Buy" if pred_M < data['Close_M'].iloc[-1] else "Sell"
        advice_V = "Buy" if pred_V < data['Close_V'].iloc[-1] else "Sell"
        st.write(f"➡️ **Mastercard Advice**: {advice_M}")
        st.write(f"➡️ **Visa Advice**: {advice_V}")

        # Visualization
        future_date = pd.to_datetime(user_date)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data.index, y=data['Close_M'], name='Mastercard Historical', line=dict(color='green')))
        fig.add_trace(go.Scatter(x=[future_date], y=[pred_M], name='Mastercard Prediction', mode='markers+lines', line=dict(color='darkgreen', dash='dot')))

        fig.add_trace(go.Scatter(x=data.index, y=data['Close_V'], name='Visa Historical', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=[future_date], y=[pred_V], name='Visa Prediction', mode='markers+lines', line=dict(color='navy', dash='dot')))

        fig.update_layout(title="Stock Prices with Prediction", xaxis_title="Date", yaxis_title="Price (USD)")
        st.plotly_chart(fig, use_container_width=True)

elif page == "ℹ️ Company Info":
    st.title("ℹ️ About Visa and Mastercard")

    st.subheader("Visa Inc.")
    st.markdown("""
    Visa Inc. is a world leader in digital payments, facilitating transactions between consumers, merchants, and financial institutions across more than 200 countries.
    
    - **Ticker**: V  
    - **Market Cap**: $500B+  
    - **Founded**: 1958  
    - **Headquarters**: Foster City, California  
    """)

    st.subheader("Mastercard Inc.")
    st.markdown("""
    Mastercard is a global technology company in the payments industry. Their mission is to connect and power an inclusive digital economy.

    - **Ticker**: MA  
    - **Market Cap**: $400B+  
    - **Founded**: 1966  
    - **Headquarters**: Purchase, New York  
    """)
