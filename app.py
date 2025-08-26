import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the trained XGBoost model
try:
    model = joblib.load('saved_model/xgboost_model.pkl')
except FileNotFoundError:
    st.error("Model file not found. Please ensure 'xgboost_model.pkl' is in the 'saved_model' directory.")
    st.stop()

# --- Streamlit App ---

st.set_page_config(page_title="Fraud Detection System", layout="wide")

# Title of the dashboard
st.title("💳 Real-Time Transaction Fraud Detection")
st.write("This dashboard uses an XGBoost model to predict fraudulent credit card transactions.")

# --- Sidebar for user input ---
st.sidebar.header("Input Transaction Features")
st.sidebar.write("Enter the transaction details to check for fraud.")

# Collect user input features into a dictionary
input_features = {}
# The original dataset has 30 features (V1-V28, Scaled Amount, Scaled Time)
# We create sliders for a few key features for demonstration
# In a real app, you might have a form for a full transaction.
input_features['scaled_Amount'] = st.sidebar.slider('Scaled Amount', -1.0, 5.0, 0.5, 0.1)
input_features['scaled_Time'] = st.sidebar.slider('Scaled Time', -2.0, 2.0, 0.0, 0.1)
st.sidebar.subheader("Anonymized Features (V1-V28)")
for i in range(1, 29):
    input_features[f'V{i}'] = st.sidebar.slider(f'V{i}', -5.0, 5.0, 0.0, 0.1)

# Convert dictionary to a DataFrame
input_df = pd.DataFrame([input_features])

# --- Prediction and Display ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Transaction Details")
    st.write("The features below are used for prediction:")
    st.dataframe(input_df)

    if st.button('Check for Fraud', type="primary"):
        # Predict using the loaded model
        prediction = model.predict(input_df)
        prediction_proba = model.predict_proba(input_df)

        with col2:
            st.subheader("Prediction Result")
            if prediction[0] == 1:
                st.error("🚨 ALERT: This transaction is likely FRAUDULENT!")
                st.write(f"Confidence Score (Fraud): **{prediction_proba[0][1]*100:.2f}%**")
            else:
                st.success("✅ This transaction appears to be LEGITIMATE.")
                st.write(f"Confidence Score (Legitimate): **{prediction_proba[0][0]*100:.2f}%**")

# --- Display some sample suspicious transactions (for investigator alerts) ---
st.subheader("Investigator Alerts: Sample Suspicious Transactions")
st.write("Here are some transactions from the dataset that the model flagged as potentially fraudulent:")

# For demonstration, we'll just show some hardcoded high-risk examples.
# In a real system, this would be a live feed from your database.
suspicious_data = {
    'scaled_Amount': [3.9, 0.2, 1.5],
    'scaled_Time': [-1.5, 0.8, 1.2],
    'V4': [4.8, 6.1, 5.5],
    'V11': [9.2, 8.5, 7.8],
    'V12': [-11.5, -9.8, -10.2],
    'Prediction': ['Fraudulent', 'Fraudulent', 'Fraudulent']
}
suspicious_df = pd.DataFrame(suspicious_data)
st.dataframe(suspicious_df)