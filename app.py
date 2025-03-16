import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the trained model with caching
@st.cache_resource
def load_model():
    try:
        return joblib.load('model.joblib')
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# Define the required columns (ensure this matches training data)
REQUIRED_COLUMNS = [
    'Creator', 'agent_related', 'hamper_type', 'household', 'dw', 'is_single_pickup',
    'avg_days_between_pickups', 'sex_Male', 'status_married', 'location_cluster_2',
    'location_cluster_3', 'location_cluster_4', 'location_cluster_5', 'location_cluster_6',
    'location_cluster_7'
]

# Function to preprocess input data
def preprocess_input(input_data):
    input_df = pd.DataFrame([input_data])

    # Ensure all required columns exist
    for col in REQUIRED_COLUMNS:
        if col not in input_df.columns:
            input_df[col] = 0  # Set missing columns to 0

    # Ensure correct column order
    input_df = input_df[REQUIRED_COLUMNS]

    # Convert all data to numerical format
    input_df = input_df.astype(float)

    return input_df

# Streamlit app UI
st.title("Hamper Return Prediction App")
st.write("Fill in the details below to predict if a client will return.")

# User input fields
creator = st.number_input("Creator", min_value=1, value=1, step=1)
agent_related = st.number_input("Agent Related", min_value=1, value=1, step=1)
hamper_type = st.number_input("Hamper Type", min_value=1, value=1, step=1)
household = st.number_input("Household", min_value=1, value=1, step=1)
dw = st.selectbox("DW", ["yes", "no"])
is_single_pickup = st.selectbox("Is Single Pickup", [0, 1])  # Assuming 0/1 encoding
avg_days_between_pickups = st.number_input("Avg Days Between Pickups", min_value=1, value=7)
sex_male = st.selectbox("Sex (Male)", [0, 1])  # Assuming 0/1 encoding
status_married = st.selectbox("Status (Married)", [0, 1])  # Assuming 0/1 encoding

# Location cluster selection
location_cluster = st.selectbox("Location Cluster", [2, 3, 4, 5, 6, 7])

# Prepare input data
input_data = {
    'Creator': creator,
    'agent_related': agent_related,
    'hamper_type': hamper_type,
    'household': household,
    'dw': 1 if dw == "yes" else 0,  # Convert 'dw' to numerical
    'is_single_pickup': is_single_pickup,
    'avg_days_between_pickups': avg_days_between_pickups,
    'sex_Male': sex_male,
    'status_married': status_married,
    'location_cluster_2': 1 if location_cluster == 2 else 0,
    'location_cluster_3': 1 if location_cluster == 3 else 0,
    'location_cluster_4': 1 if location_cluster == 4 else 0,
    'location_cluster_5': 1 if location_cluster == 5 else 0,
    'location_cluster_6': 1 if location_cluster == 6 else 0,
    'location_cluster_7': 1 if location_cluster == 7 else 0,
}

# Prediction
if st.button("Predict"):
    if model is None:
        st.error("Model not loaded. Please check if 'model.joblib' exists.")
    else:
        try:
            input_df = preprocess_input(input_data)
            
            # Check for NaN values
            if input_df.isnull().values.any():
                st.error("Input contains missing values. Please check your inputs.")

            # Check for unexpected column types
            if not all(input_df.dtypes == float):
                st.error("Input data has incorrect types. Ensure all inputs are numeric.")

            # Make prediction
            prediction = model.predict(input_df)
            probability = model.predict_proba(input_df)

            st.subheader("Prediction Result:")
            st.write("✅ Prediction: **Yes**" if prediction[0] == 1 else "❌ Prediction: **No**")
            st.write(f"📊 Probability (Yes): **{probability[0][1]:.4f}**")
            st.write(f"📊 Probability (No): **{probability[0][0]:.4f}**")
        except Exception as e:
            st.error(f"Prediction error: {e}")
