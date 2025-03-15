import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load trained model
@st.cache_resource
def load_model():
    with open("best_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

# Define user inputs
st.title("Hamper Pickup Prediction App")
st.write("Enter details to predict the outcome of a hamper pickup.")

# User inputs
dependents_qty = st.number_input("Number of Dependents", min_value=0, max_value=10, step=1)
total_visits = st.number_input("Total Visits", min_value=1, max_value=100, step=1)
avg_days_between_pickups = st.number_input("Avg Days Between Pickups", min_value=1.0, max_value=100.0, step=0.1)
distance_to_center = st.number_input("Distance to Center (km)", min_value=0.0, max_value=50.0, step=0.1)

# Categorical inputs
sex = st.selectbox("Sex", ["Male", "Female"])
status = st.selectbox("Client Status", ["Active", "Closed", "Flagged", "Outreach", "Pending"])
location_cluster = st.selectbox("Location Cluster", [1, 2, 3, 4, 5])

# Convert categorical values
sex_encoded = 1 if sex == "Male" else 0
status_encoded = ["Active", "Closed", "Flagged", "Outreach", "Pending"].index(status)
location_cluster_encoded = int(location_cluster)

# Prepare input data for model
input_data = np.array([[dependents_qty, total_visits, avg_days_between_pickups, 
                         distance_to_center, sex_encoded, status_encoded, location_cluster_encoded]])

# Make Prediction
if st.button("Predict"):
    prediction = model.predict(input_data)
    st.success(f"Predicted Outcome: {prediction[0]}")

