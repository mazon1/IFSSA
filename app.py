import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load trained model and preprocessor
@st.cache_resource
def load_model():
    with open("best_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

@st.cache_resource
def load_preprocessor():
    try:
        with open("preprocessor.pkl", "rb") as f:
            preprocessor = pickle.load(f)
        return preprocessor
    except FileNotFoundError:
        return None  # If no preprocessor was used, proceed without transformation

model = load_model()
preprocessor = load_preprocessor()

# Define user inputs
st.title("Hamper Pickup Prediction App")
st.write("Enter details to predict the likelihood of a hamper pickup.")

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
status_mapping = {"Active": 0, "Closed": 1, "Flagged": 2, "Outreach": 3, "Pending": 4}
status_encoded = status_mapping[status]
location_cluster_encoded = int(location_cluster)

# Prepare input data for model
input_data = np.array([[dependents_qty, total_visits, avg_days_between_pickups, 
                         distance_to_center, sex_encoded, status_encoded, location_cluster_encoded]])

# Apply preprocessing (if available)
if preprocessor:
    input_data = preprocessor.transform(input_data)

# Make Prediction
if st.button("Predict"):
    prediction = model.predict(input_data)
    st.success(f"Predicted Outcome: {prediction[0]}")

# Model Explanation
if st.checkbox("Show Model Details"):
    st.write("This model was trained on historical hamper pickup data, using features such as client visit history, demographics, and pickup locations.")
