import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Load Model and Data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("df.csv")  # Load dataset
        return df
    except FileNotFoundError:
        st.error("Dataset 'df.csv' not found.")
        return None

@st.cache_resource
def load_model():
    try:
        model = joblib.load("best_model.pkl")  # Load model
        return model
    except FileNotFoundError:
        st.error("Model file 'best_model.pkl' not found.")
        return None

# Initialize Data and Model
df = load_data()
model = load_model()

# Streamlit App
st.title("Customer Churn Prediction App")

# Sidebar Navigation
page = st.sidebar.selectbox("Select a page", ["Dashboard", "EDA", "ML"])

# Dashboard Page
if page == "Dashboard":
    st.header("Project Overview")
    st.write("This project aims to predict customer churn using machine learning.")
    
    st.subheader("Dataset Preview")
    if df is not None:
        st.dataframe(df.head())

    st.subheader("Model Status")
    if model:
        st.success("Model successfully loaded!")
    else:
        st.error("Model not found. Please check your files.")

# EDA Page
elif page == "EDA":
    st.header("Exploratory Data Analysis")

    if df is not None:
        # Show dataset info
        st.write(f"Dataset contains {df.shape[0]} rows and {df.shape[1]} columns.")

        # Select a feature for visualization
        feature = st.selectbox("Select a numerical feature", df.select_dtypes(include=np.number).columns)
        
        # Histogram
        st.subheader(f"Histogram of {feature}")
        fig, ax = plt.subplots()
        sns.histplot(df[feature], bins=30, kde=True, ax=ax)
        st.pyplot(fig)

        # Correlation Heatmap
        st.subheader("Feature Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax)
        st.pyplot(fig)

    else:
        st.error("Dataset not available.")

# ML Page
elif page == "ML":
    st.header("Make a Prediction")

    if model is not None and df is not None:
        st.subheader("Enter Feature Values")

        input_data = {}
        for col in df.columns[:-1]:  # Exclude target column
            if df[col].dtype == "object":
                input_data[col] = st.selectbox(f"Select {col}", df[col].unique())
            else:
                input_data[col] = st.number_input(f"Enter {col}", value=float(df[col].mean()))

        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])

        # Make Prediction
        if st.button("Predict"):
            prediction = model.predict(input_df)[0]
            st.success(f"Prediction: {'Churn' if prediction == 1 else 'Not Churn'}")
    else:
        st.error("Model or dataset not available.")
