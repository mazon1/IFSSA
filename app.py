import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.preprocessing import LabelEncoder

# Load the pre-trained model and data
try:
    model_pipeline = joblib.load("best_model.pkl")
    comparison_df = pd.read_csv("model_performance_comparison.csv")  # Load model performance data

    # Define feature names
    numerical_features = ["numerical_feature_1", "numerical_feature_2"]
    categorical_features = ["categorical_feature_1", "categorical_feature_2"]

    feature_names = numerical_features + categorical_features

except FileNotFoundError:
    st.error("Model file not found. Please ensure 'xgboost_tuned_model.pkl' and required files exist.")
    st.stop()

# Streamlit App
st.title("Customer Churn Prediction App")

# Navigation
page = st.sidebar.selectbox("Select a page", ["Dashboard", "EDA", "ML"])

# Dashboard Page
if page == "Dashboard":
    st.header("Project Overview")
    st.write("This project aims to predict customer churn using machine learning. "
             "We leverage an XGBoost model to identify factors contributing to churn and provide actionable insights.")

    st.subheader("Dataset Description")
    st.write("The dataset includes customer demographics, service usage patterns, and billing details.")

    st.subheader("Key Findings")
    st.write("- The model achieved an **F1-score** of 0.85 on the test set.")
    st.write("- Key churn factors include: **Service type, Monthly Charges, Contract Type**.")
    st.dataframe(comparison_df)

# EDA Page
elif page == "EDA":
    st.header("Exploratory Data Analysis")

    # User selects a numerical feature for histogram
    feature_to_plot = st.selectbox("Select a numerical feature to visualize", numerical_features)

    # Display histogram
    st.subheader(f"Histogram of {feature_to_plot}")
    df = pd.read_csv("your_dataset.csv")  # Load dataset
    fig, ax = plt.subplots()
    sns.histplot(df[feature_to_plot], bins=30, kde=True, ax=ax)
    st.pyplot(fig)

    # Correlation Heatmap
    st.subheader("Feature Correlation Heatmap")
    correlation_matrix = df[numerical_features].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax)
    st.pyplot(fig)

# ML Page (Model Prediction)
elif page == "ML":
    st.header("Machine Learning Prediction")

    # User Inputs for Prediction
    st.subheader("Enter Feature Values")

    input_data = {}

    for feature in feature_names:
        if feature in numerical_features:
            input_data[feature] = st.number_input(f"Enter {feature}", value=0.0)
        else:
            unique_values = ["Option A", "Option B"]  # Replace with actual categories from your dataset
            input_data[feature] = st.selectbox(f"Select {feature}", unique_values)

    # Convert input into DataFrame
    input_df = pd.DataFrame([input_data])

    # Preprocess inputs before prediction (Apply Label Encoding for Categorical Features)
    for cat_feature in categorical_features:
        encoder = LabelEncoder()
        input_df[cat_feature] = encoder.fit_transform(input_df[cat_feature])

    # Make Prediction
    if st.button("Predict Churn"):
        try:
            prediction = model_pipeline.predict(input_df)[0]
            probability = model_pipeline.predict_proba(input_df)[0, 1]  # Probability of churn

            st.success(f"Prediction: {'Churn' if prediction == 1 else 'Not Churn'}")
            st.info(f"Probability of Churn: {probability:.2f}")

        except Exception as e:
            st.error(f"Error during prediction: {e}")

    # Feature Importance
    st.subheader("Feature Importance")
    try:
        feature_importance = model_pipeline.named_steps['classifier'].feature_importances_
        importance_df = pd.DataFrame({"Feature": feature_names, "Importance": feature_importance})
        importance_df = importance_df.sort_values(by="Importance", ascending=False)
        st.bar_chart(importance_df.set_index("Feature"))

    except AttributeError:
        st.error("Feature importance is not available for the current model.")
