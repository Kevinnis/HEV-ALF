import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sksurv.ensemble import RandomSurvivalForest
import shap

# Load the model
model = joblib.load('rsf.mod1.pkl')

# Define feature names
feature_names = ["NEU", "ALB", "AST", "TBIL", "UREA", "INR"]

# Streamlit user interface
st.title("HEV-ALF Predictor")

# Custom title font
st.markdown("<h2 style='font-weight: bold;'>Predicting the risk of HEV-ALF onset among hospitalized patients with acute hepatitis E</h2>", unsafe_allow_html=True)

# Numerical input
INR = st.number_input("International normalized ratio", min_value=0.0, max_value=100.0, format="%.2f")
TBIL = st.number_input("Total bilirubin (μmol/L)", min_value=0.0, max_value=10000.0, format="%.2f")
ALB = st.number_input("Albumin (g/L)", min_value=0.0, max_value=10000.0, format="%.2f")
NEU = st.number_input("Neutrophil count (10^9/L)", min_value=0.0, max_value=10000.0, format="%.2f")
AST = st.number_input("Aspartate aminotransferase (U/L)", min_value=0.0, max_value=10000.0, format="%.2f")
UREA = st.number_input("Urea (mmol/L)", min_value=0.0, max_value=10000.0, format="%.2f")

feature_values = [NEU, ALB, AST, TBIL, UREA, INR]
features = np.array([feature_values])

# Center the predict button
st.markdown("""
    <style>
    div.stButton > button:first-child {
        display: block;
        margin: 0 auto;
    }
    </style>""", unsafe_allow_html=True)

if st.button("Predict"):    
    # Predict risk score
    risk_score = model.predict(features)[0]
    
    # Display Risk Score
    st.markdown(f"<h3 style='text-align: center;'>Risk Score: {risk_score:.4f}</h3>", unsafe_allow_html=True)
    
    # Display 7-day and 14-day HEV-ALF onset risk
    if risk_score >= 2.787183:
        st.markdown("<h3 style='text-align: center; color: red;'>7-day HEV-ALF onset risk: High-risk</h3>", unsafe_allow_html=True)
    else:
        st.markdown("<h3 style='text-align: center; color: green;'>7-day HEV-ALF onset risk: Low-risk</h3>", unsafe_allow_html=True)
        
    if risk_score >= 2.640324:
        st.markdown("<h3 style='text-align: center; color: red;'>14-day HEV-ALF onset risk: High-risk</h3>", unsafe_allow_html=True)
    else:
        st.markdown("<h3 style='text-align: center; color: green;'>14-day HEV-ALF onset risk: Low-risk</h3>", unsafe_allow_html=True)

    # Function for SHAP predictions
    def predict_fn(X):
        return model.predict(X)
    
    # Compute SHAP values
    explainer = shap.Explainer(predict_fn, pd.DataFrame([feature_values], columns=feature_names))
    shap_values = explainer(pd.DataFrame([feature_values], columns=feature_names))
    
    # Create a plot with a specific figure size to reduce the image size
    plt.figure(figsize=(5, 5))  # Adjust the figsize to smaller dimensions

    # Plot and display SHAP values
    shap.waterfall_plot(shap_values[0], max_display=len(feature_names))

    # Save the plot with adjusted size and dpi
    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=100)
    st.image("shap_force_plot.png")
    
    
    
    
    
