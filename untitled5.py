import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sksurv.ensemble import RandomSurvivalForest

# Load the model
model = joblib.load('rsf.mod1.pkl')
# Define feature names
feature_names = ["NEU", "ALB", "AST", "TBIL", "UREA", "INR"]
# Streamlit user interface
st.title("HEV-ALF Predictor")

#  numerical input
INR = st.number_input("INR:", min_value=0, max_value=100)

TBIL = st.number_input("TBIL:", min_value=0, max_value=10000)

ALB = st.number_input("ALB:", min_value=0, max_value=10000)

NEU = st.number_input("NEU:", min_value=0, max_value=10000)

AST = st.number_input("AST:", min_value=0, max_value=10000)

UREA = st.number_input("UREA:", min_value=0, max_value=10000)

feature_values = [INR,TBIL,ALB,NEU,AST,UREA]

features = np.array([feature_values])

if st.button("Predict"):    
    # Predict class and probabilities    
    predicted_class = model.predict(features)[0]    
    predicted_proba = model.predict(features)[0]
    # Display prediction results    
    st.write(f"**Predicted Class:** {predicted_class}")    
    st.write(f"**Prediction Probabilities:** {predicted_proba}")
    


    
    
    
    
    
    
    
    
    
    