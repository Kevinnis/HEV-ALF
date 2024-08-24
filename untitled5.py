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
# 设置自定义标题字体
st.markdown("<h2 style='font-weight: bold;'>Predicting the risk of HEV-ALF onset among hospitalized patients with acute hepatitis E</h2>", unsafe_allow_html=True)

import streamlit as st

# 设置每个数值输入的格式为两位小数
INR = st.number_input("International Normalized Ratio", min_value=0.0, max_value=100.0, format="%.2f")

TBIL = st.number_input("Total bilirubin (μmol/L)", min_value=0.0, max_value=10000.0, format="%.2f")

ALB = st.number_input("Albumin (g/L)", min_value=0.0, max_value=10000.0, format="%.2f")

NEU = st.number_input("Neutrophil count (10^9/L)", min_value=0.0, max_value=10000.0, format="%.2f")

AST = st.number_input("Aspartate aminotransferase (U/L)", min_value=0.0, max_value=10000.0, format="%.2f")

UREA = st.number_input("Urea (mmol/L)", min_value=0.0, max_value=10000.0, format="%.2f")


feature_values = [INR,TBIL,ALB,NEU,AST,UREA]

features = np.array([feature_values])

if st.button("Predict"):    
    # 预测风险评分
    risk_score = model.predict(features)[0]
    
    # 显示 Risk Score
    st.write(f"**Risk Score:** {risk_score:.4f}")
    
    # 计算并显示 7-day 和 14-day HEV-ALF onset risk
    if risk_score >= 2.787183:
        st.write("**7-day HEV-ALF onset risk:** High-risk")
    else:
        st.write("**7-day HEV-ALF onset risk:** Low-risk")
        
    if risk_score >= 2.640324:
        st.write("**14-day HEV-ALF onset risk:** High-risk")
    else:
        st.write("**14-day HEV-ALF onset risk:** Low-risk")


    
    
    
    
    
    
    
    
    
    