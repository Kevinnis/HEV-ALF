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
INR = st.number_input("International normalized ratio", min_value=0.0, max_value=100.0, format="%.2f")

TBIL = st.number_input("Total bilirubin (μmol/L)", min_value=0.0, max_value=10000.0, format="%.2f")

ALB = st.number_input("Albumin (g/L)", min_value=0.0, max_value=10000.0, format="%.2f")

NEU = st.number_input("Neutrophil count (10^9/L)", min_value=0.0, max_value=10000.0, format="%.2f")

AST = st.number_input("Aspartate aminotransferase (U/L)", min_value=0.0, max_value=10000.0, format="%.2f")

UREA = st.number_input("Urea (mmol/L)", min_value=0.0, max_value=10000.0, format="%.2f")


feature_values = [INR,TBIL,ALB,NEU,AST,UREA]

features = np.array([feature_values])

center_button = st.markdown("""
    <style>
    div.stButton > button:first-child {
        display: block;
        margin: 0 auto;
    }
    </style>""", unsafe_allow_html=True) #预测按钮居中
    
if st.button("Predict"):    
    # 预测风险评分
    risk_score = model.predict(features)[0]
    
    # 显示 Risk Score
    st.markdown(f"<h3 style='text-align: center;'>Risk Score: {risk_score:.4f}</h3>", unsafe_allow_html=True)
    
    # 计算并显示 7-day 和 14-day HEV-ALF onset risk
    if risk_score >= 2.787183:
        st.markdown("<h3 style='text-align: center; color: red;'>7-day HEV-ALF onset risk: High-risk</h3>", unsafe_allow_html=True)
    else:
        st.markdown("<h3 style='text-align: center; color: green;'>7-day HEV-ALF onset risk: Low-risk</h3>", unsafe_allow_html=True)
        
    if risk_score >= 2.640324:
        st.markdown("<h3 style='text-align: center; color: red;'>14-day HEV-ALF onset risk: High-risk</h3>", unsafe_allow_html=True)
    else:
        st.markdown("<h3 style='text-align: center; color: green;'>14-day HEV-ALF onset risk: Low-risk</h3>", unsafe_allow_html=True)

    
    
    
    
    
    
    
    
    
    