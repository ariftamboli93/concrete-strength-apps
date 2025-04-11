import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/esvs2202/Concrete-Compressive-Strength-Prediction/refs/heads/main/dataset/concrete_data.csv"
    return pd.read_csv(url)

df = load_data()
X = df.drop('concrete_compressive_strength', axis=1)
y = df['concrete_compressive_strength']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

st.title("Concrete Compressive Strength Predictor")

cement = st.number_input("Cement (kg/m³)", min_value=0.0, value=540.0)
slag = st.number_input("Blast Furnace Slag (kg/m³)", min_value=0.0, value=0.0)
fly_ash = st.number_input("Fly Ash (kg/m³)", min_value=0.0, value=0.0)
water = st.number_input("Water (kg/m³)", min_value=0.0, value=162.0)
superplasticizer = st.number_input("Superplasticizer (kg/m³)", min_value=0.0, value=2.5)
coarse_agg = st.number_input("Coarse Aggregate (kg/m³)", min_value=0.0, value=1040.0)
fine_agg = st.number_input("Fine Aggregate (kg/m³)", min_value=0.0, value=676.0)
age = st.number_input("Age (days)", min_value=1, value=28)

if st.button("Predict Compressive Strength"):
    input_data = np.array([[cement, slag, fly_ash, water, superplasticizer, coarse_agg, fine_agg, age]])
    prediction = model.predict(input_data)
    st.success(f"Predicted Strength: {prediction[0]:.2f} MPa")
