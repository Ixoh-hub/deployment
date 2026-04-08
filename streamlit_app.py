import os

import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = os.getenv("MODEL_PATH", "models/lightgbm_life_expectancy.pkl")


@st.cache_resource
def load_model(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path}")
    payload = joblib.load(path)
    return payload["model"], payload["features"]


model, feature_columns = load_model(MODEL_PATH)

st.title("Life Expectancy Prediction (LightGBM)")
st.write("Enter feature values to get a prediction from the trained model.")

input_data = {}
for col in feature_columns:
    input_data[col] = st.number_input(col, value=0.0)

if st.button("Predict"):
    X = pd.DataFrame([input_data], columns=feature_columns)
    prediction = model.predict(X, num_iteration=model.best_iteration)[0]
    st.success(f"Predicted Life Expectancy: {prediction:.2f} years")

st.markdown("### Model features")
st.write(feature_columns)
