import os
import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = os.getenv('MODEL_PATH', 'models/lightgbm_life_expectancy.pkl')

@st.cache_data
def load_model(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path}")
    payload = joblib.load(path)
    return payload['model'], payload['features'], payload.get('metadata', {})

model, feature_columns, metadata = load_model(MODEL_PATH)

st.set_page_config(page_title='Life Expectancy Predictor', page_icon='🌍', layout='wide')
st.title('🌍 Life Expectancy Prediction')
st.caption('Production-style demo UI for the trained LightGBM model')

feature_ranges = metadata.get('feature_ranges', {})
metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
metrics_col1.metric('Features', len(feature_columns))
metrics_col2.metric('Train RMSE', f"{metadata.get('rmse', 0):.3f}" if metadata.get('rmse') is not None else 'N/A')
metrics_col3.metric('Train R²', f"{metadata.get('r2', 0):.3f}" if metadata.get('r2') is not None else 'N/A')

st.markdown('### Input features')
input_data = {}
left, right = st.columns(2)
for idx, col in enumerate(feature_columns):
    default = 0.0
    min_val = None
    max_val = None
    if col in feature_ranges:
        feature_info = feature_ranges[col]
        default = feature_info.get('median', 0.0)
        min_val = feature_info.get('min')
        max_val = feature_info.get('max')

    target_col = left if idx % 2 == 0 else right
    with target_col:
        input_data[col] = st.number_input(
            label=col,
            value=float(default),
            min_value=float(min_val) if min_val is not None else None,
            max_value=float(max_val) if max_val is not None else None,
            help='Auto-filled with median value from training data where available.',
        )

if st.button('Predict life expectancy', type='primary', use_container_width=True):
    X = pd.DataFrame([input_data], columns=feature_columns)
    prediction = model.predict(X)[0]
    st.success(f'Predicted Life Expectancy: {prediction:.2f} years')

with st.expander('Show model feature list'):
    st.write(feature_columns)
