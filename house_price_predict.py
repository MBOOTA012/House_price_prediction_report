import streamlit as st
import numpy as np
import pickle

# Load the model and scaler
with open('house_price_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('house_price_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Define feature names (must match training order)
feature_names = [
    'longitude', 'latitude', 'housing_median_age',
    'population', 'households', 'median_income',
    'rooms_per_household', 'bedrooms_per_room', 'population_per_household',
    'log_total_rooms', 'log_total_bedrooms', 'log_population'
]

# Streamlit UI
st.title("🏠 House Price Prediction App")

st.markdown("Enter the features to predict the house price:")

# Collect input
input_values = []
for feature in feature_names:
    value = st.number_input(f"{feature.replace('_', ' ').capitalize()}:", value=0.0)
    input_values.append(value)

# Predict
if st.button("Predict Price"):
    try:
        input_array = np.array([input_values])
        input_scaled = scaler.transform(input_array)
        predicted_log_price = model.predict(input_scaled)
        predicted_price = np.expm1(np.clip(predicted_log_price, None, 20))
        st.success(f"💰 Predicted House Price: ${predicted_price[0]:,.2f}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
