import streamlit as st
import pickle
import pandas as pd
import numpy as np

# 1. Load the model and encoders
with open('safety_model.pkl', 'rb') as f:
    data = pickle.load(f)

model = data['model']
encoders = data['encoders']
feature_names = data['feature_names']

# UI Design
st.set_page_config(page_title="Accident Predictor", page_icon="🚗")
st.title("🚗 Road Accident Severity Predictor")
st.write("Enter details below to predict severity:")

# 2. Creating Input Fields
user_inputs = {}
col1, col2 = st.columns(2)

for i, col in enumerate(feature_names):
    target_col = col1 if i % 2 == 0 else col2
    if col in encoders:
        options = list(encoders[col].classes_)
        selected = target_col.selectbox(f"Select {col}", options)
        user_inputs[col] = encoders[col].transform([selected])[0]
    else:
        val = target_col.number_input(f"Enter {col}", value=0)
        user_inputs[col] = val

# 3. Prediction
if st.button("Predict Severity"):
    input_df = pd.DataFrame([user_inputs])[feature_names]
    prediction = model.predict(input_df)
    result = encoders['Severity'].inverse_transform(prediction)[0]
    
    st.divider()
    if result == 'Fatal':
        st.error(f"The predicted severity is: {result}")
    elif result == 'Serious':
        st.warning(f"The predicted severity is: {result}")
    else:
        st.success(f"The predicted severity is: {result}")