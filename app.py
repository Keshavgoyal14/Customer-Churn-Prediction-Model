import streamlit as st
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf

# Load model
model = tf.keras.models.load_model("churn_model.h5")

# Load encoders and scaler
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

with open("one_hot_encoder.pkl", "rb") as f:
    one_hot_encoder = pickle.load(f)

with open("scalar.pkl", "rb") as f:
    scaler = pickle.load(f)

st.title("Customer Churn Prediction")

# User input
credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=600)
geography = st.selectbox("Geography", options=["France", "Spain", "Germany"])
gender = st.selectbox("Gender", options=["Male", "Female"])
age = st.number_input("Age", min_value=18, max_value=100, value=30)
tenure = st.number_input("Tenure", min_value=0, max_value=10, value=3)
balance = st.number_input("Balance", min_value=0.0, value=10000.0)
num_of_products = st.number_input("Number of Products", min_value=1, max_value=4, value=1)
has_cr_card = st.selectbox("Has Credit Card", options=[0, 1])
is_active_member = st.selectbox("Is Active Member", options=[0, 1])
estimated_salary = st.number_input("Estimated Salary", min_value=0.0, value=50000.0)

# Base input (Gender label-encoded)
input_df = pd.DataFrame([{
    "CreditScore": credit_score,
    "Gender": int(label_encoder.transform([gender])[0]),
    "Age": age,
    "Tenure": tenure,
    "Balance": balance,
    "NumOfProducts": num_of_products,
    "HasCrCard": has_cr_card,
    "IsActiveMember": is_active_member,
    "EstimatedSalary": estimated_salary
}])

# One-hot encode Geography
geo_col = "Geography"
geo_input = pd.DataFrame([{geo_col: geography}])

geo_transformed = one_hot_encoder.transform(geo_input[[geo_col]])
geo_values = geo_transformed.toarray() if hasattr(geo_transformed, "toarray") else np.asarray(geo_transformed)

geo_encoded_df = pd.DataFrame(
    geo_values,
    columns=one_hot_encoder.get_feature_names_out([geo_col])
)

# Combine
input_df = pd.concat([input_df.reset_index(drop=True), geo_encoded_df.reset_index(drop=True)], axis=1)

# Align column order with scaler if available
if hasattr(scaler, "feature_names_in_"):
    input_df = input_df.reindex(columns=scaler.feature_names_in_, fill_value=0)

# Scale
input_scaled = scaler.transform(input_df)

# Predict
if st.button("Predict Churn"):
    prediction = model.predict(input_scaled, verbose=0)
    prediction_proba = float(prediction[0][0])

    st.write(f"Churn Probability: {prediction_proba:.2f}")
    if prediction_proba > 0.5:
        st.write("The customer is likely to churn.")
    else:
        st.write("The customer is not likely to churn.")