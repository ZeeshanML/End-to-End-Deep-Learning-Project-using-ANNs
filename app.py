import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from tensorflow.keras.models import load_model
import tensorflow as tf 
import pickle

model = load_model('model.keras')

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('label_encoder_gender.pkl', 'rb') as f:
    encoder_gender = pickle.load(f)

with open('one_hot_encoder_geo.pkl', 'rb') as f:
    encoder_geo = pickle.load(f)

st.title('Customer Churn Prediction')

geography = st.selectbox("Geography", encoder_geo.categories_[0])
gender = st.selectbox("Gender", encoder_gender.classes_)
age = st.slider("Age", 18, 100, 30)
balance = st.number_input("Balance")
credit_score = st.number_input("Credit Score")
estimated_salary = st.number_input("Estimated Salary")
tenure = st.number_input("Tenure", 0, 10)
num_of_products = st.slider("Number of Products", 1, 4)
has_credit_card = st.selectbox("Has Credit Card", [0, 1])
is_active_member = st.selectbox("Is Active Member", [0, 1])

input_data = pd.DataFrame({
    'Gender': [encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_credit_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary],
    'CreditScore': [credit_score]
})

geo_encoded = encoder_geo.transform([[geography]])
geo_encoded_df = pd.DataFrame(geo_encoded, columns=encoder_geo.get_feature_names_out(['Geography']))

input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

expected_columns = scaler.feature_names_in_

input_data = input_data[expected_columns]

input_data_scaled = scaler.transform(input_data)

prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

st.write(f'Prediction Probability: {prediction_proba:.2f}')

if prediction_proba > 0.5:
    st.write('Customer is likely to churn')

else:
    st.write('Customer is not likely to churn')


