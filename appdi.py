import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

## Carregando o modelo treinado
modeldi=tf.keras.models.load_model('modeldi.h5')

## Carregando o encoder e o scaler
with open('onehot_encoder_geo.pkl','rb') as file:
    onehot_encoder_smoking=pickle.load(file)

with open('label_encoder_gender.pkl','rb') as file:
    label_encoder_genderdi=pickle.load(file)

with open('scaler.pkl','rb') as file:
    scalerdi=pickle.load(file)


## Streamlit app
st.title('NN Diabetes')

smoking = st.selectbox('Smoking History', onehot_encoder_smoking.categories_[0])
gender = st.selectbox('Gender', label_encoder_genderdi)
age = st.selectbox('Age', 18, 92)
hypertension = st.number_input('Hypertension')
heart_desease = st.number_input('Heart Desease')
bmi = st.number_input('BMI')
HbA1c_level = st.number_input('HbA1c Level')
blood_glucose_level = st.number_input('Blood Glucose Level')


input_data=pd.DataFrame({
    'gender': [label_encoder_genderdi.transform([gender])[0]],
    'age': [age],
    'hypertension': [hypertension],
    'heart_disease': [heart_desease],
    'bmi': [bmi],
    'HbA1c_level': [HbA1c_level],
    'blood_glucose_level': [blood_glucose_level]  
})

## Codificando o input de geography
smoking_encoded=onehot_encoder_smoking.transform([[smoking]]).toarray()
smoking_encoded_df=pd.DataFrame(smoking_encoded, columns=onehot_encoder_smoking.get_feature_names_out(['smoking_history']))

## Juntando com o input
input_data = pd.concat([input_data.reset_index(drop=True), smoking_encoded_df], axis=1)

## Scaling input data
input_data_scaled = scalerdi.transform(input_data)

## Prediction
prediction = modeldi.predict(input_data_scaled)
prediction_proba = prediction[0][0]

st.write(f'Diabetes Probability: {prediction_proba:.2f}')

if prediction_proba > 0.5:
    st.write('Diabético')
else:

    st.write('Não Diabético')
