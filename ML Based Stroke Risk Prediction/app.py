# importing the libraries 
import streamlit as st 
import pickle as pkl 
import pandas as pd
import numpy as np 

# having a wide page layout
st.set_page_config(layout="wide")

# Title of the page
st.title("ML Based Stroke Risk Prediction")

# Importing data and model from pickle
model = pkl.load(open('model.pkl','rb'))

# First Row and columns 
col1, col2, col3, col4 = st.columns(4)
with col1: 
    age = st.selectbox('Age', sorted(range(29,77)), key="age")
with col2: 
    sex = st.selectbox('Sex (0 = Female, 1 = Male)', sorted(range(0, 2)), key="sex")
with col3: 
    cp = st.selectbox('Chest pain Level ', sorted(range(0, 4)), key="cp")
with col4: 
    trestbps = st.selectbox('Resting blood pressure (in mm Hg)', sorted(range(94,201)), key="trestbps")

col5, col6, col7, col8 = st.columns(4)
with col5: 
    chol = st.selectbox('Serum cholesterol (in mg/dl)', sorted(range(126,565)), key="chol")
with col6: 
    fbs = st.selectbox('Fasting blood sugar (> 120 mg/dl, 1 = true, 0 = false)', sorted(range(0, 2)), key="fbs")
with col7: 
    restecg = st.selectbox('Resting electrocardiographic results ', sorted(range(0, 3)), key="restecg")
with col8: 
    thalach = st.selectbox('Maximum heart rate achieved', sorted(range(71, 202)), key="thalach")

list1=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 
 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 
 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 
 3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 
 4.0, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 
 5.0, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 
 6.0, 6.1, 6.2]


col9, col10, col11, col12 = st.columns(4)
with col9: 
    exang = st.selectbox('Exercise-induced angina (1 = yes, 0 = no)', sorted(range(0, 2)), key="exang")
with col10: 
    oldpeak = st.selectbox('ST depression induced by exercise relative to rest ', list1, key="oldpeak")
with col11: 
    slope = st.selectbox('Slope of the peak exercise ST segment ', sorted(range(0, 3)), key="slope")
with col12: 
    ca = st.selectbox('Number of major vessels (0-3) colored by fluoroscopy', sorted(range(0, 4)), key="ca")

col13 = st.columns(1)[0]  # Get the first (and only) column from the list
with col13: 
    thal = st.selectbox('Thalassemia type (1 = normal, 2 = fixed defect, 3 = reversible defect)', sorted(range(0, 4)), key="thal")

# Now, you can use the model for prediction or other operations
input_values = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]

# Convert the list into a NumPy array
input_array = np.array(input_values).reshape(1, -1)  # Reshape to make it 2D as required by model

if st.button('Predict Probabilities'):
    prediction = model.predict(input_array)  # Use input_array instead of input_values
    if prediction[0] == 0:
        st.header("Good News, the patient doesn't have any heart disease.")
    else: 
        st.header("The patient should visit the doctor.")
