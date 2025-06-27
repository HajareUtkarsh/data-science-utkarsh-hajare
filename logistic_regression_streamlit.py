import streamlit as st
import numpy as np
import pickle

# Load the trained model
model = pickle.load(open('titanic_model.pkl', 'rb'))

st.title("Titanic Survival Predictor")

# Get user input
pclass = st.selectbox("Passenger Class", [1, 2, 3])
age = st.slider("Age", 1, 80, 30)
fare = st.slider("Fare", 0, 500, 50)
sex = st.selectbox("Sex", ["male", "female"])
embarked = st.selectbox("Embarked", ["C", "Q", "S"])

# Convert inputs to match model features
sex_male = 1 if sex == "male" else 0
embarked_Q = 1 if embarked == "Q" else 0
embarked_S = 1 if embarked == "S" else 0

features = np.array([[pclass, age, fare, sex_male, embarked_Q, embarked_S]])

# Make prediction
prediction = model.predict(features)

# Show result
if prediction[0] == 1:
    st.success("Passenger would survive.")
else:
    st.error("Passenger would not survive.")
