import streamlit as st
import pandas as pd
import pickle as pk

with open("salary.pickle", "rb") as f:
    model = pk.load(f)

st.title("Salary Prediction")
age = st.number_input("Enter your age:", min_value=18, max_value=60, value=30)
exp = st.number_input("Enter your years of experience:", min_value=0, max_value=40, value=0)
edu = st.selectbox("Select your education level:", ["Bachelors", "Masters", "PhD"])

if edu == "Bachelors":
    b = 1; m = 0; p = 0
elif edu == "Masters":
    b = 0; m = 1; p = 0
else:
    b = 0; m = 0; p = 1

if st.button("Predict Salary"):
    data = pd.DataFrame(
        {"Age": [age],
          "Years of Experience": [exp],
          "Bachelor's": [b],
          "Master's": [m],
          "PhD": [p]}
        )
    prediction = model.predict(data)
    st.write(f"Predicted Salary: ${prediction[0]:.2f}")