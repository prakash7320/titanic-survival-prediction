import streamlit as st
import joblib
import pandas as pd

model = joblib.load("model.pkl")
encoder = joblib.load("encoder.pkl")

st.title("🚢 Titanic Survival Prediction")

pclass = st.selectbox("Passenger Class", [1,2,3])

sex = st.selectbox("Gender", ["male","female"])

age = st.number_input("Age",1,80,25)

sibsp = st.number_input("Siblings/Spouse",0,10,0)

parch = st.number_input("Parents/Children",0,10,0)

fare = st.number_input("Fare",0.0,600.0,32.0)

embarked = st.selectbox("Embarked",["C","Q","S"])

if st.button("Predict"):

    sex = encoder["Sex"].transform([sex])[0]
    embarked = encoder["Embarked"].transform([embarked])[0]

    data = pd.DataFrame([[pclass,sex,age,sibsp,parch,fare,embarked]],
        columns=[
            "Pclass",
            "Sex",
            "Age",
            "SibSp",
            "Parch",
            "Fare",
            "Embarked"
        ])

    prediction = model.predict(data)[0]

    if prediction == 1:
        st.success("🎉 Passenger Survived")
    else:
        st.error("❌ Passenger Did Not Survive")