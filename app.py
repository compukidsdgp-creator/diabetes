# pip install streamlit pandas scikit-learn seaborn matplotlib pillow

# IMPORTS
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# LOAD DATA
df = pd.read_csv("diabetes.csv")

# TITLE
st.title("Diabetes Prediction System")
st.sidebar.header("Enter Patient Data")

# DATA OVERVIEW
st.subheader("Training Dataset Statistics")
st.write(df.describe())

# FEATURES AND LABEL
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# SPLIT DATA
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# USER INPUT FUNCTION
def user_report():

    pregnancies = st.sidebar.slider("Pregnancies", 0, 17, 3)
    glucose = st.sidebar.slider("Glucose", 0, 200, 120)
    bloodpressure = st.sidebar.slider("Blood Pressure", 0, 122, 70)
    skinthickness = st.sidebar.slider("Skin Thickness", 0, 100, 20)
    insulin = st.sidebar.slider("Insulin", 0, 846, 79)
    bmi = st.sidebar.slider("BMI", 0.0, 67.0, 25.0)
    dpf = st.sidebar.slider("Diabetes Pedigree Function", 0.0, 2.5, 0.5)
    age = st.sidebar.slider("Age", 21, 90, 33)

    user_data = {
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": bloodpressure,
        "SkinThickness": skinthickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": dpf,
        "Age": age
    }

    report_data = pd.DataFrame(user_data, index=[0])
    return report_data


# GET USER INPUT
user_data = user_report()

st.subheader("Patient Input Data")
st.write(user_data)

# MODEL TRAINING
rf = RandomForestClassifier()
rf.fit(x_train, y_train)

# PREDICTION
user_result = rf.predict(user_data)

# VISUALIZATION TITLE
st.header("Patient Data Visualization")

# COLOR FOR USER POINT
color = "blue" if user_result[0] == 0 else "red"

# AGE vs GLUCOSE GRAPH
st.subheader("Age vs Glucose")

fig = plt.figure()
sns.scatterplot(x="Age", y="Glucose", data=df, hue="Outcome")
sns.scatterplot(x=user_data["Age"], y=user_data["Glucose"],
                s=200, color=color)
plt.title("0 = Healthy | 1 = Diabetic")
st.pyplot(fig)

# AGE vs BMI GRAPH
st.subheader("Age vs BMI")

fig2 = plt.figure()
sns.scatterplot(x="Age", y="BMI", data=df, hue="Outcome")
sns.scatterplot(x=user_data["Age"], y=user_data["BMI"],
                s=200, color=color)
plt.title("0 = Healthy | 1 = Diabetic")
st.pyplot(fig2)

# OUTPUT RESULT
st.subheader("Prediction Result")

if user_result[0] == 0:
    st.success("You are NOT Diabetic")
else:
    st.error("You are Diabetic")

# MODEL ACCURACY
accuracy = accuracy_score(y_test, rf.predict(x_test)) * 100
st.subheader("Model Accuracy")
st.write(f"{accuracy:.2f}%")
