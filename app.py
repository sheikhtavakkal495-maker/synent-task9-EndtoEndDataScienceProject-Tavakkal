#Task 9: End-to-End Data Science Project
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# DATA COLLECTION
df = pd.read_csv(r"C:\Users\DELL\Desktop\StudentsPerformance.csv")
st.title("Student Performance Prediction System")

# DATA CLEANING
st.header("Dataset Information")
st.write("First 5 Rows")
st.write(df.head())
st.write("Missing Values")
st.write(df.isnull().sum())
df = df.dropna()

# EDA
st.header("Exploratory Data Analysis")
correlation = df[
    ["math score", "reading score", "writing score"]
].corr()
st.write("Correlation Matrix")
st.write(correlation)
fig, ax = plt.subplots()
ax.scatter(
    df["math score"],
    df["writing score"]
)
ax.set_xlabel("Math Score")
ax.set_ylabel("Writing Score")
ax.set_title("Math Score vs Writing Score")
st.pyplot(fig)

# FEATURE SELECTION
X = df[["math score", "reading score"]]
y = df["writing score"]

# TRAIN TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# MODEL TRAINING
model = LinearRegression()
model.fit(X_train, y_train)

# MODEL EVALUATION
y_pred = model.predict(X_test)
rmse = mean_squared_error(
    y_test,
    y_pred
) ** 0.5
st.header("Model Evaluation")
st.write("RMSE:", rmse)

# PREDICTION SECTION
st.header("Predict Writing Score")
math_score = st.number_input(
    "Enter Math Score",
    min_value=0,
    max_value=100,
    value=50
)
reading_score = st.number_input(
    "Enter Reading Score",
    min_value=0,
    max_value=100,
    value=50
)
if st.button("Predict"):
    sample = pd.DataFrame({
        "math score": [math_score],
        "reading score": [reading_score]
})
    prediction = model.predict(sample)
    st.success(
        f"Predicted Writing Score: {prediction[0]:.2f}"
)