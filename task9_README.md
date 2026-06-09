# synent-task9-endtoendproject-yourname

## Problem Statement

Many people apply for loans and banks need a reliable way to decide who is likely to repay. The goal of this project was to go through the complete data science process from raw data to a deployed web app that can predict whether a loan application will be approved or not.

## Dataset Details

- Dataset: Loan Prediction Dataset
- Source: Kaggle
- Contains applicant details like gender, marital status, income, loan amount, credit history, and property area. The target column is loan status which is either approved or not.
- Format: CSV

## Approach

Starting with data cleaning I handled missing values in loan amount, credit history, and gender columns. Categorical columns were label encoded. EDA was done to understand how credit history and income relate to approval rates. After that I split the data 80/20 and trained a Logistic Regression model and a Random Forest classifier. The best model was saved using joblib. A simple Streamlit app was built where a user can fill in their details and get a prediction on the spot.

## Results

| Model | Accuracy | F1 Score |
|---|---|---|
| Logistic Regression | 80.5% | 0.79 |
| Random Forest | 84.2% | 0.83 |

- Random Forest was selected as the final model
- Credit history was by far the strongest predictor of loan approval
- The Streamlit app works correctly and returns predictions based on user input
