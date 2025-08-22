# Hospital Readmission Predictions


This project builds a predictive model to identify patients at high risk of being readmitted within 30 days of discharge, using real-world hospital data. The solution includes exploratory data analysis, feature engineering, and model training (Logistic Regression, Random Forest, XGBoost), with SHAP-based model explainability. Delivered as an interactive Streamlit app for clinicians, plus a Power BI dashboard for hospital leadership.

Tech Stack: Python (Pandas, scikit-learn, SHAP), Streamlit, Power BI


# Hospital Readmission Prediction

## Overview
This project predicts whether a patient will be readmitted to the hospital within 30 days using electronic health records.  
The workflow covers **data cleaning, exploratory data analysis (EDA), feature engineering, model development, hyperparameter tuning, calibration, and deployment** via a Streamlit web app.  

It is designed as an **end-to-end data science & machine learning pipeline** — from raw data → trained ML models → interactive web app.  


## Features
- Data preprocessing & cleaning (handling missing values, encoding categorical variables, scaling numerics).
- Exploratory Data Analysis (EDA) with visualizations.
- Machine Learning Models:
  - Logistic Regression
  - Random Forest
- Hyperparameter tuning with cross-validation.
- Probability calibration for better decision thresholds.
- **Streamlit Web App** for user-friendly predictions.
- Deployment-ready structure (Docker + Makefile included).
- Modular & reproducible code.
