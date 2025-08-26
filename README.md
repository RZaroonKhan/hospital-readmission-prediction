# Hospital Readmission Predictions

## Overview
This project builds a model that predicts whether a patient is at high risk to be readmitted to the hospital within 30 days using electronic health records. The workflow covers **data cleaning, exploratory data analysis (EDA), feature engineering, model development, hyperparameter tuning, calibration, and deployment** via a Streamlit web app.  

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
- Deployment-ready structure (Makefile included).
- Modular & reproducible code.

## Usage
### Creating the app
Run the Streamlit app using

make app

or

streamlit run app/app.py

### Using the app
1. Upload a CSV file of patient data with the required columns (shown in app)
2. Adjust the threshold slider to balance between recall (catching more readmissions) vs precision (avoiding false positives)
3. The app outputs the probability of readmission for each patient and binary prediction of if they will be readmitted according to the threshold

## Model Artifacts
The notebook 03_tuning_explainability.ipynb can be run to train the models needed however this may take time especially on less powerful devices so I am uploading the model artifacts onto a cloud drive so they can be downloaded. If downloading the artifacts instead of running the model create a folder called artifacts in the notebooks folder and place them in there 



