 # UPI Fraud Detection System

## Overview

UPI Fraud Detection System is a Machine Learning-based application designed to identify fraudulent digital payment transactions in real time. The system uses an XGBoost classifier along with feature engineering and risk scoring techniques to detect suspicious transaction patterns and enhance payment security.

## Features

* Real-time fraud prediction
* XGBoost-based machine learning model
* Fraud probability estimation
* Dynamic risk scoring
* High-value transaction detection
* Night-time transaction monitoring
* FastAPI REST API
* Model performance comparison

## Tech Stack

* Python
* FastAPI
* XGBoost
* Scikit-learn
* Pandas
* NumPy
* Joblib

## Machine Learning Models

| Model               | Accuracy |
| ------------------- | -------- |
| Logistic Regression | 96.34%   |
| Random Forest       | 96.07%   |
| XGBoost             | 97.94%   |

XGBoost achieved the best performance and was selected for deployment.

## API Endpoints

### GET /

Returns API status.

### GET /health

Checks API health.

### GET /stats

Displays model performance metrics.

### POST /predict

Predicts fraud probability and generates a risk score for a transaction.

## Project Workflow

1. Data Collection
2. Data Preprocessing
3. Feature Engineering
4. Model Training
5. Model Evaluation
6. Risk Scoring
7. API Deployment with FastAPI

## Installation

```bash
git clone https://github.com/your-username/UPI-Fraud-Detection.git
cd UPI-Fraud-Detection
pip install -r requirements.txt
uvicorn main:app --reload
```

## Future Enhancements

* Power BI Dashboard
* Explainable AI (SHAP)
* Real-Time Streaming Detection
* Geo-location Risk Analysis
* Cloud Deployment

## Author

Harshit Choudhary
