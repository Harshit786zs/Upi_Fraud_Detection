<div align="center">

# 🛡️ UPI Fraud Detection System

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB)
![XGBoost](https://img.shields.io/badge/XGBoost-97.94%25_ROC--AUC-FF6600?style=for-the-badge)

**Real-time ML-powered UPI fraud detection with explainable AI and role-based dashboard**

[Dashboard Repo](https://github.com/Harshit786zs/fraud-dashboard) • [API Docs](http://localhost:8000/docs)

</div>

---

## Overview

UPI fraud in India grew **85% year-over-year** in FY 2024-25. This system detects fraudulent transactions in real-time using a hybrid ML ensemble, explains every decision using SHAP, and visualizes everything on a live React dashboard with role-based access.

---

## Architecture
Dataset → Cleaning → Feature Engineering → XGBoost Model
→ SHAP Explainability → Risk Scoring (0–100)
→ FastAPI Backend → React Dashboard

---

## Model Performance

| Model | ROC-AUC |
|---|---|
| Logistic Regression | 96.34% |
| Random Forest | 96.07% |
| **XGBoost** ✅ | **97.94%** |

> 283,726 transactions • 473 fraud cases • SMOTE applied

---

## Features

- ⚡ **Live transaction feed** — scored by XGBoost every 1.8s
- 🚨 **Real-time alerts** — popup when fraud is blocked
- 🧪 **Transaction tester** — test any amount instantly
- 🔐 **Role-based login** — Admin and Analyst roles
- 📈 **Analytics** — live fraud trend charts
- 🤖 **SHAP explainability** — feature importance plots

---

## Tech Stack

**ML:** XGBoost, Scikit-Learn, SHAP, SMOTE  
**Backend:** FastAPI, Uvicorn, Pydantic  
**Frontend:** React, Recharts, Axios

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/Harshit786zs/Upi_Fraud_Detection.git
cd Upi_Fraud_Detection

# 2. Setup
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# 3. Add dataset → data/raw/creditcard.csv
# Download from: kaggle.com/datasets/mlg-ulb/creditcardfraud

# 4. Run pipeline
python src/cleaning/cleaner.py
python src/features/engineer.py
python src/models/train.py

# 5. Start API
uvicorn src.api.main:app --reload --port 8000

# 6. Start Dashboard
cd ../fraud-dashboard && npm install && npm start
```

---

## Login Credentials

| Role | Username | Password |
|---|---|---|
| Admin | `admin` | `admin123` |
| Analyst | `analyst` | `analyst123` |

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | API status |
| GET | `/health` | Health check |
| GET | `/stats` | Model metrics |
| POST | `/predict` | Fraud prediction |

---

## Project Structure
src/
├── cleaning/        # Data preprocessing
├── features/        # Feature engineering
├── models/          # ML training & evaluation
├── explainability/  # SHAP analysis
├── risk_scoring/    # 0–100 risk engine
└── api/             # FastAPI backend
---

<div align="center">

**Harshit Choudhary** — Final Year B.Tech CSE  
[![GitHub](https://img.shields.io/badge/GitHub-Harshit786zs-181717?style=flat-square&logo=github)](https://github.com/Harshit786zs)

</div>