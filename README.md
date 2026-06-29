# mlops-loan-default

TCS Capstone Use Case 1A — Loan Default Prediction Pipeline with Automated Retraining

## Project Overview
An end-to-end MLOps pipeline that predicts loan default risk using XGBoost, with automated monitoring and retraining capabilities.

## Tech Stack
- PostgreSQL — Data storage
- Apache Airflow — Pipeline orchestration
- XGBoost — ML model
- Great Expectations — Data validation
- MLflow — Experiment tracking and model registry
- FastAPI — Model serving API
- Docker — Containerization
- GitHub Actions — CI/CD
- Evidently AI — Drift monitoring

## Setup Instructions

### Prerequisites
- Docker Desktop
- Python 3.11
- Git

### Steps
1. Clone the repo
   git clone https://github.com/suryamadhab-m/mlops-loan-default.git
   cd mlops-loan-default

2. Copy env file
   cp .env.example .env

3. Start all services
   docker-compose up -d

4. Load dataset
   python data/ingest.py

5. Train model
   python training/train.py

6. Log to MLflow
   python mlflow/log_model.py

7. Access services
   - FastAPI: http://localhost:8000/docs
   - MLflow: http://localhost:5000
   - Airflow: http://localhost:8080

## Pipeline Flow
Data Ingestion → Validation → Training → MLflow Logging → Model Promotion → Serving → Monitoring

## Team
- Member 1 — FastAPI, Docker, GitHub Actions
- Member 2 — PostgreSQL, Data Validation, Monitoring
- Member 3 — Airflow DAG
- Member 4 — XGBoost Training, MLflow
- Member 5 — MLflow Model Registry
- Member 6 — Evidently AI Monitoring