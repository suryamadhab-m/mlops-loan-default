from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'mlops-team',
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'email_on_failure': False,
}

def ingest_data():
    print("Ingesting data into PostgreSQL")

def validate_data():
    print("Validating data with Great Expectations")

def train_model():
    print("Training XGBoost model")

def log_to_mlflow():
    print("Logging model to MLflow")

def promote_model():
    print("Promoting model to Production in MLflow Registry")

with DAG(
    dag_id='loan_default_pipeline',
    default_args=default_args,
    schedule_interval='@weekly',
    start_date=datetime(2024, 1, 1),
    catchup=False,
    description='Loan Default Prediction MLOps Pipeline',
) as dag:

    t1 = PythonOperator(task_id='ingest_data', python_callable=ingest_data)
    t2 = PythonOperator(task_id='validate_data', python_callable=validate_data)
    t3 = PythonOperator(task_id='train_model', python_callable=train_model)
    t4 = PythonOperator(task_id='log_to_mlflow', python_callable=log_to_mlflow)
    t5 = PythonOperator(task_id='promote_model', python_callable=promote_model)

    t1 >> t2 >> t3 >> t4 >> t5