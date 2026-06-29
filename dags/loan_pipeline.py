from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import sys
import os

sys.path.insert(0, '/opt/airflow')

default_args = {
    'owner': 'mlops-team',
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'email_on_failure': False,
}

def ingest_data():
    import pandas as pd
    from sqlalchemy import create_engine
    engine = create_engine(
        'postgresql://admin:password@postgres:5432/loan_default'
    )
    df = pd.read_csv('/opt/airflow/data/raw/application_train.csv')
    df.columns = df.columns.str.lower()
    cols = ['sk_id_curr','target','name_contract_type','code_gender',
            'flag_own_car','flag_own_realty','cnt_children',
            'amt_income_total','amt_credit','amt_annuity',
            'days_birth','days_employed']
    df = df[cols]
    df.to_sql('loan_applications', engine, if_exists='replace', index=False, chunksize=1000)
    print(f'Ingested {len(df)} rows')

def validate_data():
    import pandas as pd
    from sqlalchemy import create_engine
    engine = create_engine(
        'postgresql://admin:password@postgres:5432/loan_default'
    )
    df = pd.read_sql('SELECT * FROM loan_applications', engine)

    assert 'target' in df.columns, "Column 'target' missing"
    assert df['target'].notnull().all(), "Null values in target"
    assert df['target'].isin([0, 1]).all(), "Invalid values in target"
    assert (df['amt_income_total'] >= 0).all(), "Negative income values"
    assert len(df) >= 100, "Not enough rows"

    print(f'All validations passed. Rows: {len(df)}')

def train_model():
    import pandas as pd
    import xgboost as xgb
    import joblib
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import roc_auc_score
    from sqlalchemy import create_engine
    engine = create_engine(
        'postgresql://admin:password@postgres:5432/loan_default'
    )
    df = pd.read_sql('SELECT * FROM loan_applications', engine)
    df = df.dropna(subset=['target'])
    categorical_cols = df.select_dtypes(include='object').columns
    le = LabelEncoder()
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col].astype(str))
    X = df.drop(['target', 'sk_id_curr'], axis=1).fillna(0)
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scale = (y_train == 0).sum() / (y_train == 1).sum()
    model = xgb.XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.05,
        scale_pos_weight=scale, random_state=42, use_label_encoder=False,
        eval_metric='auc',
    )
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    print(f'ROC-AUC: {auc:.4f}')
    os.makedirs('/opt/airflow/models', exist_ok=True)
    joblib.dump(model, '/opt/airflow/models/xgboost_model.pkl')
    print('Model saved')

def log_to_mlflow():
    import mlflow
    import mlflow.xgboost
    import joblib
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import roc_auc_score, accuracy_score
    from sqlalchemy import create_engine
    mlflow.set_tracking_uri('http://mlflow:5000')
    mlflow.set_experiment('loan_default_experiment')
    engine = create_engine(
        'postgresql://admin:password@postgres:5432/loan_default'
    )
    df = pd.read_sql('SELECT * FROM loan_applications', engine)
    df = df.dropna(subset=['target'])
    categorical_cols = df.select_dtypes(include='object').columns
    le = LabelEncoder()
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col].astype(str))
    X = df.drop(['target', 'sk_id_curr'], axis=1).fillna(0)
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = joblib.load('/opt/airflow/models/xgboost_model.pkl')
    y_prob = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    acc = accuracy_score(y_test, model.predict(X_test))
    params = {
        'n_estimators': 200, 'max_depth': 6, 'learning_rate': 0.05,
    }
    with mlflow.start_run(run_name='airflow_retrain') as run:
        mlflow.log_params(params)
        mlflow.log_metric('roc_auc', auc)
        mlflow.log_metric('accuracy', acc)
        mlflow.xgboost.log_model(
            model, artifact_path='model',
            registered_model_name='loan_default_model'
        )
        print(f'Logged to MLflow. AUC: {auc:.4f}')

def promote_model():
    from mlflow.tracking import MlflowClient
    client = MlflowClient(tracking_uri='http://mlflow:5000')
    model_name = 'loan_default_model'
    versions = client.get_latest_versions(model_name)
    if versions:
        latest = versions[-1].version
        client.transition_model_version_stage(model_name, latest, 'Production')
        print(f'Model v{latest} promoted to Production')

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