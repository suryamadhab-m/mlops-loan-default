# MLflow Experiment Tracking — Member 5
import mlflow
import mlflow.xgboost
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, accuracy_score
from sqlalchemy import create_engine
from mlflow.tracking import MlflowClient
import os
from dotenv import load_dotenv

load_dotenv()

def train_and_log():
    mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000'))
    mlflow.set_experiment('loan_default_experiment')

    engine = create_engine(
        f"postgresql://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}"
        f"@{os.getenv('POSTGRES_HOST')}:{os.getenv('POSTGRES_PORT')}/{os.getenv('POSTGRES_DB')}"
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

    params = {
        'n_estimators': 200,
        'max_depth': 6,
        'learning_rate': 0.05,
        'scale_pos_weight': (y_train==0).sum() / (y_train==1).sum(),
    }

    with mlflow.start_run(run_name='xgboost_day2') as run:
        mlflow.log_params(params)

        model = xgb.XGBClassifier(**params, random_state=42, use_label_encoder=False)
        model.fit(X_train, y_train)

        y_prob = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
        acc = accuracy_score(y_test, model.predict(X_test))

        mlflow.log_metric('roc_auc', auc)
        mlflow.log_metric('accuracy', acc)

        mlflow.xgboost.log_model(
            model,
            artifact_path='model',
            registered_model_name='loan_default_model'
        )
        print(f'Run complete. AUC: {auc:.4f}')
        return run.info.run_id

if __name__ == '__main__':
    train_and_log()