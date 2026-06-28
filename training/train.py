import pandas as pd
import xgboost as xgb
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, classification_report
from sqlalchemy import create_engine
import os

def train_model():
    engine = create_engine(
        'postgresql://admin:password@127.0.0.1:5432/loan_default'
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
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        scale_pos_weight=scale,
        random_state=42,
        use_label_encoder=False,
        eval_metric='auc',
    )
    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    print(f'ROC-AUC: {auc:.4f}')
    print(classification_report(y_test, model.predict(X_test)))

    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/xgboost_model.pkl')
    print('Model saved to models/xgboost_model.pkl')

    return auc

if __name__ == '__main__':
    train_model()