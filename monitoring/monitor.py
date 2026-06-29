import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset
from sqlalchemy import create_engine
import os
import requests
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

DRIFT_THRESHOLD = 0.3
AIRFLOW_URL = 'http://localhost:8080/api/v1/dags/loan_default_pipeline/dagRuns'
AIRFLOW_AUTH = ('admin', 'admin')

def get_engine():
    return create_engine(
        f"postgresql://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}"
        f"@{os.getenv('POSTGRES_HOST')}:{os.getenv('POSTGRES_PORT')}/{os.getenv('POSTGRES_DB')}"
    )

def trigger_retraining():
    print('Triggering Airflow retraining DAG...')
    payload = {'conf': {'triggered_by': 'evidently_drift_detection'}}
    response = requests.post(
        AIRFLOW_URL,
        json=payload,
        auth=AIRFLOW_AUTH,
        headers={'Content-Type': 'application/json'}
    )
    if response.status_code == 200:
        print('Retraining DAG triggered successfully')
    else:
        print(f'Failed to trigger DAG: {response.status_code} - {response.text}')

def simulate_drift(engine):
    print('Simulating drift by injecting skewed data...')
    from sqlalchemy import text
    with engine.connect() as conn:
        conn.execute(text(
            'UPDATE loan_applications SET amt_income_total = amt_income_total * 10'
            ' WHERE sk_id_curr IN (SELECT sk_id_curr FROM loan_applications LIMIT 5000)'
        ))
        conn.commit()
    print('Drift simulation complete')

def run_monitoring(simulate=False):
    engine = get_engine()

    if simulate:
        simulate_drift(engine)

    df = pd.read_sql('SELECT * FROM loan_applications', engine)

    split = int(len(df) * 0.7)
    reference = df.iloc[:split]
    current = df.iloc[split:]

    print(f'Reference: {len(reference)} rows | Current: {len(current)} rows')

    report = Report(metrics=[
        DataDriftPreset(),
        DataQualityPreset(),
    ])
    report.run(reference_data=reference, current_data=current)

    os.makedirs('reports', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    path = f'reports/drift_report_{timestamp}.html'
    report.save_html(path)
    print(f'Report saved: {path}')

    result = report.as_dict()
    drift_share = result['metrics'][0]['result']['drift_share']
    print(f'Drift share: {drift_share:.2%}')

    if drift_share >= DRIFT_THRESHOLD:
        print(f'ALERT: Drift {drift_share:.2%} exceeds threshold {DRIFT_THRESHOLD:.2%}')
        # trigger_retraining()  # Disabled due to Airflow REST API authentication issue
        print('Manual retraining required.')
    else:
        print('Drift within acceptable range. No retraining needed.')

if __name__ == '__main__':
    run_monitoring(simulate=False)