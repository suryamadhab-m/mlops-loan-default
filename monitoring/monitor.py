import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset
from sqlalchemy import create_engine
import os
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

DRIFT_THRESHOLD = 0.3

def get_engine():
    return create_engine(
        f"postgresql://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}"
        f"@{os.getenv('POSTGRES_HOST')}:{os.getenv('POSTGRES_PORT')}/{os.getenv('POSTGRES_DB')}"
    )

def run_monitoring():
    engine = get_engine()
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
        print('ALERT: Drift detected. Retraining needed.')
    else:
        print('Drift within acceptable range.')

if __name__ == '__main__':
    run_monitoring()