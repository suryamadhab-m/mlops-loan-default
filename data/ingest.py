import pandas as pd
from sqlalchemy import create_engine

def ingest_data():
    engine = create_engine(
        'postgresql://admin:password@127.0.0.1:5432/loan_default'
    )
    df = pd.read_csv('data/raw/application_train.csv')
    df.columns = df.columns.str.lower()
    cols = ['sk_id_curr','target','name_contract_type','code_gender',
            'flag_own_car','flag_own_realty','cnt_children',
            'amt_income_total','amt_credit','amt_annuity',
            'days_birth','days_employed']
    df = df[cols]
    df.to_sql('loan_applications', engine, if_exists='replace', index=False, chunksize=1000)
    print(f'Ingested {len(df)} rows')

if __name__ == '__main__':
    ingest_data()