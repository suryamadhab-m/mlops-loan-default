import pandas as pd
from sqlalchemy import create_engine
import os
from dotenv import load_dotenv

load_dotenv()

def ingest_data():
    engine = create_engine(
        f"postgresql://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}"
        f"@{os.getenv('POSTGRES_HOST')}:{os.getenv('POSTGRES_PORT')}/{os.getenv('POSTGRES_DB')}"
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