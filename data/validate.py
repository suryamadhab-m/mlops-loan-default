import pandas as pd
import great_expectations as ge
from sqlalchemy import create_engine

def validate_data():
    engine = create_engine(
        'postgresql://admin:password@127.0.0.1:5432/loan_default'
    )
    df = pd.read_sql('SELECT * FROM loan_applications', engine)
    ge_df = ge.from_pandas(df)

    checks = [
        ge_df.expect_column_to_exist('target'),
        ge_df.expect_column_values_to_not_be_null('target'),
        ge_df.expect_column_values_to_be_in_set('target', [0, 1]),
        ge_df.expect_column_values_to_be_between('amt_income_total', min_value=0),
        ge_df.expect_table_row_count_to_be_between(min_value=100),
    ]

    failed = [c for c in checks if not c['success']]
    if failed:
        raise ValueError(f'{len(failed)} validation checks failed.')
    print('All validations passed.')
    return True

if __name__ == '__main__':
    validate_data()