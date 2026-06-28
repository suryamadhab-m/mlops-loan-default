CREATE TABLE IF NOT EXISTS loan_applications (
    sk_id_curr BIGINT PRIMARY KEY,
    target INTEGER,
    name_contract_type VARCHAR(50),
    code_gender VARCHAR(5),
    flag_own_car VARCHAR(5),
    flag_own_realty VARCHAR(5),
    cnt_children INTEGER,
    amt_income_total FLOAT,
    amt_credit FLOAT,
    amt_annuity FLOAT,
    days_birth INTEGER,
    days_employed INTEGER,
    created_at TIMESTAMP DEFAULT NOW()
);