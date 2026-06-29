from fastapi.testclient import TestClient
import sys
import os
sys.path.insert(0, 'api')

# Mock model loading
import unittest.mock as mock
with mock.patch('mlflow.pyfunc.load_model'):
    from main import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get('/health')
    assert response.status_code == 200
    assert 'status' in response.json()

def test_predict_invalid_input():
    response = client.post('/predict', json={'invalid': 'data'})
    assert response.status_code == 422

def test_predict_valid_input_structure():
    payload = {
        'name_contract_type': 'Cash loans',
        'code_gender': 'M',
        'flag_own_car': 'N',
        'flag_own_realty': 'Y',
        'cnt_children': 1,
        'amt_income_total': 150000,
        'amt_credit': 500000,
        'amt_annuity': 25000,
        'days_birth': -12000,
        'days_employed': -2000
    }
    response = client.post('/predict', json=payload)
    assert response.status_code in [200, 503]