from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
import mlflow.pyfunc
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os
from dotenv import load_dotenv

load_dotenv()

model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    try:
        mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000'))
        model = mlflow.pyfunc.load_model('models:/loan_default_model/Production')
        print('Production model loaded from MLflow Registry')
    except Exception as e:
        print(f'Model not loaded: {e}')
    yield

app = FastAPI(
    title='Loan Default Prediction API',
    description='Predicts probability of loan default using XGBoost',
    version='1.0.0',
    lifespan=lifespan
)

class LoanInput(BaseModel):
    name_contract_type: str = "Cash loans"
    code_gender: str = "M"
    flag_own_car: str = "N"
    flag_own_realty: str = "Y"
    cnt_children: int = Field(..., ge=0)
    amt_income_total: float = Field(..., gt=0)
    amt_credit: float = Field(..., gt=0)
    amt_annuity: float = Field(..., gt=0)
    days_birth: int
    days_employed: int

class PredictionOutput(BaseModel):
    prediction: int
    default_probability: float
    risk_level: str

@app.get('/health')
def health():
    return {'status': 'healthy', 'model_loaded': model is not None}

@app.post('/predict', response_model=PredictionOutput)
def predict(data: LoanInput):
    if model is None:
        raise HTTPException(status_code=503, detail='Model not loaded yet')
    try:
        input_dict = data.dict()
        df = pd.DataFrame([input_dict])

        # Encode categorical columns
        categorical_cols = ['name_contract_type', 'code_gender', 'flag_own_car', 'flag_own_realty']
        le = LabelEncoder()
        for col in categorical_cols:
            df[col] = le.fit_transform(df[col].astype(str))

        prob = float(model.predict(df)[0])
        prediction = 1 if prob >= 0.5 else 0
        risk = 'High' if prob >= 0.7 else 'Medium' if prob >= 0.4 else 'Low'
        return PredictionOutput(
            prediction=prediction,
            default_probability=round(prob, 4),
            risk_level=risk
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))