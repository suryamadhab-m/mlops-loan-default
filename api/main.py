from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager

model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    print("App starting. Model will be loaded after MLflow is ready.")
    yield

app = FastAPI(
    title="Loan Default Prediction API",
    description="Predicts probability of loan default",
    version="1.0.0",
    lifespan=lifespan
)

class LoanInput(BaseModel):
    amt_income_total: float = Field(..., gt=0)
    amt_credit: float = Field(..., gt=0)
    amt_annuity: float = Field(..., gt=0)
    days_birth: int
    days_employed: int
    cnt_children: int = Field(..., ge=0)

class PredictionOutput(BaseModel):
    prediction: int
    default_probability: float
    risk_level: str

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict", response_model=PredictionOutput)
def predict(data: LoanInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    try:
        import pandas as pd
        input_df = pd.DataFrame([data.dict()])
        prob = float(model.predict(input_df)[0])
        prediction = 1 if prob >= 0.5 else 0
        risk = "High" if prob >= 0.7 else "Medium" if prob >= 0.4 else "Low"
        return PredictionOutput(
            prediction=prediction,
            default_probability=round(prob, 4),
            risk_level=risk
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))