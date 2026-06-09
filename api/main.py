import json, joblib
from pathlib import Path
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

# Carrega modelo e lista de features ao iniciar
model    = joblib.load("models/model.joblib")
features = json.loads(Path("models/features.json").read_text())["features"]

app = FastAPI(title="Churn Prediction API")

# Pydantic valida os dados automaticamente antes de chegar no modelo
# Se faltar um campo ou o tipo estiver errado, retorna erro 422
class CustomerInput(BaseModel):
    tenure:          int   = Field(..., ge=0,  le=100)   # ge=mínimo, le=máximo
    monthly_charges: float = Field(..., ge=0)
    total_charges:   float = Field(..., ge=0)
    num_products:    int   = Field(..., ge=1,  le=10)
    support_calls:   int   = Field(..., ge=0)
    has_contract:    int   = Field(..., ge=0,  le=1)
    is_senior:       int   = Field(..., ge=0,  le=1)
    has_partner:     int   = Field(..., ge=0,  le=1)

# GET /health — sempre o primeiro endpoint a implementar
@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}

# POST /predict — recebe dados, retorna predição
@app.post("/predict")
def predict(customer: CustomerInput):
    # Transforma Pydantic → DataFrame com as features na ordem certa
    df   = pd.DataFrame([customer.model_dump()])[features]
    prob = float(model.predict_proba(df)[0][1])

    # Classifica o risco com base na probabilidade
    risk = "ALTO" if prob >= 0.7 else "MÉDIO" if prob >= 0.4 else "BAIXO"

    return {
        "churn_probability": round(prob, 4),
        "churn_prediction":  prob >= 0.5,
        "risk_level":        risk,
    }
