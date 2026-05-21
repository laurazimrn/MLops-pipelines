from fastapi import FastAPI
from pydantic import BaseModel
import os
import joblib
from typing import List

app = FastAPI()

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'model.joblib')

class PredictRequest(BaseModel):
    features: List[float]


@app.get('/')
def health():
    return {'status': 'ok'}


@app.post('/predict')
def predict(req: PredictRequest):
    if not os.path.exists(MODEL_PATH):
        return {'error': 'model not found', 'model_path': MODEL_PATH}
    model = joblib.load(MODEL_PATH)
    pred = model.predict([req.features]).tolist()
    return {'prediction': pred}
