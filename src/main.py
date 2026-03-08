from fastapi import FastAPI
import pickle
import numpy as np
from fastapi import FastAPI
import pickle

app = FastAPI()

with open("/Users/lindokuhletami/Desktop/Space/basel-credit-risk-model/src/pd_model.sav", "rb") as f:
    model = pickle.load(f)
   


@app.get("/")
def home():
    return {"message": "PD Model API running"}


@app.post("/predict")
def predict(features: list):
    X = np.array(features).reshape(1, -1)
    prediction = model.predict(X)
    probability = model.predict_proba(X)[0][1]

    return {
        "prediction": int(prediction[0]),
        "probability_default": float(probability)
    }