import pickle
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Breast Cancer Prediction API",
    description="Prédit si une tumeur est maligne ou bénigne à partir de 30 features.",
    version="1.0.0",
)

# ── Chargement du modèle au démarrage ─────────────────────────────────────────
MODEL_PATH = "model/model.pkl"

try:
    with open(MODEL_PATH, "rb") as f:
        artifact = pickle.load(f)
    model = artifact["model"]
    scaler = artifact["scaler"]
    feature_names = artifact["feature_names"]
    target_names = artifact["target_names"]
    print(f"Modèle chargé depuis {MODEL_PATH}")
except FileNotFoundError:
    raise RuntimeError(
        f"Modèle introuvable : {MODEL_PATH}. Lance d'abord train.py."
    )

# ── Schémas ───────────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    features: List[float] = Field(
        ...,
        min_length=30,
        max_length=30,
        description="Liste de 30 valeurs numériques (les 30 features du dataset Breast Cancer).",
        example=[
            17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471,
            0.2419, 0.07871, 1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904,
            0.05373, 0.01587, 0.03003, 0.006193, 25.38, 17.33, 184.6, 2019.0,
            0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189
        ],
    )

class PredictResponse(BaseModel):
    prediction: int
    label: str
    probability_malignant: float
    probability_benign: float

# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health", summary="Vérification de l'état du service")
def health():
    return {
        "status": "ok",
        "model": "LogisticRegression",
        "features_expected": len(feature_names),
    }


@app.post("/predict", response_model=PredictResponse, summary="Prédiction de tumeur")
def predict(request: PredictRequest):
    X = np.array(request.features).reshape(1, -1)

    if X.shape[1] != len(feature_names):
        raise HTTPException(
            status_code=422,
            detail=f"Attendu {len(feature_names)} features, reçu {X.shape[1]}.",
        )

    X_scaled = scaler.transform(X)
    prediction = int(model.predict(X_scaled)[0])
    probabilities = model.predict_proba(X_scaled)[0]

    return PredictResponse(
        prediction=prediction,
        label=target_names[prediction],
        probability_malignant=round(float(probabilities[0]), 4),
        probability_benign=round(float(probabilities[1]), 4),
    )