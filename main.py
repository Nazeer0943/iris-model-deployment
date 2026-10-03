"""
FastAPI Model Serving Application
Author: Nazir Sani
Description:
    Serving an Iris classification pipeline with health probes,
    model metadata retrieval, and strict input validation.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Literal, Dict, Any, List
import joblib
import numpy as np

# Initializing the FastAPI application
app = FastAPI(
    title="Iris Flower Classification Service",
    description="A production-ready model deployment API serving predictions and metadata.",
    version="1.0.0"
)

# Loading the persisted model artifact and metadata bundle
artifact = joblib.load("iris_model_artifact.pkl")
model = artifact["model"]
metadata = artifact["metadata"]
LABELS = metadata["target_classes"]

# Defining the input schema with strict range constraints
class IrisFeatures(BaseModel):
    sepal_length: float = Field(..., gt=0.0, description="Sepal length in centimeters (must be > 0)")
    sepal_width: float = Field(..., gt=0.0, description="Sepal width in centimeters (must be > 0)")
    petal_length: float = Field(..., gt=0.0, description="Petal length in centimeters (must be > 0)")
    petal_width: float = Field(..., gt=0.0, description="Petal width in centimeters (must be > 0)")

# Defining prediction output schema
class PredictionResponse(BaseModel):
    species: str
    confidence: float
    probabilities: Dict[str, float]
    model_version: str

# Defining metadata response schema
class ModelInfoResponse(BaseModel):
    model_name: str
    version: str
    training_timestamp: str
    metrics: Dict[str, float]
    target_classes: List[str]
    features: List[str]

@app.get("/health", tags=["Monitoring"])
def health_check():
    """Liveness probe verifying that the API is running and the model is loaded."""
    return {"status": "healthy", "model_loaded": model is not None}

@app.get("/model-info", response_model=ModelInfoResponse, tags=["Metadata"])
def get_model_info():
    """Returning the model's training timestamp, evaluation metrics, and feature list."""
    return metadata

@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict_species(features: IrisFeatures):
    """Executing inference on incoming flower measurement vectors."""
    try:
        # Preparing feature array
        input_data = np.array([[
            features.sepal_length,
            features.sepal_width,
            features.petal_length,
            features.petal_width
        ]])

        # Generating class probabilities
        probs = model.predict_proba(input_data)[0]
        predicted_idx = int(np.argmax(probs))

        prob_dict = {label: float(prob) for label, prob in zip(LABELS, probs)}

        return {
            "species": LABELS[predicted_idx],
            "confidence": float(probs[predicted_idx]),
            "probabilities": prob_dict,
            "model_version": metadata["version"]
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(exc)}")
