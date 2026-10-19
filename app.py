"""
FastAPI Production Computer Vision Service for Disease Detection.
Supports:
- GET  /               : Interactive Image Upload & Diagnostic Dashboard
- GET  /health         : Container & cloud load-balancer health check
- GET  /metrics        : Returns DVC scores.json & training history
- POST /predict        : Accepts Base64-encoded image JSON (Krish Naik decodeImage pattern)
- POST /predict-upload : Accepts Multipart file upload (`UploadFile`)
- POST /train          : Runs full 4-stage DVC training pipeline (`dvc repro` / `main.py`)
"""
import os
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
import uvicorn

from src.cnn_classifier.logger import logger
from src.cnn_classifier.utils.common import decodeImage, load_json
from src.cnn_classifier.pipeline.prediction import PredictionPipeline
from main import run_all_stages

app = FastAPI(
    title="Production MLOps Project 02: Deep Learning CNN Disease Detection API",
    description=(
        "ResNet-18 Transfer Learning Computer Vision system with DVC pipeline tracking, "
        "DagsHub MLflow logging, INT8 dynamic quantization, and AWS + Azure CI/CD."
    ),
    version="1.0.0",
)

templates = Jinja2Templates(directory="templates")
TEMP_IMAGE_PATH = Path("artifacts/prediction_inputImage.png")


class Base64ImageRequest(BaseModel):
    """Base64 image payload matching Krish Naik's ClientApp decodeImage pattern."""

    image: str = Field(..., description="Base64-encoded RGB microscopy image string")


@app.get("/health", tags=["Operations"])
async def health_check():
    model_ready = Path("artifacts/training/model.pt").exists()
    quant_ready = Path("artifacts/training/model_quantized.pt").exists()
    return {
        "status": "healthy" if model_ready else "degraded",
        "fp32_model_ready": model_ready,
        "int8_quantized_model_ready": quant_ready,
        "version": "1.0.0",
    }


@app.get("/metrics", tags=["Operations"])
async def get_metrics():
    score_path = Path("scores.json")
    if not score_path.exists():
        raise HTTPException(status_code=404, detail="scores.json not found. Run /train first.")
    res = {"evaluation_scores": load_json(score_path)}
    hist_path = Path("artifacts/training/training_history.json")
    if hist_path.exists():
        res["training_history"] = load_json(hist_path)
    return res


@app.post("/predict", tags=["Inference"])
async def predict_base64(payload: Base64ImageRequest):
    """Decodes base64 image string and runs ResNet-18 disease classification."""
    try:
        os.makedirs("artifacts", exist_ok=True)
        decodeImage(payload.image, TEMP_IMAGE_PATH)
        classifier = PredictionPipeline(TEMP_IMAGE_PATH)
        return classifier.predict()
    except Exception as e:
        logger.error(f"Base64 prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict-upload", tags=["Inference"])
async def predict_upload(file: UploadFile = File(...)):
    """Accepts multipart image file upload and returns classification + probabilities."""
    try:
        os.makedirs("artifacts", exist_ok=True)
        content = await file.read()
        with open(TEMP_IMAGE_PATH, "wb") as f:
            f.write(content)
        classifier = PredictionPipeline(TEMP_IMAGE_PATH)
        return classifier.predict()
    except Exception as e:
        logger.error(f"Upload prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/train", tags=["Pipeline"])
async def trigger_training():
    """Executes all 4 DVC pipeline stages."""
    try:
        scores = run_all_stages()
        return {"status": "Training & Evaluation Completed", "scores": scores}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/", response_class=HTMLResponse, tags=["UI"])
async def home(request: Request):
    score_path = Path("scores.json")
    scores = load_json(score_path) if score_path.exists() else None
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"scores": scores},
    )


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8080, reload=False)
