"""
FastAPI Production Computer Vision & Clinical Triage Portal for Project 02 (v2.0).
Exposes 6 Dynamic Web Pages + Full REST API:
- GET  /                  : Page 2 — Interactive Grad-CAM, TTA Uncertainty & Clinical Triage Studio
- GET  /intro             : Page 1 — "What This Project Is & How I Developed It" Engineering Walkthrough
- GET  /batch-triage      : Page 3 — Multi-Flock / Multi-Sample Batch Triage Queue & Priority Sorter
- GET  /from-scratch-lab  : Page 4 — First-Principles 2D Convolution Kernel & Micrograd Autograd Lab
- GET  /dvc-metrics       : Page 5 — DVC 4-Stage Pipeline DAG, Epoch Curves & INT8 Quantization Benchmark
- GET  /health            : Page 6 — Health & Telemetry Audit Trail (HTML in browser, JSON for API/TestClient)
- GET  /metrics           : DVC scores.json & training history (HTML in browser, JSON for API/TestClient)
- POST /predict           : Accepts Base64-encoded image JSON (Krish Naik decodeImage pattern) + Grad-CAM & Uncertainty
- POST /predict-upload    : Accepts Multipart file upload (`UploadFile`)
- POST /train             : Runs full 4-stage DVC training pipeline (`dvc repro` / `main.py`)
"""
import os
from pathlib import Path
from typing import Any, Dict, Optional
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
import uvicorn

from src.cnn_classifier.logger import logger
from src.cnn_classifier.utils.common import decodeImage, load_json
from src.cnn_classifier.pipeline.prediction import PredictionPipeline
from src.cnn_classifier.components.clinical_cv_suite import ClinicalCVDiagnosticSuite, KERNELS_2D
from main import run_all_stages

app = FastAPI(
    title="Production MLOps Project 02: Deep Learning CNN Disease Detection & Clinical Triage Suite",
    description=(
        "ResNet-18 Transfer Learning Computer Vision system with Grad-CAM Localization, "
        "Test-Time Augmentation (TTA) Epistemic Uncertainty, Morphological Biomarkers, "
        "First-Principles 2D Convolution & Micrograd Lab, DVC Pipeline Tracking, and Dynamic INT8 Quantization."
    ),
    version="2.0.0",
)

templates = Jinja2Templates(directory="templates")
TEMP_IMAGE_PATH = Path("artifacts/prediction_inputImage.png")
cv_suite = ClinicalCVDiagnosticSuite()


class Base64ImageRequest(BaseModel):
    """Base64 image payload matching Krish Naik's ClientApp decodeImage pattern."""

    image: str = Field(..., description="Base64-encoded RGB microscopy image string")


def _wants_html(request: Request, format_param: Optional[str] = None) -> bool:
    if format_param == "json":
        return False
    if format_param == "html":
        return True
    accept = request.headers.get("accept", "").lower()
    return "text/html" in accept


def _build_ui_context(active_page: str = "studio") -> Dict[str, Any]:
    score_path = Path("scores.json")
    scores = load_json(score_path) if score_path.exists() else {}
    hist_path = Path("artifacts/training/training_history.json")
    history = load_json(hist_path) if hist_path.exists() else {}

    model_ready = Path("artifacts/training/model.pt").exists()
    quant_ready = Path("artifacts/training/model_quantized.pt").exists()

    return {
        "active_page": active_page,
        "scores": scores,
        "history": history,
        "health": {
            "status": "healthy" if model_ready else "degraded",
            "fp32_model_ready": model_ready,
            "int8_quantized_model_ready": quant_ready,
            "version": "2.0.0",
        },
        "available_samples": list(cv_suite.sample_paths.keys()),
        "available_kernels": list(KERNELS_2D.keys()),
        "diag_result": None,
        "batch_results": None,
        "kernel_lab": None,
        "micrograd_lab": None,
        "audit_log": cv_suite.get_audit_log(limit=12),
    }


# =====================================================================
# Multi-Page HTML Routes + Content-Negotiated /health & /metrics
# =====================================================================
@app.get("/", response_class=HTMLResponse, tags=["UI Pages"])
async def home(request: Request, sample: str = "Coccidiosis", threshold: float = 70.0):
    ctx = _build_ui_context(active_page="studio")
    sample_path = cv_suite.sample_paths.get(sample) or next(iter(cv_suite.sample_paths.values()), None)
    if sample_path and sample_path.exists():
        ctx["diag_result"] = cv_suite.run_full_diagnostic(
            sample_path, sensitivity_threshold=float(threshold)
        )
        ctx["audit_log"] = cv_suite.get_audit_log(limit=12)
    return templates.TemplateResponse(request=request, name="index.html", context=ctx)


@app.get("/intro", response_class=HTMLResponse, tags=["UI Pages"])
async def intro_page(request: Request):
    ctx = _build_ui_context(active_page="intro")
    return templates.TemplateResponse(request=request, name="index.html", context=ctx)


@app.get("/studio", response_class=HTMLResponse, tags=["UI Pages"])
async def studio_page(request: Request, sample: str = "Coccidiosis"):
    return await home(request, sample=sample)


@app.post("/studio-upload", response_class=HTMLResponse, tags=["UI Pages"])
async def studio_upload_form(
    request: Request,
    sample_select: str = Form("Coccidiosis"),
    sensitivity_threshold: float = Form(70.0),
    file: Optional[UploadFile] = File(default=None),
):
    ctx = _build_ui_context(active_page="studio")
    os.makedirs("artifacts", exist_ok=True)
    target_path = None

    if file and file.filename and file.filename.strip() != "":
        raw = await file.read()
        if len(raw) > 0:
            with open(TEMP_IMAGE_PATH, "wb") as f:
                f.write(raw)
            target_path = TEMP_IMAGE_PATH

    if target_path is None:
        target_path = cv_suite.sample_paths.get(sample_select) or next(iter(cv_suite.sample_paths.values()))

    ctx["diag_result"] = cv_suite.run_full_diagnostic(
        target_path, sensitivity_threshold=float(sensitivity_threshold)
    )
    ctx["audit_log"] = cv_suite.get_audit_log(limit=12)
    return templates.TemplateResponse(request=request, name="index.html", context=ctx)


@app.get("/batch-triage", response_class=HTMLResponse, tags=["UI Pages"])
async def batch_triage_page(request: Request):
    ctx = _build_ui_context(active_page="batch")
    ctx["batch_results"] = cv_suite.run_batch_triage_cohort()
    ctx["audit_log"] = cv_suite.get_audit_log(limit=12)
    return templates.TemplateResponse(request=request, name="index.html", context=ctx)


@app.get("/from-scratch-lab", response_class=HTMLResponse, tags=["UI Pages"])
async def from_scratch_lab_page(
    request: Request, sample_class: str = "Coccidiosis", kernel_name: str = "sobel_edge_x"
):
    ctx = _build_ui_context(active_page="scratch")
    ctx["kernel_lab"] = cv_suite.run_2d_kernel_lab(sample_class=sample_class, kernel_name=kernel_name)
    ctx["micrograd_lab"] = cv_suite.run_micrograd_neuron_demo()
    return templates.TemplateResponse(request=request, name="index.html", context=ctx)


@app.post("/from-scratch-lab", response_class=HTMLResponse, tags=["UI Pages"])
async def from_scratch_lab_post(
    request: Request,
    sample_class: str = Form("Coccidiosis"),
    kernel_name: str = Form("sobel_edge_x"),
    x1: float = Form(2.0),
    x2: float = Form(-1.5),
    w1: float = Form(0.8),
    w2: float = Form(-1.2),
    b: float = Form(0.5),
):
    ctx = _build_ui_context(active_page="scratch")
    ctx["kernel_lab"] = cv_suite.run_2d_kernel_lab(sample_class=sample_class, kernel_name=kernel_name)
    ctx["micrograd_lab"] = cv_suite.run_micrograd_neuron_demo(x1=x1, x2=x2, w1=w1, w2=w2, b=b)
    return templates.TemplateResponse(request=request, name="index.html", context=ctx)


@app.get("/dvc-metrics", response_class=HTMLResponse, tags=["UI Pages"])
async def dvc_metrics_page(request: Request):
    ctx = _build_ui_context(active_page="dvc")
    return templates.TemplateResponse(request=request, name="index.html", context=ctx)


@app.get("/health", tags=["Operations"])
async def health_check(request: Request, format: Optional[str] = None):
    ctx = _build_ui_context(active_page="telemetry")
    if _wants_html(request, format):
        return templates.TemplateResponse(request=request, name="index.html", context=ctx)
    return JSONResponse(content=ctx["health"])


@app.get("/metrics", tags=["Operations"])
async def get_metrics(request: Request, format: Optional[str] = None):
    score_path = Path("scores.json")
    if not score_path.exists():
        raise HTTPException(status_code=404, detail="scores.json not found. Run /train first.")
    ctx = _build_ui_context(active_page="dvc")
    if _wants_html(request, format):
        return templates.TemplateResponse(request=request, name="index.html", context=ctx)
    return JSONResponse(
        content={
            "evaluation_scores": ctx["scores"],
            "training_history": ctx["history"],
        }
    )


# =====================================================================
# JSON REST Inference & Training Endpoints
# =====================================================================
@app.post("/predict", tags=["Inference"])
async def predict_base64(payload: Base64ImageRequest):
    """Decodes base64 image string and runs ResNet-18 disease classification + Grad-CAM + Uncertainty."""
    try:
        os.makedirs("artifacts", exist_ok=True)
        decodeImage(payload.image, TEMP_IMAGE_PATH)
        classifier = PredictionPipeline(TEMP_IMAGE_PATH)
        base_res = classifier.predict()
        full_diag = cv_suite.run_full_diagnostic(TEMP_IMAGE_PATH)
        return {
            **base_res,
            "triage_level": full_diag["triage_level"],
            "triage_badge": full_diag["triage_badge"],
            "uncertainty": full_diag["uncertainty"],
            "morphology": full_diag["morphology"],
            "ai_pathologist_narrative": full_diag["ai_pathologist_narrative"],
            "gradcam_overlay_b64": full_diag["gradcam_overlay_b64"],
        }
    except Exception as e:
        logger.error(f"Base64 prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict-upload", tags=["Inference"])
async def predict_upload(file: UploadFile = File(...)):
    """Accepts multipart image file upload and returns classification + Grad-CAM + TTA uncertainty."""
    try:
        os.makedirs("artifacts", exist_ok=True)
        content = await file.read()
        with open(TEMP_IMAGE_PATH, "wb") as f:
            f.write(content)
        return cv_suite.run_full_diagnostic(TEMP_IMAGE_PATH)
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


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8080, reload=False)
