"""
Automated Unit & Integration Test Suite for Project 02: CNN Disease Classifier.
Tests:
1. First-principles 2D Convolution & Residual Block forward/backward pass
2. Transfer Learning layer freezing (`DiseaseClassifierTransferNet`)
3. Full 4-stage DVC pipeline execution & `scores.json` verification
4. Single-image `PredictionPipeline` & Base64 encoding/decoding
5. FastAPI endpoints (`/health`, `/metrics`, `/predict`, `/predict-upload`)
"""
import os
from pathlib import Path
import numpy as np
import torch
from fastapi.testclient import TestClient

from src.cnn_classifier.utils.common import encodeImageIntoBase64
from src.cnn_classifier.components.prepare_base_model import (
    conv2d_from_scratch,
    FirstPrinciplesResidualBlock,
    DiseaseClassifierTransferNet,
)
from src.cnn_classifier.pipeline.prediction import PredictionPipeline
from main import run_all_stages
from app import app

client = TestClient(app)


def test_first_principles_conv_and_residual_block():
    img = np.ones((10, 10), dtype=np.float32)
    kernel = np.ones((3, 3), dtype=np.float32)
    out = conv2d_from_scratch(img, kernel, stride=1, padding=1)
    assert out.shape == (10, 10)

    block = FirstPrinciplesResidualBlock(channels=8)
    x = torch.randn(2, 8, 16, 16, requires_grad=True)
    y = block(x)
    assert y.shape == (2, 8, 16, 16)
    y.sum().backward()
    assert x.grad is not None


def test_transfer_learning_layer_freezing():
    frozen_model = DiseaseClassifierTransferNet(
        num_classes=4, pretrained=False, freeze_all=True
    )
    feat_trainable = sum(
        p.numel() for p in frozen_model.features.parameters() if p.requires_grad
    )
    head_trainable = sum(
        p.numel() for p in frozen_model.classifier.parameters() if p.requires_grad
    )
    assert feat_trainable == 0
    assert head_trainable > 0


def test_full_4_stage_pipeline_and_artifacts():
    if (
        not Path("scores.json").exists()
        or not Path("artifacts/data_ingestion/dataset/train").exists()
    ):
        os.environ["DISABLE_REMOTE_MLFLOW"] = "1"
        scores = run_all_stages()
        assert scores["accuracy"] >= 0.85

    assert Path("artifacts/data_ingestion/dataset/train").exists()
    assert Path("artifacts/prepare_base_model/base_model_updated.pt").exists()
    assert Path("artifacts/training/model.pt").exists()
    assert Path("artifacts/training/model_quantized.pt").exists()
    assert Path("scores.json").exists()


def test_prediction_pipeline_and_fastapi_endpoints():
    sample_img = Path("artifacts/data_ingestion/sample_images/sample_coccidiosis.png")
    assert sample_img.exists()

    pred_pipe = PredictionPipeline(sample_img)
    res = pred_pipe.predict()
    assert res["predicted_class"] in [
        "Coccidiosis",
        "Healthy",
        "Newcastle_Disease",
        "Salmonella",
    ]
    assert res["confidence_percent"] > 50.0

    # FastAPI /health
    health_resp = client.get("/health")
    assert health_resp.status_code == 200
    assert health_resp.json()["status"] == "healthy"

    # FastAPI /metrics
    metrics_resp = client.get("/metrics")
    assert metrics_resp.status_code == 200
    assert metrics_resp.json()["evaluation_scores"]["accuracy"] >= 0.85

    # FastAPI /predict (Base64 payload)
    b64_str = encodeImageIntoBase64(sample_img)
    b64_resp = client.post("/predict", json={"image": b64_str})
    assert b64_resp.status_code == 200
    assert b64_resp.json()["predicted_class"] == "Coccidiosis"

    # FastAPI /predict-upload (Multipart upload)
    with open(sample_img, "rb") as f:
        up_resp = client.post(
            "/predict-upload", files={"file": ("sample_coccidiosis.png", f, "image/png")}
        )
    assert up_resp.status_code == 200
    up_data = up_resp.json()
    assert up_data["predicted_class"] == "Coccidiosis"
    assert "gradcam_overlay_b64" in up_data
    assert "uncertainty" in up_data
    assert "morphology" in up_data


def test_multi_page_routes_and_12_cv_features():
    for route in ["/", "/intro", "/studio", "/batch-triage", "/from-scratch-lab", "/dvc-metrics"]:
        resp = client.get(route)
        assert resp.status_code == 200

    # Browser HTML health & metrics content negotiation
    r_h = client.get("/health", headers={"Accept": "text/html"})
    assert r_h.status_code == 200
    assert "Diagnostic Audit Trail" in r_h.text

    r_m = client.get("/metrics", headers={"Accept": "text/html"})
    assert r_m.status_code == 200
    assert "DVC Reproducible Pipeline" in r_m.text

    # Test interactive 2D Convolution & Micrograd Lab POST
    r_lab = client.post(
        "/from-scratch-lab",
        data={
            "sample_class": "Coccidiosis",
            "kernel_name": "laplacian_lesion_detector",
            "x1": "1.5",
            "x2": "-0.8",
            "w1": "0.6",
            "w2": "-1.1",
            "b": "0.2",
        },
    )
    assert r_lab.status_code == 200
    assert "laplacian_lesion_detector" in r_lab.text
