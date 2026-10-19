# Changelog

All notable changes to **Project 02: Deep Learning + Computer Vision Disease Detection System** are documented in this file.

## [1.0.0] - 2026-10-19
### Added
- **DVC Pipeline (`dvc.yaml`)**: Defined reproducible 4-stage DAG (`data_ingestion` $\rightarrow$ `prepare_base_model` $\rightarrow$ `training` $\rightarrow$ `evaluation`) with `scores.json` metric tracking.
- **First-Principles CNN & ResNet Module**: Implemented `conv2d_from_scratch` and `FirstPrinciplesResidualBlock` alongside `DiseaseClassifierTransferNet` (`ResNet-18` transfer learning with configurable `FREEZE_ALL` / `FREEZE_TILL` layer freezing).
- **Data Augmentation & Training Callbacks**: Added stochastic training augmentations (`RandomHorizontalFlip`, `RandomVerticalFlip`, `RandomRotation`, `ColorJitter`), `ModelCheckpoint`, `EarlyStopping`, and `CosineAnnealingLR`.
- **Dynamic INT8 Quantization**: Exported quantized model (`artifacts/training/model_quantized.pt`) and benchmarked FP32 vs. INT8 latency and model size in `scores.json`.
- **DagsHub MLflow Integration**: Logged hyperparameters, multi-class ROC-AUC, Macro/Weighted F1, confusion matrix, and latency benchmarks to DagsHub MLflow.
- **FastAPI Computer Vision Service**: Built `/predict` (Base64 `decodeImage`), `/predict-upload` (multipart file upload), `/health`, `/metrics`, `/train`, and interactive diagnostic UI (`templates/index.html`).
- **Multi-Cloud CI/CD & Documentation**: Added `Dockerfile`, `docker-compose.yml`, `.github/workflows/main.yaml` (Docker Hub + AWS EC2 + Azure Web App), `pytest` suite, and comprehensive role-based documentation in `docs/`.
