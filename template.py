"""
Project 02 Scaffolding Script (Krish Naik Deep Learning + DVC MLOps Pattern)
Creates the modular structure for the CNN Disease Classification system.
"""
import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="[%(asctime)s]: %(message)s")

project_name = "cnn_classifier"

list_of_files = [
    ".github/workflows/main.yaml",
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/components/data_ingestion.py",
    f"src/{project_name}/components/prepare_base_model.py",
    f"src/{project_name}/components/model_training.py",
    f"src/{project_name}/components/model_evaluation.py",
    f"src/{project_name}/utils/__init__.py",
    f"src/{project_name}/utils/common.py",
    f"src/{project_name}/config/__init__.py",
    f"src/{project_name}/config/configuration.py",
    f"src/{project_name}/pipeline/__init__.py",
    f"src/{project_name}/pipeline/stage_01_data_ingestion.py",
    f"src/{project_name}/pipeline/stage_02_prepare_base_model.py",
    f"src/{project_name}/pipeline/stage_03_model_training.py",
    f"src/{project_name}/pipeline/stage_04_model_evaluation.py",
    f"src/{project_name}/pipeline/prediction.py",
    f"src/{project_name}/entity/__init__.py",
    f"src/{project_name}/entity/config_entity.py",
    f"src/{project_name}/constants/__init__.py",
    f"src/{project_name}/logger.py",
    f"src/{project_name}/exception.py",
    "config/config.yaml",
    "params.yaml",
    "dvc.yaml",
    "main.py",
    "app.py",
    "Dockerfile",
    "docker-compose.yml",
    ".dockerignore",
    ".gitignore",
    "requirements.txt",
    "setup.py",
    "templates/index.html",
    "notebooks/01_cnn_from_scratch_and_transfer_learning.py",
    "tests/__init__.py",
    "tests/test_cnn_pipeline.py",
    "docs/01_DATA_AND_DVC_SPEC.md",
    "docs/02_EXPERIMENT_LOG_AND_MODEL_CARD.md",
    "docs/03_MULTICLOUD_DEPLOYMENT_RUNBOOK.md",
    "docs/IMPLEMENTATION_GUIDE.md",
    "CHANGELOG.md",
    "README.md",
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory: {filedir} for the file: {filename}")

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w", encoding="utf-8") as f:
            pass
        logging.info(f"Creating empty file: {filepath}")
    else:
        logging.info(f"{filename} already exists")
