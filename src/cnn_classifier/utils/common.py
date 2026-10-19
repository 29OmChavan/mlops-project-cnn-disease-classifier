"""
Common Utilities for YAML, JSON, Directory Management, and Base64 Image Encoding/Decoding
(following Krish Naik's End-to-End Deep Learning MLOps pattern).
"""
import os
import sys
import json
import base64
import yaml
from pathlib import Path
from typing import Any, Dict, List
from src.cnn_classifier.logger import logger
from src.cnn_classifier.exception import CustomException


def read_yaml(path_to_yaml: Path) -> Dict[str, Any]:
    try:
        with open(path_to_yaml, "r", encoding="utf-8") as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"YAML file loaded successfully: {path_to_yaml}")
            return content or {}
    except Exception as e:
        raise CustomException(e, sys)


def create_directories(path_to_directories: List[Any], verbose: bool = True) -> None:
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"Created directory at: {path}")


def save_json(path: Path, data: Dict[str, Any]) -> None:
    try:
        dir_name = os.path.dirname(path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
        logger.info(f"JSON file saved at: {path}")
    except Exception as e:
        raise CustomException(e, sys)


def load_json(path: Path) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = json.load(f)
        logger.info(f"JSON file loaded from: {path}")
        return content
    except Exception as e:
        raise CustomException(e, sys)


def decodeImage(imgstring: str, fileName: Path) -> None:
    """Decodes a base64 image string and writes it to disk."""
    os.makedirs(os.path.dirname(fileName) or ".", exist_ok=True)
    if "," in imgstring:
        imgstring = imgstring.split(",", 1)[1]
    imgdata = base64.b64decode(imgstring)
    with open(fileName, "wb") as f:
        f.write(imgdata)


def encodeImageIntoBase64(croppedImagePath: Path) -> str:
    """Reads an image file and encodes it as a UTF-8 base64 string."""
    with open(croppedImagePath, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")
