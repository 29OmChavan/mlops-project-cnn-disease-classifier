"""
Immutable Dataclasses for Project 02 CNN Disease Classifier Pipeline Stages.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    dataset_dir: Path
    classes: List[str]
    samples_per_class: int
    val_split: float
    image_size: List[int]


@dataclass(frozen=True)
class PrepareBaseModelConfig:
    root_dir: Path
    base_model_path: Path
    updated_base_model_path: Path
    params_image_size: List[int]
    params_learning_rate: float
    params_include_top: bool
    params_weights: str
    params_classes: int
    params_freeze_all: bool
    params_freeze_till: int
    params_dropout_rate: float


@dataclass(frozen=True)
class TrainingConfig:
    root_dir: Path
    trained_model_path: Path
    quantized_model_path: Path
    training_history_path: Path
    updated_base_model_path: Path
    training_data: Path
    params_epochs: int
    params_batch_size: int
    params_is_augmentation: bool
    params_image_size: List[int]
    params_learning_rate: float
    params_early_stopping_patience: int
    params_classes: int


@dataclass(frozen=True)
class EvaluationConfig:
    path_of_model: Path
    quantized_model_path: Path
    training_data: Path
    score_file: Path
    mlflow_uri: str
    experiment_name: str
    registered_model_name: str
    all_params: dict
    params_image_size: List[int]
    params_batch_size: int
    classes: List[str]
