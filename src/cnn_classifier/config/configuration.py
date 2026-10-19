"""
ConfigurationManager for Project 02 CNN Disease Classifier.
"""
from pathlib import Path
from src.cnn_classifier.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH
from src.cnn_classifier.utils.common import read_yaml, create_directories
from src.cnn_classifier.entity.config_entity import (
    DataIngestionConfig,
    PrepareBaseModelConfig,
    TrainingConfig,
    EvaluationConfig,
)


class ConfigurationManager:
    def __init__(
        self,
        config_filepath: Path = CONFIG_FILE_PATH,
        params_filepath: Path = PARAMS_FILE_PATH,
    ):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        create_directories([self.config["artifacts_root"]])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config["data_ingestion"]
        create_directories([config["root_dir"], config["dataset_dir"]])
        return DataIngestionConfig(
            root_dir=Path(config["root_dir"]),
            dataset_dir=Path(config["dataset_dir"]),
            classes=list(config["classes"]),
            samples_per_class=int(config["samples_per_class"]),
            val_split=float(config["val_split"]),
            image_size=list(self.params["IMAGE_SIZE"]),
        )

    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        config = self.config["prepare_base_model"]
        create_directories([config["root_dir"]])
        return PrepareBaseModelConfig(
            root_dir=Path(config["root_dir"]),
            base_model_path=Path(config["base_model_path"]),
            updated_base_model_path=Path(config["updated_base_model_path"]),
            params_image_size=list(self.params["IMAGE_SIZE"]),
            params_learning_rate=float(self.params["LEARNING_RATE"]),
            params_include_top=bool(self.params["INCLUDE_TOP"]),
            params_weights=str(self.params["WEIGHTS"]),
            params_classes=int(self.params["CLASSES"]),
            params_freeze_all=bool(self.params["FREEZE_ALL"]),
            params_freeze_till=int(self.params["FREEZE_TILL"]),
            params_dropout_rate=float(self.params["DROPOUT_RATE"]),
        )

    def get_training_config(self) -> TrainingConfig:
        training = self.config["training"]
        prepare_base_model = self.config["prepare_base_model"]
        ingestion = self.config["data_ingestion"]
        create_directories([training["root_dir"]])

        return TrainingConfig(
            root_dir=Path(training["root_dir"]),
            trained_model_path=Path(training["trained_model_path"]),
            quantized_model_path=Path(training["quantized_model_path"]),
            training_history_path=Path(training["training_history_path"]),
            updated_base_model_path=Path(prepare_base_model["updated_base_model_path"]),
            training_data=Path(ingestion["dataset_dir"]),
            params_epochs=int(self.params["EPOCHS"]),
            params_batch_size=int(self.params["BATCH_SIZE"]),
            params_is_augmentation=bool(self.params["AUGMENTATION"]),
            params_image_size=list(self.params["IMAGE_SIZE"]),
            params_learning_rate=float(self.params["LEARNING_RATE"]),
            params_early_stopping_patience=int(self.params["EARLY_STOPPING_PATIENCE"]),
            params_classes=int(self.params["CLASSES"]),
        )

    def get_evaluation_config(self) -> EvaluationConfig:
        eval_cfg = self.config["evaluation"]
        ingestion = self.config["data_ingestion"]
        create_directories([eval_cfg["root_dir"]])

        return EvaluationConfig(
            path_of_model=Path(eval_cfg["path_of_model"]),
            quantized_model_path=Path(eval_cfg["quantized_model_path"]),
            training_data=Path(eval_cfg["training_data"]),
            score_file=Path(eval_cfg["score_file"]),
            mlflow_uri=str(eval_cfg["mlflow_uri"]),
            experiment_name=str(eval_cfg["experiment_name"]),
            registered_model_name=str(eval_cfg["registered_model_name"]),
            all_params=dict(self.params),
            params_image_size=list(self.params["IMAGE_SIZE"]),
            params_batch_size=int(self.params["BATCH_SIZE"]),
            classes=list(ingestion["classes"]),
        )
