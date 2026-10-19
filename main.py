"""
Master Entry Point for Project 02: Deep Learning + Computer Vision Disease Detection System.
Executes all 4 DVC pipeline stages sequentially.
"""
import sys
from src.cnn_classifier.logger import logger
from src.cnn_classifier.exception import CustomException
from src.cnn_classifier.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from src.cnn_classifier.pipeline.stage_02_prepare_base_model import (
    PrepareBaseModelTrainingPipeline,
)
from src.cnn_classifier.pipeline.stage_03_model_training import ModelTrainingPipeline
from src.cnn_classifier.pipeline.stage_04_model_evaluation import EvaluationPipeline


def run_all_stages():
    try:
        STAGE_NAME = "01 - Data Ingestion Stage"
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        DataIngestionTrainingPipeline().main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")

        STAGE_NAME = "02 - Prepare Base Model Stage"
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        PrepareBaseModelTrainingPipeline().main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")

        STAGE_NAME = "03 - Model Training Stage"
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        ModelTrainingPipeline().main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")

        STAGE_NAME = "04 - Model Evaluation Stage"
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        scores = EvaluationPipeline().main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
        return scores
    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    run_all_stages()
