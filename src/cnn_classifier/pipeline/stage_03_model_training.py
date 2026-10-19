"""
DVC Stage 03: Model Training Pipeline Runner
"""
import sys
from src.cnn_classifier.logger import logger
from src.cnn_classifier.exception import CustomException
from src.cnn_classifier.config.configuration import ConfigurationManager
from src.cnn_classifier.components.model_training import ModelTraining

STAGE_NAME = "Model Training Stage"


class ModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        training_config = config.get_training_config()
        training = ModelTraining(config=training_config)
        return training.train()


if __name__ == "__main__":
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        raise CustomException(e, sys)
