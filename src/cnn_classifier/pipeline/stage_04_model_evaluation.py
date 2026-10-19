"""
DVC Stage 04: Model Evaluation & MLflow Tracking Pipeline Runner
"""
import sys
from src.cnn_classifier.logger import logger
from src.cnn_classifier.exception import CustomException
from src.cnn_classifier.config.configuration import ConfigurationManager
from src.cnn_classifier.components.model_evaluation import Evaluation

STAGE_NAME = "Evaluation Stage"


class EvaluationPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        eval_config = config.get_evaluation_config()
        evaluation = Evaluation(eval_config)
        scores = evaluation.evaluation()
        evaluation.log_into_mlflow()
        return scores


if __name__ == "__main__":
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = EvaluationPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        raise CustomException(e, sys)
