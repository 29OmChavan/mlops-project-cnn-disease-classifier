"""
Custom Exception Handling with Traceback Inspection for Project 02.
"""
import sys
from src.cnn_classifier.logger import logger


def error_message_detail(error: Exception, error_detail: sys) -> str:
    _, _, exc_tb = error_detail.exc_info()
    if exc_tb is not None:
        file_name = exc_tb.tb_frame.f_code.co_filename
        line_number = exc_tb.tb_lineno
    else:
        file_name = "unknown"
        line_number = 0

    return (
        f"Error occurred in python script name [{file_name}] "
        f"line number [{line_number}] error message [{str(error)}]"
    )


class CustomException(Exception):
    def __init__(self, error_message: Exception, error_detail: sys):
        super().__init__(str(error_message))
        self.error_message = error_message_detail(error_message, error_detail=error_detail)
        logger.error(self.error_message)

    def __str__(self) -> str:
        return self.error_message
