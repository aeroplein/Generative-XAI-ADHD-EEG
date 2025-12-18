import logging
import sys
from datetime import datetime
from pathlib import Path

class LoggerHelper:
    _logger = None

    @staticmethod
    def setup(log_dir:str="logs",
              level:int=logging.INFO,
              name:str="eeg_pipeline"):
        
        if LoggerHelper._logger is not None:
            return LoggerHelper._logger
        
        Path(log_dir).mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = Path(log_dir)/f"{name}_{timestamp}.txt"

        logger_instance = logging.getLogger(name)
        logger_instance.setLevel(level)
        logger_instance.propagate = False
        
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(name)s - %(message)s", 
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)

        logger_instance.addHandler(file_handler)
        logger_instance.addHandler(console_handler)

        LoggerHelper._logger = logger_instance
        return logger_instance
    
    @staticmethod
    def get_logger():
        if LoggerHelper._logger is None:
            return LoggerHelper.setup()
        return LoggerHelper._logger
    
    @staticmethod
    def info(msg:str):
        LoggerHelper.get_logger().info(msg)
    
    @staticmethod
    def warning(msg:str):
        LoggerHelper.get_logger().warning(msg)

    @staticmethod
    def error(msg:str):
        LoggerHelper.get_logger().error(msg)

    @staticmethod
    def debug(msg:str):
        LoggerHelper.get_logger().debug(msg)