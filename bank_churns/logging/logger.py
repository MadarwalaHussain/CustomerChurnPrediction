# Centralized logging configuration for the Bank Churn Prediction system.

import logging
import os
from datetime import datetime
from pathlib import Path

# create logs directory if it doesn't exist
LOG_DIR= 'logs'
os.makedirs(LOG_DIR, exist_ok=True)

# Generate log file name with timestamp
LOG_FILE_NAME = f"bankchurn_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
LOG_FILE_PATH = os.path.join(LOG_DIR, LOG_FILE_NAME)

# Configure logging format
LOG_FORMAT = "[%(asctime)s] %(levelname)s [%(name)s:%(lineno)d] - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

def setup_logger(
        name:str='bankchurns',
        log_file:str = LOG_FILE_PATH,
        level: int=logging.INFO
        )-> logging.Logger:
    """
    Set up and return a configured logger instance.
    
    Args:
        name: Logger name
        log_file: Path to log file
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        
    Returns:
        Configured logger instance
    """

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # avoid duplicate handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    # File handler -writes to file
    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setLevel(level)
    file_formatter= logging.Formatter(LOG_FORMAT,datefmt=DATE_FORMAT) 
    file_handler.setFormatter(file_formatter)

    # Console handler - writes to stdout
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter("%(levelname)s - %(message)s")
    console_handler.setFormatter(console_formatter)

    # add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


# Create default logger instance
logging = setup_logger()