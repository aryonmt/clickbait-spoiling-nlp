import logging
import os
from datetime import datetime


def setup_logger(name: str, log_dir: str = "logs") -> logging.Logger:
    """Sets up a logger that streams to the console and saves logs to a timestamped file.

    Features a defensive propagation guard to prevent duplicate terminal output.

    Args:
        name (str): The module or script name initializing the logger.
        log_dir (str): Directory where log files are stored. Defaults to "logs".

    Returns:
        logging.Logger: An initialized and configured Logger instance.
    """
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Prevent duplicating console logs if third-party libraries (like PyTorch/Transformers)
    # have configured their own root loggers.
    logger.propagate = False

    # Prevent adding duplicate handlers if the logger is already configured
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File Handler
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{name}_{timestamp}.log")
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
