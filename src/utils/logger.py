import logging
import sys
from logging.handlers import RotatingFileHandler

def setup_enterprise_logger(name: str) -> logging.Logger:
    """
    Configures a production-grade logger with console and rotating file handlers.
    Ensures all outputs are formatted for log aggregation tools (e.g., Datadog, ELK).
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Prevent adding multiple handlers if logger is instantiated multiple times
    if not logger.handlers:
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        # 1. Console Handler (Crucial for Docker containers)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # 2. File Handler with Rotation (Max 10MB per file, keep 5 backups)
        file_handler = RotatingFileHandler(
            "agent_system_core.log", 
            maxBytes=10*1024*1024, 
            backupCount=5,
            encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

# Export a singleton logger instance for system-wide usage
logger = setup_enterprise_logger("NextGen_Agent_OS")
