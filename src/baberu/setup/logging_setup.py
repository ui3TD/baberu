from baberu.__main__ import APP_NAME

import platformdirs

import sys
import logging
import logging.handlers
from pathlib import Path


def setup_logging(
    console_level: str = "WARNING",
    file_level: str = "DEBUG",
    log_to_file: bool = False,
    log_dir: Path | None = None
):
    """
    Configures logging with separate levels for console and file output.

    Args:
        console_level (str): The logging level for console output (e.g., 'INFO', 'WARNING').
        file_level (str): The logging level for file output (e.g., 'DEBUG').
        log_to_file (bool): Whether to enable file logging.
    """
    logger = logging.getLogger()
    logger.setLevel(logging.NOTSET)
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level.upper())
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    logging.info(f"Console logging configured at level {console_level.upper()}.")

    if log_to_file:
        if not log_dir:
            log_dir = Path(platformdirs.user_state_dir(APP_NAME, appauthor=False))

        log_dir.mkdir(parents=True, exist_ok=True)
        log_file_path = log_dir / f"{APP_NAME}.log"

        file_handler = logging.handlers.RotatingFileHandler(
            log_file_path, maxBytes=1_000_000, backupCount=5, encoding='utf-8'
        )
        file_handler.setLevel(file_level.upper())
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        if log_file_path.resolve() != log_file_path:
            logging.warning(f"File logging configured at level {file_level.upper()} to {log_file_path.resolve()}")
        else:
            logging.info(f"File logging configured at level {file_level.upper()} to {log_file_path}")