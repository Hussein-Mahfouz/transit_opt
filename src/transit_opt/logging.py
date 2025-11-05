import logging
from pathlib import Path


def setup_logger(name: str, log_dir: str, log_file: str = "run.log",
                 console_level: str = "INFO", file_level: str = "DEBUG"):
    """
    Set up a logger that writes to both console and a file in log_dir.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # Capture all levels, handlers will filter

    # Avoid duplicate handlers
    if not logger.handlers:
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(getattr(logging, console_level.upper(), logging.INFO))
        ch_fmt = logging.Formatter("[%(levelname)s] %(message)s")
        ch.setFormatter(ch_fmt)
        logger.addHandler(ch)

        # File handler
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(Path(log_dir) / log_file, encoding="utf-8")
        fh.setLevel(getattr(logging, file_level.upper(), logging.DEBUG))
        fh_fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        fh.setFormatter(fh_fmt)
        logger.addHandler(fh)

    return logger
