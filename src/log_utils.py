"""Unified logging setup: console (INFO) + file (DEBUG)."""

import logging
import sys
import time
from pathlib import Path


def setup_logging(
    log_level: str = "INFO",
    log_dir: str = "logs",
    run_name: str = "run",
) -> logging.Logger:
    """Configure root logger with console + file handlers.

    Args:
        log_level: Console log level (e.g. "INFO", "DEBUG").
        log_dir: Directory for log files.
        run_name: Used as log filename: {log_dir}/{run_name}.log

    Returns:
        Root logger (callers should use logging.getLogger(__name__) in modules).
    """
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)  # capture everything, handlers filter

    # Clear existing handlers (avoid duplicates on re-init)
    root.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Console handler — INFO level, stderr
    console = logging.StreamHandler(sys.stderr)
    console.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    console.setFormatter(fmt)
    root.addHandler(console)

    # File handler — DEBUG level, full detail
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = Path(log_dir) / timestamp
    log_path.mkdir(parents=True, exist_ok=True)
    filepath = log_path / f"{run_name}.log"
    file_h = logging.FileHandler(filepath, mode="w", encoding="utf-8")
    file_h.setLevel(logging.DEBUG)
    file_h.setFormatter(fmt)
    root.addHandler(file_h)

    # Suppress noisy third-party loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("py4j").setLevel(logging.WARNING)
    logging.getLogger("py4j.java_gateway").setLevel(logging.WARNING)
    logging.getLogger("py4j.clientserver").setLevel(logging.WARNING)

    root.info(f"Logging to console ({log_level}) and file ({filepath})")
    return root
