# src/puv/logging_setup.py
import logging

def setup(level=logging.INFO):
    """Configure root logging once per session (safe to re-run)."""
    root = logging.getLogger()
    if not root.handlers:  # prevents duplicate handlers if run multiple times
        logging.basicConfig(
            level=level,
            format="%(levelname)s [%(name)s] %(message)s"
        )