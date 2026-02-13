from __future__ import annotations

import ctypes
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import time
from typing import Optional


def setup_logger(log_file: str, name: str = "autopilot") -> logging.Logger:
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = RotatingFileHandler(log_file, maxBytes=2_000_000, backupCount=3, encoding="utf-8")
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        logger.addHandler(sh)
    return logger


def monotonic_ms() -> int:
    return int(time.monotonic() * 1000)


def is_window_active(title_contains: Optional[str]) -> bool:
    if not title_contains:
        return True
    try:
        user32 = ctypes.windll.user32
        hwnd = user32.GetForegroundWindow()
        if hwnd == 0:
            return False
        length = user32.GetWindowTextLengthW(hwnd)
        buf = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buf, length + 1)
        return title_contains.lower() in buf.value.lower()
    except Exception:
        return False
