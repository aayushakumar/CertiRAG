"""
CertiRAG Utilities
===================

Shared helper functions for logging, reproducibility, hashing,
and text processing used across all modules.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np


# ── Reproducibility ────────────────────────────────────────────────

def set_all_seeds(seed: int = 42) -> None:
    """
    Set random seeds for Python, NumPy, and (optionally) PyTorch.

    This ensures reproducible results across runs. Must be called
    before any random operations.

    Args:
        seed: Integer seed value. Default is 42.
    """
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass  # torch not installed (lite mode)


def generate_run_id() -> str:
    """
    Generate a unique run ID for tracking experiments.

    Format: certirag-{timestamp}-{short_uuid}
    Example: certirag-20250209-143022-a1b2c3d4
    """
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    short_id = uuid.uuid4().hex[:8]
    return f"certirag-{timestamp}-{short_id}"


# ── Hashing ────────────────────────────────────────────────────────

def compute_hash(data: str | bytes | dict, length: int = 16) -> str:
    """
    Compute a truncated SHA-256 hash.

    Used for:
    - Config hashing (reproducibility stamps)
    - Certificate integrity verification
    - Model version tracking

    Args:
        data: String, bytes, or dict to hash.
        length: Number of hex characters to return (max 64).

    Returns:
        Hex digest string of specified length.
    """
    if isinstance(data, dict):
        data = json.dumps(data, sort_keys=True, default=str)
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()[:length]


def compute_content_hash(obj: Any) -> str:
    """
    Compute a content-addressable hash for any JSON-serializable object.

    This is used for certificate integrity: the hash of the certificate
    content (excluding the hash field) is stored in the certificate.
    """
    canonical = json.dumps(obj, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ── Logging ────────────────────────────────────────────────────────

def setup_logging(
    level: str = "INFO",
    format_style: str = "text",
    run_id: str | None = None
) -> logging.Logger:
    """
    Configure structured logging for CertiRAG.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR).
        format_style: "json" for structured logs, "text" for human-readable.
        run_id: Optional run ID to include in all log entries.

    Returns:
        Configured Logger instance.
    """
    logger = logging.getLogger("certirag")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Remove existing handlers
    logger.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)

    if format_style == "json":
        class JsonFormatter(logging.Formatter):
            def format(self, record):
                log_entry = {
                    "timestamp": self.formatTime(record),
                    "level": record.levelname,
                    "module": record.module,
                    "message": record.getMessage(),
                }
                if run_id:
                    log_entry["run_id"] = run_id
                if record.exc_info:
                    log_entry["exception"] = self.formatException(record.exc_info)
                return json.dumps(log_entry)

        handler.setFormatter(JsonFormatter())
    else:
        fmt = "%(asctime)s | %(levelname)-8s | %(name)s.%(module)s | %(message)s"
        if run_id:
            fmt = f"%(asctime)s | %(levelname)-8s | {run_id} | %(name)s.%(module)s | %(message)s"
        handler.setFormatter(logging.Formatter(fmt, datefmt="%H:%M:%S"))

    logger.addHandler(handler)
    return logger


# ── Text Processing Helpers ────────────────────────────────────────

def truncate_text(text: str, max_tokens: int = 512, model: str = "gpt-4o-mini") -> str:
    """
    Truncate text to approximately max_tokens.

    Uses tiktoken if available (accurate), otherwise falls back to
    word-based estimation (~0.75 tokens per word).

    Args:
        text: Input text to truncate.
        max_tokens: Maximum number of tokens.
        model: Model name for tiktoken encoding.

    Returns:
        Truncated text string.
    """
    try:
        import tiktoken
        enc = tiktoken.encoding_for_model(model)
        tokens = enc.encode(text)
        if len(tokens) <= max_tokens:
            return text
        return enc.decode(tokens[:max_tokens])
    except ImportError:
        # Fallback: rough word-based estimation
        words = text.split()
        estimated_max_words = int(max_tokens * 0.75)
        if len(words) <= estimated_max_words:
            return text
        return " ".join(words[:estimated_max_words]) + "..."


def count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    """
    Count the number of tokens in a text string.

    Args:
        text: Input text.
        model: Model name for tiktoken encoding.

    Returns:
        Token count.
    """
    try:
        import tiktoken
        enc = tiktoken.encoding_for_model(model)
        return len(enc.encode(text))
    except ImportError:
        # Fallback: rough word-based estimation
        return int(len(text.split()) / 0.75)


def normalize_whitespace(text: str) -> str:
    """Collapse multiple whitespace characters into single spaces and strip."""
    return " ".join(text.split())


# ── File I/O Helpers ───────────────────────────────────────────────

def save_json(data: Any, path: str | Path, indent: int = 2) -> Path:
    """Save data as formatted JSON file with UTF-8 encoding."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False, default=str)
    return path


def load_json(path: str | Path) -> Any:
    """Load a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
