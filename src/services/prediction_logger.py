"""
Structured Prediction Logger for JobClarity (Day 7).

Provides thread-safe, resilient structured prediction logging to JSONL storage:
  - Generates unique prediction ID (UUID4) and UTC timestamp (ISO 8601).
  - Captures endpoint, model name/version, probability, threshold, risk tier, and latency.
  - Computes non-reversible text hash (SHA-256) and text length metrics for auditability.
  - STRICT PRIVACY GUARANTEE: Never stores raw job description text or personal identifiers.
  - FAULT TOLERANCE: Logging failures are caught safely and never break model inference.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import threading
from typing import Any, Mapping
import uuid

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOG_DIR = PROJECT_ROOT / "logs"
DEFAULT_LOG_FILE = DEFAULT_LOG_DIR / "predictions.jsonl"

_LOG_LOCK = threading.Lock()


def get_prediction_log_path() -> Path:
    """Return default prediction log file path, ensuring parent directory exists."""
    DEFAULT_LOG_DIR.mkdir(parents=True, exist_ok=True)
    return DEFAULT_LOG_FILE


def hash_text(text: str) -> str:
    """Return SHA-256 hash of text for privacy-preserving deduplication and audit tracking."""
    if not isinstance(text, str):
        return ""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def log_prediction(
    endpoint: str,
    model_name: str,
    model_version: str,
    probability: float | None,
    decision_threshold: float | dict[str, float] | None,
    risk_tier: str | None,
    latency_ms: float | None = None,
    job_description: str | None = None,
    action: str | None = None,
    status: str = "success",
    error_message: str | None = None,
    extra_metadata: Mapping[str, Any] | None = None,
    log_file_path: Path | str | None = None,
) -> dict[str, Any] | None:
    """
    Log prediction event in structured JSON Lines format.

    Guarantees:
      - Raw text is NEVER written to the log.
      - Any logging error is swallowed and logged to error stream, never raising to caller.

    Returns:
        The logged record dictionary on success, or None on failure.
    """
    try:
        prediction_id = str(uuid.uuid4())
        timestamp_utc = datetime.now(timezone.utc).isoformat()

        # Compute privacy-preserving metadata from raw text without retaining text
        char_length = len(job_description) if isinstance(job_description, str) else 0
        word_count = len(job_description.split()) if isinstance(job_description, str) else 0
        text_sha256 = hash_text(job_description) if job_description else None

        prob_rounded = (
            round(float(probability), 4)
            if probability is not None and not isinstance(probability, bool)
            else None
        )
        latency_rounded = (
            round(float(latency_ms), 2)
            if latency_ms is not None and not isinstance(latency_ms, bool)
            else None
        )

        record: dict[str, Any] = {
            "prediction_id": prediction_id,
            "timestamp_utc": timestamp_utc,
            "endpoint": str(endpoint),
            "model_name": str(model_name),
            "model_version": str(model_version),
            "probability": prob_rounded,
            "decision_threshold": decision_threshold,
            "risk_tier": str(risk_tier) if risk_tier is not None else None,
            "action": str(action) if action is not None else None,
            "latency_ms": latency_rounded,
            "input_char_length": char_length,
            "input_word_count": word_count,
            "input_sha256": text_sha256,
            "status": str(status),
            "error_message": str(error_message) if error_message is not None else None,
        }

        if extra_metadata:
            # Safely merge extra metadata without overwriting primary schema keys
            for k, v in extra_metadata.items():
                if k not in record and k != "raw_text" and k != "job_description":
                    record[k] = v

        target_path = Path(log_file_path) if log_file_path is not None else get_prediction_log_path()
        target_path.parent.mkdir(parents=True, exist_ok=True)

        json_line = json.dumps(record, ensure_ascii=False) + "\n"

        with _LOG_LOCK:
            with open(target_path, "a", encoding="utf-8") as f:
                f.write(json_line)

        return record

    except Exception as exc:
        # Logging failures must not break inference
        logger.error(
            "Failed to record prediction log entry: %s",
            str(exc),
            exc_info=True,
        )
        return None


def get_recent_prediction_logs(
    limit: int = 100,
    log_file_path: Path | str | None = None,
) -> list[dict[str, Any]]:
    """
    Retrieve the most recent prediction log entries.
    """
    target_path = Path(log_file_path) if log_file_path is not None else get_prediction_log_path()
    if not target_path.exists():
        return []

    entries: list[dict[str, Any]] = []
    try:
        with _LOG_LOCK:
            with open(target_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            entries.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
        return entries[-limit:]
    except Exception as exc:
        logger.warning("Error reading prediction logs: %s", str(exc))
        return []


def clear_prediction_logs(log_file_path: Path | str | None = None) -> None:
    """Clear prediction log file (used primarily for test cleanup and isolation)."""
    target_path = Path(log_file_path) if log_file_path is not None else get_prediction_log_path()
    try:
        with _LOG_LOCK:
            if target_path.exists():
                target_path.unlink()
    except Exception as exc:
        logger.warning("Failed to clear prediction log file: %s", str(exc))
