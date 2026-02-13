"""Structured logging configuration.

Instead of plain text logs like:
    INFO: Processing transcript for Sarah Cohen

We output JSON logs like:
    {"timestamp": "2024-01-15T10:30:00Z", "level": "INFO", "message": "Processing transcript", "request_id": "abc123", "duration_ms": 1234}

This makes logs machine-parseable for tools like Datadog, ELK, or CloudWatch.
"""

import logging
import json
import sys
from datetime import datetime, timezone


class JSONFormatter(logging.Formatter):
    """Format log records as JSON for structured logging.

    Each log line is a single JSON object with consistent fields,
    making it easy to search, filter, and aggregate in log tools.
    """

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # Add any extra fields passed via `logger.info("msg", extra={...})`
        # This is how we attach request_id, duration, token counts, etc.
        if hasattr(record, "extra_data"):
            log_data.update(record.extra_data)
        return json.dumps(log_data)


def setup_logging(level: str = "INFO") -> None:
    """Configure the root logger with JSON formatting.

    Call this once at app startup. All loggers created with
    logging.getLogger(__name__) will inherit this configuration.
    """
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONFormatter())

    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    root_logger.addHandler(handler)

    # Quiet down noisy third-party loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
