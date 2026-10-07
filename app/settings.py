import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

CHROMA_DIR = PROJECT_ROOT / "chroma_db"
DEFAULT_PDF_DIR = PROJECT_ROOT / "pdfs"


def required_setting(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(
            f"{name} is not configured. Copy .env.example to .env and add the required key."
        )
    return value


def optional_setting(name: str) -> str | None:
    value = os.getenv(name, "").strip()
    return value or None


def nvidia_request_timeout() -> float:
    value = os.getenv("NVIDIA_REQUEST_TIMEOUT_SECONDS", "180").strip()
    try:
        timeout = float(value)
    except ValueError as exc:
        raise RuntimeError("NVIDIA_REQUEST_TIMEOUT_SECONDS must be a number of seconds.") from exc
    if timeout <= 0:
        raise RuntimeError("NVIDIA_REQUEST_TIMEOUT_SECONDS must be greater than zero.")
    return timeout
