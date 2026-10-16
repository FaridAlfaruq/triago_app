"""Shared environment configuration for the kiosk and backend."""
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config():
    # Explicit environment variables take precedence over the local .env file.
    load_dotenv(PROJECT_ROOT / ".env", override=False)
