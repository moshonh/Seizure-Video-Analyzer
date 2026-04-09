"""
utils.py
Shared utility functions for the Seizure Video Analyzer.
"""

import os
import tempfile

ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}


def save_uploaded_file(uploaded_file) -> str:
    """
    Persist a Streamlit UploadedFile to a temporary path on disk.

    Returns
    -------
    Absolute path to the saved file.
    """
    suffix = os.path.splitext(uploaded_file.name)[-1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        return tmp.name


def is_valid_video(uploaded_file) -> bool:
    """Return True if the uploaded file has an accepted video extension."""
    suffix = os.path.splitext(uploaded_file.name)[-1].lower()
    return suffix in ALLOWED_EXTENSIONS


def format_duration(seconds: float) -> str:
    """Format a duration in seconds as 'Xm Ys'."""
    m, s = divmod(int(max(seconds, 0)), 60)
    return f"{m}m {s}s"


def classification_emoji(classification: str) -> str:
    return {"PNES": "🟡", "ES": "🔵", "Indeterminate": "⚪"}.get(classification, "❓")


def confidence_colour(confidence: str) -> str:
    return {"high": "green", "moderate": "orange", "low": "red"}.get(confidence, "gray")
