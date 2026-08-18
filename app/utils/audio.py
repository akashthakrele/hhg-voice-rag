"""
Audio preprocessing utilities.
Sarvam STT API natively supports WAV, WebM, OGG, MP3, etc.
so we pass raw bytes through and just detect the format for the API call.
"""

from __future__ import annotations

from pathlib import Path

import structlog

logger = structlog.get_logger(__name__)

# Formats natively supported by Sarvam STT API — no conversion needed
SARVAM_SUPPORTED_FORMATS = {"wav", "mp3", "ogg", "webm", "m4a", "flac", "aac", "mp4", "opus"}


async def preprocess_audio(audio_bytes: bytes, filename: str = "audio") -> tuple[bytes, str]:
    """
    Validate and detect format of uploaded audio for Sarvam STT API.

    Sarvam natively supports WebM, WAV, MP3, OGG, etc.,
    so we pass raw bytes through without conversion (no pydub/ffmpeg needed).

    Args:
        audio_bytes: Raw uploaded audio bytes.
        filename: Original filename (used to detect format).

    Returns:
        Tuple of (audio_bytes, detected_format).
    """
    # Detect format from extension
    ext = Path(filename).suffix.lower().lstrip(".")
    if ext not in SARVAM_SUPPORTED_FORMATS:
        ext = "wav"  # fallback

    logger.info(
        "audio_preprocessed",
        original_format=ext,
        size_bytes=len(audio_bytes),
    )
    return audio_bytes, ext


def validate_audio_size(audio_bytes: bytes, max_mb: float = 25.0) -> None:
    """Raise ValueError if audio exceeds size limit."""
    size_mb = len(audio_bytes) / (1024 * 1024)
    if size_mb > max_mb:
        raise ValueError(
            f"Audio file too large: {size_mb:.1f}MB (max {max_mb}MB)"
        )
