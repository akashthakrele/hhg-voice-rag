"""
Audio preprocessing utilities.
Converts uploaded audio to the format expected by Sarvam STT API.
"""

from __future__ import annotations

import io
import tempfile
from pathlib import Path

import structlog

logger = structlog.get_logger(__name__)

# Sarvam expects: WAV, 16kHz, mono, 16-bit PCM
TARGET_SAMPLE_RATE = 16000
TARGET_CHANNELS = 1


async def preprocess_audio(audio_bytes: bytes, filename: str = "audio") -> bytes:
    """
    Normalize uploaded audio to WAV format suitable for STT.

    Args:
        audio_bytes: Raw uploaded audio bytes.
        filename: Original filename (used to detect format).

    Returns:
        Normalized WAV bytes (16kHz, mono, 16-bit PCM).
    """
    from pydub import AudioSegment

    # Detect format from extension
    ext = Path(filename).suffix.lower().lstrip(".")
    if ext not in ("wav", "mp3", "ogg", "webm", "m4a", "flac"):
        ext = "wav"  # fallback

    try:
        audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format=ext)
    except Exception:
        logger.warning("audio_format_detection_failed", filename=filename, fallback="wav")
        audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format="wav")

    # Normalize
    audio = (
        audio
        .set_frame_rate(TARGET_SAMPLE_RATE)
        .set_channels(TARGET_CHANNELS)
        .set_sample_width(2)  # 16-bit
    )

    # Export to WAV bytes
    buf = io.BytesIO()
    audio.export(buf, format="wav")
    buf.seek(0)

    logger.info(
        "audio_preprocessed",
        original_format=ext,
        duration_ms=len(audio),
        sample_rate=TARGET_SAMPLE_RATE,
    )
    return buf.read()


def validate_audio_size(audio_bytes: bytes, max_mb: float = 25.0) -> None:
    """Raise ValueError if audio exceeds size limit."""
    size_mb = len(audio_bytes) / (1024 * 1024)
    if size_mb > max_mb:
        raise ValueError(
            f"Audio file too large: {size_mb:.1f}MB (max {max_mb}MB)"
        )
