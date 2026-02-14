"""Detection heuristics for audio content classification.

These are used by the bakeoff runner and can also be called by
consumers of the /transcribe endpoint (via the returned fields).
"""

import re

from .audio import get_bitrate_kbps

# Dramatic intro patterns (Graphic Audio, full-cast productions)
_DRAMATIC_INTRO_RE = re.compile(
    r"graphic\s*audio|a\s+movie\s+in\s+your\s+mind|full\s+cast\s+production",
    re.IGNORECASE,
)

# Cassette rip patterns (old Recorded Books, etc.)
_CASSETTE_RE = re.compile(
    r"(?:side\s+\d|four\s+sides?\s+per\s+cassette|R\.?C\.?\s*[-.]?\s*\d{3,6})",
    re.IGNORECASE,
)


def detect_music_intro(whisper_result: dict, threshold: float = 10.0) -> bool:
    """Detect a music intro based on first_word_time from Whisper.

    Returns True if speech starts after `threshold` seconds or no speech at all.
    """
    ft = whisper_result.get("first_word_time")
    text = (whisper_result.get("text") or "").strip()

    if ft is None and not text:
        # No speech found at all
        return True
    if ft is not None and ft > threshold:
        # Speech starts late
        return True
    return False


def detect_dramatic_intro(text: str) -> bool:
    """Detect dramatic intro patterns (Graphic Audio, etc.) in transcript text."""
    if not text:
        return False
    return bool(_DRAMATIC_INTRO_RE.search(text))


def detect_non_english(whisper_result: dict) -> dict | None:
    """Detect non-English audio based on Whisper language detection.

    Returns {"language": str, "probability": float} if non-English with prob > 0.5,
    otherwise None.
    """
    lang = whisper_result.get("language", "en")
    prob = whisper_result.get("language_probability", 0)
    if lang != "en" and prob > 0.5:
        return {"language": lang, "probability": prob}
    return None


def detect_aged_media(text: str, audio_path: str | None = None) -> dict | None:
    """Detect aged/cassette media from transcript patterns and optional bitrate check.

    Returns {"cassette_pattern": True, "bitrate_kbps": int|None} if detected,
    otherwise None.
    """
    if not text or not _CASSETTE_RE.search(text):
        return None

    bitrate = None
    if audio_path:
        bitrate = get_bitrate_kbps(audio_path)

    return {"cassette_pattern": True, "bitrate_kbps": bitrate}
