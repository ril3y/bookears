"""Audio utility functions for Bookears."""

import os
import subprocess
import tempfile

from fastapi import UploadFile


async def save_upload_to_tempfile(upload: UploadFile) -> str:
    """Save a multipart upload to a temporary WAV file.

    Returns the path to the temp file. Caller must clean up with cleanup_tempfile().
    """
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    try:
        content = await upload.read()
        tmp.write(content)
    finally:
        tmp.close()
    return tmp.name


def cleanup_tempfile(path: str):
    """Silently remove a temporary file."""
    try:
        os.unlink(path)
    except OSError:
        pass


def extract_audio_sample(audio_path: str, duration: int = 30, offset: int = 0) -> str:
    """Extract N seconds as 16kHz mono WAV, optionally starting at offset.

    Used by the bakeoff runner (not the /transcribe endpoint).
    Returns the path to the temp WAV file.
    """
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    cmd = ["ffmpeg", "-y"]
    if offset > 0:
        cmd += ["-ss", str(offset)]
    cmd += [
        "-i", str(audio_path),
        "-t", str(duration),
        "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        "-f", "wav", tmp.name,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        cleanup_tempfile(tmp.name)
        raise RuntimeError(f"ffmpeg failed: {result.stderr[:500]}")
    return tmp.name


def get_bitrate_kbps(audio_path: str) -> int | None:
    """Get audio bitrate in kbps via ffprobe. Returns None on failure."""
    try:
        probe = subprocess.run(
            [
                "ffprobe", "-v", "quiet",
                "-show_entries", "format=bit_rate",
                "-of", "csv=p=0",
                audio_path,
            ],
            capture_output=True, text=True, timeout=5,
        )
        if probe.stdout.strip():
            return int(probe.stdout.strip()) // 1000
    except Exception:
        pass
    return None
