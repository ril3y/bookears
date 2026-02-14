"""Whisper ASR engine using faster-whisper."""

import logging
import os

import torch

logger = logging.getLogger(__name__)


class WhisperEngine:
    """Manages a faster-whisper model instance for transcription."""

    def __init__(
        self,
        model_size: str = "large-v3-turbo",
        device: str = "cuda",
        compute_type: str = "float16",
        cache_dir: str | None = None,
    ):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.cache_dir = cache_dir or os.environ.get(
            "MODEL_CACHE_DIR", os.path.expanduser("~/.cache/whisper")
        )
        self._model = None

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def load(self):
        """Load the Whisper model onto the device."""
        if self._model is not None:
            return
        from faster_whisper import WhisperModel

        logger.info("Loading Whisper %s (device=%s, compute=%s)...",
                     self.model_size, self.device, self.compute_type)
        self._model = WhisperModel(
            self.model_size,
            device=self.device,
            compute_type=self.compute_type,
            download_root=self.cache_dir,
        )
        logger.info("Whisper %s loaded.", self.model_size)

    def unload(self):
        """Unload the model and free GPU memory."""
        if self._model is not None:
            logger.info("Unloading Whisper...")
            self._model = None
            torch.cuda.empty_cache()

    def transcribe(self, wav_path: str) -> dict:
        """Transcribe a WAV file.

        Returns:
            {
                "text": str,
                "language": str,
                "language_probability": float,
                "first_word_time": float | None,
            }
        """
        if self._model is None:
            raise RuntimeError("Whisper model not loaded. Call load() first.")

        segments, info = self._model.transcribe(
            wav_path, beam_size=5, vad_filter=True
        )
        seg_list = list(segments)

        # Ensure clean UTF-8 (Whisper occasionally emits malformed bytes)
        text = " ".join(seg.text.strip() for seg in seg_list)
        text = text.encode("utf-8", errors="replace").decode("utf-8")

        first_word_time = seg_list[0].start if seg_list else None

        return {
            "text": text,
            "language": info.language,
            "language_probability": round(info.language_probability, 3),
            "first_word_time": round(first_word_time, 2) if first_word_time is not None else None,
        }
