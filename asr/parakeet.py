"""Parakeet (NeMo TDT) ASR engine."""

import logging

import soundfile as sf
import torch

logger = logging.getLogger(__name__)


class ParakeetEngine:
    """Manages a NeMo Parakeet model instance for transcription."""

    def __init__(
        self,
        model_name: str = "nvidia/parakeet-tdt-0.6b-v2",
        device: str = "cuda",
    ):
        self.model_name = model_name
        self.device = device
        self._model = None

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def load(self):
        """Load the Parakeet model onto the device."""
        if self._model is not None:
            return
        import nemo.collections.asr as nemo_asr

        logger.info("Loading Parakeet %s...", self.model_name)
        self._model = nemo_asr.models.ASRModel.from_pretrained(
            self.model_name, map_location=self.device
        )
        if self.device == "cuda":
            self._model = self._model.cuda()
        self._model.eval()
        logger.info("Parakeet %s loaded.", self.model_name)

    def unload(self):
        """Unload the model and free GPU memory."""
        if self._model is not None:
            logger.info("Unloading Parakeet...")
            self._model = None
            torch.cuda.empty_cache()

    def transcribe(self, wav_path: str) -> str:
        """Transcribe a WAV file using direct tensor inference (bypasses Lhotse).

        Returns the transcribed text string.
        """
        if self._model is None:
            raise RuntimeError("Parakeet model not loaded. Call load() first.")

        audio_data, sr = sf.read(wav_path, dtype="float32")
        audio_tensor = torch.tensor(audio_data).unsqueeze(0).cuda()
        audio_len = torch.tensor([audio_data.shape[0]]).cuda()

        with torch.no_grad():
            processed, processed_len = self._model.preprocessor(
                input_signal=audio_tensor, length=audio_len
            )
            encoded, encoded_len = self._model.encoder(
                audio_signal=processed, length=processed_len
            )
            result = self._model.decoding.rnnt_decoder_predictions_tensor(
                encoded, encoded_len, return_hypotheses=True,
            )
            hyps = result[0] if isinstance(result, tuple) else result
            if hyps and hasattr(hyps[0], "text"):
                return hyps[0].text
            return str(hyps[0]) if hyps else ""
