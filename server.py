#!/usr/bin/env python3
"""
Bookears — FastAPI server providing dual-engine audiobook transcription.

Endpoints:
  GET  /health      — Model status + GPU info
  POST /transcribe  — Multipart WAV upload → dual Whisper+Parakeet transcription
"""

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager

import torch
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from asr.audio import cleanup_tempfile, save_upload_to_tempfile
from asr.parakeet import ParakeetEngine
from asr.whisper import WhisperEngine

# ── Configuration ────────────────────────────────────────────────────────────

PORT = int(os.environ.get("PORT", 8200))
HOST = os.environ.get("HOST", "0.0.0.0")
DEVICE = os.environ.get("DEVICE", "cuda")
MODEL_WHISPER = os.environ.get("MODEL_WHISPER", "large-v3-turbo")
MODEL_PARAKEET = os.environ.get("MODEL_PARAKEET", "nvidia/parakeet-tdt-0.6b-v2")
COMPUTE_TYPE = os.environ.get("COMPUTE_TYPE", "float16")
MODEL_CACHE_DIR = os.environ.get("MODEL_CACHE_DIR", os.path.expanduser("~/.cache/whisper"))
LOG_LEVEL = os.environ.get("LOG_LEVEL", "info").upper()

VERSION = "1.0.0"

# ── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("bookears")

# ── Engine instances ─────────────────────────────────────────────────────────

whisper_engine = WhisperEngine(
    model_size=MODEL_WHISPER,
    device=DEVICE,
    compute_type=COMPUTE_TYPE,
    cache_dir=MODEL_CACHE_DIR,
)
parakeet_engine = ParakeetEngine(
    model_name=MODEL_PARAKEET,
    device=DEVICE,
)

# Track loading mode
dual_loaded = False
sequential_mode = False


# ── Lifespan ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup, clean up on shutdown."""
    global dual_loaded, sequential_mode

    logger.info("Starting Bookears v%s", VERSION)

    # ── GPU requirement check ──
    if DEVICE == "cuda" and not torch.cuda.is_available():
        logger.fatal(
            "CUDA GPU required but not available. "
            "Run with --gpus or deploy.resources.reservations.devices in docker-compose. "
            "Set DEVICE=cpu to force CPU mode (very slow, not recommended)."
        )
        raise SystemExit(1)

    if DEVICE == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logger.info("GPU: %s (%.1f GB VRAM)", gpu_name, vram_gb)
        if vram_gb < 6.0:
            logger.warning(
                "GPU has %.1f GB VRAM — dual-loading needs ~5.5 GB. "
                "Sequential fallback mode likely.", vram_gb
            )

    logger.info("Whisper: %s | Parakeet: %s | Device: %s", MODEL_WHISPER, MODEL_PARAKEET, DEVICE)

    # Try dual-loading both models
    try:
        whisper_engine.load()
        parakeet_engine.load()
        dual_loaded = True
        logger.info("Dual-loaded: Whisper + Parakeet on GPU")
    except (RuntimeError, Exception) as e:
        if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
            logger.warning("OOM dual-loading, falling back to sequential mode: %s", e)
            parakeet_engine.unload()
            torch.cuda.empty_cache()
            # Ensure Whisper is loaded for the first request
            if not whisper_engine.loaded:
                whisper_engine.load()
            sequential_mode = True
            dual_loaded = False
        else:
            raise

    yield

    # Shutdown
    logger.info("Shutting down Bookears...")
    whisper_engine.unload()
    parakeet_engine.unload()


# ── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(title="Bookears", version=VERSION, lifespan=lifespan)


@app.get("/health")
async def health():
    """Return model status and GPU info."""
    gpu_info = {}
    if torch.cuda.is_available():
        gpu_info = {
            "name": torch.cuda.get_device_name(0),
            "vram_total_gb": round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 1),
            "vram_used_gb": round(torch.cuda.memory_allocated(0) / (1024**3), 1),
        }

    models = []
    if whisper_engine.loaded:
        models.append(f"whisper-{MODEL_WHISPER}")
    if parakeet_engine.loaded:
        models.append(f"parakeet-{MODEL_PARAKEET.split('/')[-1]}")

    return {
        "status": "ok",
        "models": models,
        "gpu": gpu_info,
        "whisper_loaded": whisper_engine.loaded,
        "parakeet_loaded": parakeet_engine.loaded,
        "sequential_mode": sequential_mode,
        "version": VERSION,
    }


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    """Transcribe a WAV file using both Whisper and Parakeet.

    Accepts multipart/form-data with a `file` field containing a WAV audio file.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    wav_path = await save_upload_to_tempfile(file)
    try:
        loop = asyncio.get_event_loop()

        if sequential_mode:
            # Sequential: swap models per request
            result = await loop.run_in_executor(None, _transcribe_sequential, wav_path)
        else:
            # Dual-loaded: run both in sequence (GPU can't truly parallelize)
            result = await loop.run_in_executor(None, _transcribe_dual, wav_path)

        return JSONResponse(content=result)

    except Exception as e:
        logger.error("Transcription failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cleanup_tempfile(wav_path)


def _transcribe_dual(wav_path: str) -> dict:
    """Transcribe with both models dual-loaded on GPU."""
    t0 = time.time()
    whisper_result = whisper_engine.transcribe(wav_path)
    t1 = time.time()
    parakeet_text = parakeet_engine.transcribe(wav_path)
    t2 = time.time()

    return {
        "whisper": whisper_result["text"],
        "parakeet": parakeet_text,
        "language": whisper_result["language"],
        "language_probability": whisper_result["language_probability"],
        "first_word_time": whisper_result["first_word_time"],
        "whisper_time": round(t1 - t0, 2),
        "parakeet_time": round(t2 - t1, 2),
    }


def _transcribe_sequential(wav_path: str) -> dict:
    """Transcribe by swapping models (OOM fallback mode)."""
    # Ensure Whisper is loaded
    if not whisper_engine.loaded:
        parakeet_engine.unload()
        whisper_engine.load()

    t0 = time.time()
    whisper_result = whisper_engine.transcribe(wav_path)
    t1 = time.time()

    # Swap: unload Whisper, load Parakeet
    whisper_engine.unload()
    parakeet_engine.load()

    parakeet_text = parakeet_engine.transcribe(wav_path)
    t2 = time.time()

    # Swap back: unload Parakeet, load Whisper for next request
    parakeet_engine.unload()
    whisper_engine.load()

    return {
        "whisper": whisper_result["text"],
        "parakeet": parakeet_text,
        "language": whisper_result["language"],
        "language_probability": whisper_result["language_probability"],
        "first_word_time": whisper_result["first_word_time"],
        "whisper_time": round(t1 - t0, 2),
        "parakeet_time": round(t2 - t1, 2),
    }


if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host=HOST,
        port=PORT,
        log_level=LOG_LEVEL.lower(),
    )
