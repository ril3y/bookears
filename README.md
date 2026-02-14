# Bookears

Listens to audiobook intros and verifies the audio matches its metadata. Runs dual-engine transcription (Whisper + Parakeet) on GPU and returns structured results via HTTP.

The "ears" for the Audiobookshelf AIMetadataCleaner plugin -- the plugin handles the "brain" (lexicon, parser, resolver) in Node.js, Bookears handles the listening.

## Quick Start

### Bare Metal

```bash
pip install -r requirements.txt
python server.py
```

### Docker

```bash
docker-compose up --build
```

## API

### `GET /health`

```json
{
  "status": "ok",
  "models": ["whisper-large-v3-turbo", "parakeet-tdt-0.6b-v2"],
  "gpu": {"name": "NVIDIA GeForce RTX 3060 Ti", "vram_total_gb": 8.0, "vram_used_gb": 5.5},
  "whisper_loaded": true,
  "parakeet_loaded": true,
  "version": "1.0.0"
}
```

### `POST /transcribe`

Multipart form-data with `file` field containing a WAV audio file.

```bash
curl -X POST http://localhost:8200/transcribe -F "file=@/tmp/test.wav" --max-time 120
```

Response:

```json
{
  "whisper": "This is Audible. Recorded Books presents Along Came a Spider by James Patterson...",
  "parakeet": "this is audible recorded books presents along came a spider by james patterson...",
  "language": "en",
  "language_probability": 0.98,
  "first_word_time": 1.24,
  "whisper_time": 3.21,
  "parakeet_time": 1.05
}
```

## Configuration

All settings via environment variables (see `.env.example`):

| Variable | Default | Description |
|---|---|---|
| `PORT` | `8200` | Server port |
| `HOST` | `0.0.0.0` | Bind address |
| `DEVICE` | `cuda` | `cuda` or `cpu` |
| `MODEL_WHISPER` | `large-v3-turbo` | Whisper model size |
| `MODEL_PARAKEET` | `nvidia/parakeet-tdt-0.6b-v2` | Parakeet model name |
| `COMPUTE_TYPE` | `float16` | Whisper compute type |
| `MODEL_CACHE_DIR` | `~/.cache/whisper` | Model download cache |
| `LOG_LEVEL` | `info` | Logging level |

## Bakeoff Runner

Multi-model comparison testing tool:

```bash
python -m bakeoff.runner --models fw-large-v3-turbo,nemo-parakeet-v2 --num-books 5 --serve 8099
```

Then open `http://localhost:8099/bakeoff_ui.html?file=/results.json`

### List available models

```bash
python -m bakeoff.runner --list-models
```

## Architecture

- **`server.py`** - FastAPI server with `/health` and `/transcribe` endpoints
- **`asr/whisper.py`** - WhisperEngine class (faster-whisper, CUDA)
- **`asr/parakeet.py`** - ParakeetEngine class (NeMo TDT, CUDA)
- **`asr/audio.py`** - Audio file utilities (temp files, ffmpeg extraction)
- **`asr/detection.py`** - Content detection heuristics (music intro, dramatic intro, aged media, non-English)
- **`bakeoff/`** - Model comparison testing framework with dark-themed dashboard UI

## GPU Memory

Both models fit on an 8GB GPU (~5.5GB VRAM). If OOM occurs during dual-load, the service falls back to sequential mode (swapping models per request, ~30s overhead).
