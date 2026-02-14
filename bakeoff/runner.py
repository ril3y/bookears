#!/usr/bin/env python3
"""
Bakeoff test runner — multi-model comparison for audiobook STT.

Usage:
  python -m bakeoff.runner --models fw-large-v3-turbo,nemo-parakeet-v2 --num-books 5 --serve 8099
  python -m bakeoff.runner --list-models
"""

import argparse
import http.server
import json
import os
import re
import sys
import threading
import time
from difflib import SequenceMatcher
from pathlib import Path

import torch

from .models import MODELS, PROMPTS, build_metadata_prompt

# ── Configuration ────────────────────────────────────────────────────────────

ABS_URL = os.environ.get("ABS_URL", "http://localhost:3333")
ABS_TOKEN = os.environ.get("ABS_TOKEN", "")
MODEL_CACHE_DIR = os.environ.get("MODEL_CACHE_DIR", "/mnt/llm-models/stt-models")


# ── ABS API ──────────────────────────────────────────────────────────────────

def abs_api(path):
    import urllib.request
    req = urllib.request.Request(
        f"{ABS_URL}/api{path}",
        headers={"Authorization": f"Bearer {ABS_TOKEN}"},
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def get_series_books(library_id, num_books=12):
    """Pull books that are part of a series, with audio files and diverse authors."""
    data = abs_api(f"/libraries/{library_id}/series?limit=40&sort=numBooks&desc=1")
    candidates = []
    seen_series = set()

    for s in data.get("results", []):
        if len(candidates) >= num_books:
            break
        series_name = s["name"]
        if series_name in seen_series:
            continue
        for book_item in s.get("books", [])[:2]:
            item_id = book_item.get("id")
            if not item_id:
                continue
            try:
                detail = abs_api(f"/items/{item_id}?expanded=1")
                media = detail.get("media", {})
                meta = media.get("metadata", {})
                audio_files = media.get("audioFiles", [])
                authors = [a["name"] for a in meta.get("authors", [])]
                series = meta.get("series", [])
                if not audio_files or not authors or not series:
                    continue
                first_path = audio_files[0].get("metadata", {}).get("path", "")
                if not first_path:
                    continue
                seen_series.add(series_name)
                candidates.append({
                    "id": item_id,
                    "title": meta.get("title", ""),
                    "authors": authors,
                    "narrators": meta.get("narrators", []),
                    "series": [{"name": ss.get("name"), "sequence": ss.get("sequence")} for ss in series],
                    "publisher": meta.get("publisher", ""),
                    "audio_path": first_path,
                })
                break
            except Exception:
                continue
    return candidates


# ── Audio extraction ─────────────────────────────────────────────────────────

def extract_audio_sample(audio_path, duration=30):
    """Extract first N seconds as 16kHz mono WAV using ffmpeg."""
    from asr.audio import extract_audio_sample as _extract
    return _extract(audio_path, duration)


# ── Model runners ────────────────────────────────────────────────────────────

_fw_model_cache = {}
_moonshine_cache = {}
_seamless_cache = {}
_nemo_cache = {}


def run_faster_whisper(audio_path, model_size, device="cuda", compute_type=None, initial_prompt=None):
    from faster_whisper import WhisperModel
    if compute_type is None:
        compute_type = "float16" if device == "cuda" else "int8"
    cache_key = f"{model_size}_{device}_{compute_type}"
    if cache_key not in _fw_model_cache:
        _fw_model_cache[cache_key] = WhisperModel(
            model_size, device=device, compute_type=compute_type,
            download_root=MODEL_CACHE_DIR,
        )
    model = _fw_model_cache[cache_key]
    kwargs = {"beam_size": 5, "vad_filter": True}
    if initial_prompt:
        kwargs["initial_prompt"] = initial_prompt
    segments, info = model.transcribe(audio_path, **kwargs)
    text = " ".join(seg.text.strip() for seg in segments)
    return text


def run_hf_whisper(audio_path, model_id, device="cuda", initial_prompt=None):
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    processor = AutoProcessor.from_pretrained(model_id, cache_dir=MODEL_CACHE_DIR)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True,
        cache_dir=MODEL_CACHE_DIR,
    ).to(device)
    pipe = pipeline(
        "automatic-speech-recognition", model=model,
        tokenizer=processor.tokenizer, feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype, device=device,
    )
    result = pipe(audio_path, chunk_length_s=30, stride_length_s=5)
    return result["text"]


def run_moonshine(audio_path, model_id, device="cuda", initial_prompt=None):
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
    cache_key = f"{model_id}_{device}"
    if cache_key not in _moonshine_cache:
        torch_dtype = torch.float16 if device == "cuda" else torch.float32
        processor = AutoProcessor.from_pretrained(model_id, cache_dir=MODEL_CACHE_DIR)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True,
            cache_dir=MODEL_CACHE_DIR,
        ).to(device)
        pipe = pipeline(
            "automatic-speech-recognition", model=model,
            tokenizer=processor.tokenizer, feature_extractor=processor.feature_extractor,
            torch_dtype=torch_dtype, device=device,
        )
        _moonshine_cache[cache_key] = pipe
    result = _moonshine_cache[cache_key](audio_path, chunk_length_s=30, stride_length_s=5)
    return result["text"]


def run_seamless(audio_path, model_id, device="cuda", initial_prompt=None):
    from transformers import AutoProcessor, SeamlessM4Tv2ForSpeechToText, pipeline
    cache_key = f"{model_id}_{device}"
    if cache_key not in _seamless_cache:
        torch_dtype = torch.float16 if device == "cuda" else torch.float32
        processor = AutoProcessor.from_pretrained(model_id, cache_dir=MODEL_CACHE_DIR)
        model = SeamlessM4Tv2ForSpeechToText.from_pretrained(
            model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True,
            cache_dir=MODEL_CACHE_DIR,
        ).to(device)
        pipe = pipeline(
            "automatic-speech-recognition", model=model,
            tokenizer=processor.tokenizer, feature_extractor=processor.feature_extractor,
            torch_dtype=torch_dtype, device=device,
        )
        _seamless_cache[cache_key] = pipe
    result = _seamless_cache[cache_key](audio_path, chunk_length_s=30, stride_length_s=5)
    return result["text"]


def run_nemo(audio_path, model_id, device="cuda", initial_prompt=None):
    import nemo.collections.asr as nemo_asr
    cache_key = f"{model_id}_{device}"
    if cache_key not in _nemo_cache:
        model = nemo_asr.models.ASRModel.from_pretrained(
            model_name=model_id, map_location=device,
        )
        if device == "cuda":
            model = model.cuda()
        model.eval()
        _nemo_cache[cache_key] = model
    model = _nemo_cache[cache_key]
    transcriptions = model.transcribe([audio_path])
    if transcriptions and isinstance(transcriptions[0], str):
        return transcriptions[0]
    elif hasattr(transcriptions[0], "text"):
        return transcriptions[0].text
    return str(transcriptions[0])


def clear_model_caches():
    _fw_model_cache.clear()
    _moonshine_cache.clear()
    _seamless_cache.clear()
    _nemo_cache.clear()
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass


def check_framework(runner):
    try:
        if runner == "faster_whisper":
            import faster_whisper  # noqa: F401
        elif runner in ("hf_whisper", "moonshine", "seamless"):
            import transformers  # noqa: F401
        elif runner == "nemo":
            import nemo.collections.asr  # noqa: F401
        return True
    except ImportError:
        return False


# ── Scoring ──────────────────────────────────────────────────────────────────

def normalize(text):
    if not text:
        return ""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def find_in_text(text, target, threshold=0.6):
    if not text or not target:
        return 0.0
    norm_text = normalize(text)
    norm_target = normalize(target)
    if norm_target in norm_text:
        return 1.0
    words = norm_text.split()
    target_words = len(norm_target.split())
    best = 0.0
    for window_size in range(max(1, target_words - 1), target_words + 3):
        for i in range(len(words) - window_size + 1):
            chunk = " ".join(words[i : i + window_size])
            score = SequenceMatcher(None, chunk, norm_target).ratio()
            if score > best:
                best = score
    return best if best >= threshold else 0.0


def score_transcription(transcription, ground_truth):
    scores = {}
    scores["title"] = round(find_in_text(transcription, ground_truth["title"]) * 100)

    if ground_truth["authors"]:
        s = [find_in_text(transcription, a) for a in ground_truth["authors"]]
        scores["authors"] = round(sum(s) / len(s) * 100)
    else:
        scores["authors"] = 0

    if ground_truth["narrators"]:
        s = [find_in_text(transcription, n) for n in ground_truth["narrators"]]
        scores["narrators"] = round(sum(s) / len(s) * 100)
    else:
        scores["narrators"] = None

    if ground_truth["series"]:
        s = [find_in_text(transcription, ser["name"]) for ser in ground_truth["series"]]
        scores["series"] = round(sum(s) / len(s) * 100)
    else:
        scores["series"] = None

    if ground_truth["publisher"]:
        scores["publisher"] = round(find_in_text(transcription, ground_truth["publisher"]) * 100)
    else:
        scores["publisher"] = None

    weights = {"title": 3, "authors": 3, "narrators": 2, "series": 2, "publisher": 1}
    tw = ts = 0
    for f, w in weights.items():
        if scores.get(f) is not None:
            tw += w
            ts += scores[f] * w
    scores["overall"] = round(ts / tw) if tw > 0 else 0
    return scores


# ── Runner ───────────────────────────────────────────────────────────────────

RUNNER_MAP = {
    "faster_whisper": run_faster_whisper,
    "hf_whisper": run_hf_whisper,
    "moonshine": run_moonshine,
    "seamless": run_seamless,
    "nemo": run_nemo,
}


def save_live(results, path):
    try:
        with open(path, "w") as f:
            json.dump(results, f, indent=2, default=str)
    except Exception:
        pass


def run_bakeoff(books, wav_paths, duration, model_keys, prompt_keys, device, output_path=None):
    results = {
        "meta": {
            "duration": duration,
            "device": device,
            "models": model_keys,
            "prompts": prompt_keys,
            "num_books": len(books),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "books": books,
        "runs": [],
    }

    total_runs = len(model_keys) * len(prompt_keys) * len(books)
    run_num = 0

    for model_key in model_keys:
        model_info = MODELS[model_key]
        model_name = model_info["name"]

        if not check_framework(model_info["runner"]):
            print(f"\n  SKIPPED model {model_name}: {model_info['runner']} not installed")
            continue

        print(f"\n{'#' * 90}")
        print(f"  MODEL: {model_name} ({model_info['model_id']})")
        print(f"{'#' * 90}")

        runner_fn = RUNNER_MAP.get(model_info["runner"])
        if not runner_fn:
            print(f"  Unknown runner: {model_info['runner']}")
            continue

        for prompt_key in prompt_keys:
            prompt_info = PROMPTS.get(prompt_key)
            if prompt_key == "metadata":
                prompt_label = "Metadata-Aware"
                prompt_desc = "Uses existing ABS metadata as hints"
            else:
                prompt_label = prompt_info["name"]
                prompt_desc = prompt_info["description"]

            print(f"\n  -- Prompt: {prompt_label} --")
            print(f"     {prompt_desc}")

            for book_idx, book in enumerate(books):
                run_num += 1
                wav_path = wav_paths.get(book_idx)
                if not wav_path:
                    continue

                if prompt_key == "metadata":
                    prompt_text = build_metadata_prompt(book)
                elif prompt_info and prompt_info["prompt"]:
                    prompt_text = prompt_info["prompt"]
                else:
                    prompt_text = None

                run_label = f"{model_name} | {prompt_label}"
                series_str = ", ".join(
                    f"{s['name']} #{s['sequence']}" for s in book["series"]
                ) if book["series"] else "-"
                print(f"\n  [{run_num}/{total_runs}] {book['title'][:35]} ({series_str})")

                try:
                    start = time.time()

                    if model_info["runner"] == "faster_whisper":
                        text = runner_fn(wav_path, model_info["model_id"], device, initial_prompt=prompt_text)
                    elif model_info["runner"] in ("hf_whisper", "moonshine", "seamless", "nemo"):
                        text = runner_fn(wav_path, model_info["model_id"], device, initial_prompt=prompt_text)
                    else:
                        continue

                    elapsed = time.time() - start
                    scores = score_transcription(text, book)

                    print(
                        f"    {elapsed:.1f}s | overall={scores['overall']} "
                        f"title={scores['title']} author={scores['authors']} "
                        f"narrator={scores.get('narrators', '-')} series={scores.get('series', '-')} "
                        f"pub={scores.get('publisher', '-')}"
                    )
                    print(f"    {text[:200]}{'...' if len(text) > 200 else ''}")

                    results["runs"].append({
                        "model": model_key,
                        "model_name": model_name,
                        "prompt": prompt_key,
                        "prompt_name": prompt_label,
                        "prompt_text": prompt_text,
                        "book_idx": book_idx,
                        "transcription": text,
                        "scores": scores,
                        "time_seconds": round(elapsed, 1),
                    })

                except Exception as e:
                    print(f"    ERROR: {e}")
                    import traceback
                    traceback.print_exc()
                    results["runs"].append({
                        "model": model_key,
                        "model_name": model_name,
                        "prompt": prompt_key,
                        "prompt_name": prompt_label,
                        "book_idx": book_idx,
                        "error": str(e),
                    })

            if output_path:
                save_live(results, output_path)

        clear_model_caches()

    return results


# ── Web server ───────────────────────────────────────────────────────────────

def start_result_server(port, results_path, ui_dir):
    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=str(ui_dir), **kw)

        def do_GET(self):
            if self.path.startswith("/results.json"):
                try:
                    with open(results_path, "rb") as f:
                        content = f.read()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.send_header("Cache-Control", "no-cache")
                    self.end_headers()
                    self.wfile.write(content)
                except FileNotFoundError:
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(b'{"meta":{},"books":[],"runs":[]}')
                return
            return super().do_GET()

        def log_message(self, fmt, *a):
            pass

    server = http.server.HTTPServer(("0.0.0.0", port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="STT Bake-Off Runner")
    parser.add_argument("--num-books", type=int, default=12, help="Books to sample (default: 12)")
    parser.add_argument("--duration", type=int, default=30, help="Seconds of audio (default: 30)")
    parser.add_argument("--models", type=str, default=None, help="Comma-separated model keys")
    parser.add_argument("--prompts", type=str, default=None, help="Comma-separated prompt keys")
    parser.add_argument("--gpu-only", action="store_true")
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--list-models", action="store_true")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--books-file", type=str, default=None)
    parser.add_argument("--serve", type=int, default=None, nargs="?", const=8099)
    parser.add_argument("--abs-url", type=str, default=None)
    parser.add_argument("--abs-token", type=str, default=None)
    args = parser.parse_args()

    global ABS_URL, ABS_TOKEN
    if args.abs_url:
        ABS_URL = args.abs_url
    if args.abs_token:
        ABS_TOKEN = args.abs_token

    if args.list_models:
        print(f"{'Key':<25} {'Name':<25} {'Size':<10} {'VRAM':<10} {'Runner'}")
        print(f"{'-' * 25} {'-' * 25} {'-' * 10} {'-' * 10} {'-' * 15}")
        for key, m in MODELS.items():
            print(f"{key:<25} {m['name']:<25} {m['size']:<10} {m.get('vram', '?'):<10} {m['runner']}")
        return

    # GPU check
    has_gpu = torch.cuda.is_available()
    if has_gpu:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"GPU: {gpu_name} ({gpu_mem:.1f}GB)")
    else:
        print("GPU: No CUDA GPU detected")

    if args.gpu_only:
        device = "cuda" if has_gpu else sys.exit("No GPU")
    elif args.cpu_only:
        device = "cpu"
    else:
        device = "cuda" if has_gpu else "cpu"
    print(f"Device: {device}")

    # Models
    if args.models:
        model_keys = [k.strip() for k in args.models.split(",")]
        invalid = [k for k in model_keys if k not in MODELS]
        if invalid:
            sys.exit(f"Unknown models: {', '.join(invalid)}\nAvailable: {', '.join(MODELS.keys())}")
    else:
        model_keys = [
            "fw-large-v3-turbo", "fw-large-v3", "fw-large-v2",
            "fw-medium", "fw-medium-en", "fw-small", "fw-small-en",
            "nemo-parakeet-v2",
        ]

    # Prompts
    if args.prompts:
        prompt_keys = [k.strip() for k in args.prompts.split(",")]
    else:
        prompt_keys = ["none", "generic", "detailed", "metadata"]
    print(f"Prompts: {', '.join(prompt_keys)}")
    print(f"Models: {', '.join(model_keys)}")

    # Books
    if args.books_file:
        print(f"Loading books from {args.books_file}")
        with open(args.books_file) as f:
            books = json.load(f)
    else:
        print(f"\nFetching series books from Audiobookshelf ({ABS_URL})...")
        libs = abs_api("/libraries")
        book_libs = [l for l in libs["libraries"] if l["mediaType"] == "book"]
        if not book_libs:
            sys.exit("No book libraries found")
        books = get_series_books(book_libs[0]["id"], args.num_books)

    print(f"Books: {len(books)}")
    if not books:
        sys.exit("No suitable books found")

    # Extract audio samples
    print(f"\nExtracting {args.duration}s audio samples...")
    wav_paths = {}
    for i, book in enumerate(books):
        series_str = ", ".join(
            f"{s['name']} #{s['sequence']}" for s in book["series"]
        ) if book["series"] else "-"
        print(f"  {i + 1:>3}. {book['title'][:40]:<40} | {', '.join(book['authors'])[:20]:<20} | {series_str}")
        if os.path.isfile(book["audio_path"]):
            try:
                wav_paths[i] = extract_audio_sample(book["audio_path"], args.duration)
            except Exception as e:
                print(f"       ERROR: {e}")
        else:
            print(f"       MISSING: {book['audio_path']}")

    total_runs = len(model_keys) * len(prompt_keys) * len(wav_paths)
    print(f"\nTotal runs: {total_runs} ({len(model_keys)} models x {len(prompt_keys)} prompts x {len(wav_paths)} books)")

    output_path = args.output or f"/tmp/stt_bakeoff_{int(time.time())}.json"

    # Web UI
    if args.serve:
        ui_dir = Path(__file__).parent / "ui"
        start_result_server(args.serve, output_path, ui_dir)
        print(f"\n  Web UI: http://0.0.0.0:{args.serve}/bakeoff_ui.html?file=/results.json")

    # Run
    results = run_bakeoff(books, wav_paths, args.duration, model_keys, prompt_keys, device, output_path)

    # Cleanup WAVs
    for wp in wav_paths.values():
        try:
            os.unlink(wp)
        except OSError:
            pass

    # Summary
    runs = [r for r in results["runs"] if "scores" in r]
    if runs:
        avg_overall = sum(r["scores"]["overall"] for r in runs) / len(runs)
        print(f"\n{'=' * 80}")
        print(f"  {len(runs)} runs completed, avg overall: {avg_overall:.1f}")

    # Final save
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")

    if args.serve:
        print(f"Web UI: http://0.0.0.0:{args.serve}/bakeoff_ui.html?file=/results.json")
        print("Press Ctrl+C to stop.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nDone.")


if __name__ == "__main__":
    main()
