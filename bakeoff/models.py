"""Model registry and prompt strategies for the STT bakeoff."""


# ── Prompt strategies ────────────────────────────────────────────────────────

PROMPTS = {
    "none": {
        "name": "No Prompt",
        "description": "Baseline -- no initial prompt given",
        "prompt": None,
    },
    "generic": {
        "name": "Generic Structural",
        "description": "Describes the structure without specific names",
        "prompt": (
            "[Publisher] presents [Book Title], book [number] of the [Series Name] "
            "series, written by [Author Name], performed by [Narrator Name]. Chapter one."
        ),
    },
    "detailed": {
        "name": "Detailed Structural",
        "description": "More detailed pattern description with multiple format examples",
        "prompt": (
            "This is an audiobook introduction. The narrator states the publishing company, "
            "then the book title, then the series name and book number, then the author, "
            "and finally the narrator or performer. "
            "[Publisher] presents [Title], book [N] of the [Series], "
            "written by [Author], narrated by [Narrator]. Chapter one."
        ),
    },
}


def build_metadata_prompt(book: dict) -> str:
    """Build a prompt using existing metadata as hints (what the plugin would do)."""
    parts = []
    if book.get("publisher"):
        parts.append(f"{book['publisher']} presents")
    parts.append(f"{book['title']},")
    if book.get("series"):
        s = book["series"][0]
        seq = s.get("sequence", "")
        parts.append(f"book {seq} of the {s['name']} series,")
    if book.get("authors"):
        parts.append(f"written by {', '.join(book['authors'])},")
    if book.get("narrators"):
        parts.append(f"performed by {', '.join(book['narrators'])}.")
    parts.append("Chapter one.")
    return " ".join(parts)


# ── Model registry ───────────────────────────────────────────────────────────

MODELS = {
    # ── faster-whisper -- multilingual ───────────────────────────────────────
    "fw-large-v3-turbo": {"name": "FW large-v3-turbo", "runner": "faster_whisper", "model_id": "large-v3-turbo", "size": "~1.6GB", "vram": "~4GB", "gpu": True, "cpu": True},
    "fw-large-v3":       {"name": "FW large-v3",       "runner": "faster_whisper", "model_id": "large-v3",       "size": "~3GB",   "vram": "~5GB", "gpu": True, "cpu": True},
    "fw-large-v2":       {"name": "FW large-v2",       "runner": "faster_whisper", "model_id": "large-v2",       "size": "~3GB",   "vram": "~5GB", "gpu": True, "cpu": True},
    "fw-medium":         {"name": "FW medium",         "runner": "faster_whisper", "model_id": "medium",         "size": "~1.5GB", "vram": "~3GB", "gpu": True, "cpu": True},
    "fw-small":          {"name": "FW small",          "runner": "faster_whisper", "model_id": "small",          "size": "~460MB", "vram": "~2GB", "gpu": True, "cpu": True},
    "fw-base":           {"name": "FW base",           "runner": "faster_whisper", "model_id": "base",           "size": "~140MB", "vram": "~1GB", "gpu": True, "cpu": True},
    "fw-tiny":           {"name": "FW tiny",           "runner": "faster_whisper", "model_id": "tiny",           "size": "~75MB",  "vram": "~1GB", "gpu": True, "cpu": True},
    # ── faster-whisper -- English-optimized ──────────────────────────────────
    "fw-medium-en":      {"name": "FW medium.en",      "runner": "faster_whisper", "model_id": "medium.en",      "size": "~1.5GB", "vram": "~3GB", "gpu": True, "cpu": True},
    "fw-small-en":       {"name": "FW small.en",       "runner": "faster_whisper", "model_id": "small.en",       "size": "~460MB", "vram": "~2GB", "gpu": True, "cpu": True},
    "fw-base-en":        {"name": "FW base.en",        "runner": "faster_whisper", "model_id": "base.en",        "size": "~140MB", "vram": "~1GB", "gpu": True, "cpu": True},
    "fw-tiny-en":        {"name": "FW tiny.en",        "runner": "faster_whisper", "model_id": "tiny.en",        "size": "~75MB",  "vram": "~1GB", "gpu": True, "cpu": True},
    # ── faster-whisper -- distilled ──────────────────────────────────────────
    "fw-distil-large-v3":  {"name": "FW distil-large-v3",  "runner": "faster_whisper", "model_id": "distil-large-v3",  "size": "~1.5GB", "vram": "~4GB",   "gpu": True, "cpu": True},
    "fw-distil-large-v2":  {"name": "FW distil-large-v2",  "runner": "faster_whisper", "model_id": "distil-large-v2",  "size": "~1.5GB", "vram": "~4GB",   "gpu": True, "cpu": True},
    "fw-distil-medium-en": {"name": "FW distil-medium.en", "runner": "faster_whisper", "model_id": "distil-medium.en", "size": "~750MB", "vram": "~2GB",   "gpu": True, "cpu": True},
    "fw-distil-small-en":  {"name": "FW distil-small.en",  "runner": "faster_whisper", "model_id": "distil-small.en",  "size": "~330MB", "vram": "~1.5GB", "gpu": True, "cpu": True},
    # ── HuggingFace -- framework comparison ──────────────────────────────────
    "hf-large-v3-turbo": {"name": "HF large-v3-turbo", "runner": "hf_whisper", "model_id": "openai/whisper-large-v3-turbo", "size": "~1.6GB", "vram": "~4GB", "gpu": True, "cpu": True},
    # ── Moonshine (UsefulSensors) ────────────────────────────────────────────
    "moonshine-base": {"name": "Moonshine base", "runner": "moonshine", "model_id": "UsefulSensors/moonshine-base", "size": "~330MB", "vram": "~1.5GB", "gpu": True, "cpu": True},
    "moonshine-tiny": {"name": "Moonshine tiny", "runner": "moonshine", "model_id": "UsefulSensors/moonshine-tiny", "size": "~75MB",  "vram": "~1GB",   "gpu": True, "cpu": True},
    # ── NVIDIA NeMo ──────────────────────────────────────────────────────────
    "nemo-parakeet-v2":     {"name": "NeMo Parakeet v2",     "runner": "nemo", "model_id": "nvidia/parakeet-tdt-0.6b-v2",          "size": "~1.2GB", "vram": "~3GB", "gpu": True, "cpu": True},
    "nemo-parakeet-110m":   {"name": "NeMo Parakeet 110M",   "runner": "nemo", "model_id": "parakeet-tdt_ctc-110m",                "size": "~440MB", "vram": "~2GB", "gpu": True, "cpu": True},
    "nemo-fastconf-hybrid": {"name": "NeMo FastConf Hybrid", "runner": "nemo", "model_id": "stt_en_fastconformer_hybrid_large_pc", "size": "~460MB", "vram": "~2GB", "gpu": True, "cpu": True},
    # ── SeamlessM4T v2 (Meta) ────────────────────────────────────────────────
    "seamless-m4t-v2-large": {"name": "SeamlessM4T v2 Large", "runner": "seamless", "model_id": "facebook/seamless-m4t-v2-large", "size": "~9GB", "vram": "~6GB", "gpu": True, "cpu": True},
}
