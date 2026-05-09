"""Runtime configuration for the AI NPC Fast/Slow Track prototype."""

from __future__ import annotations

import os
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent

# Cloud fallback is intentionally disabled by default.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or None
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# Fast Track: DistilBERT + spaCy + prebuilt hybrid reaction list.
EMOTION_MODEL_NAME = os.getenv(
    "EMOTION_MODEL_NAME",
    "joeddav/distilbert-base-uncased-go-emotions-student",
)
SPACY_MODEL_NAME = os.getenv("SPACY_MODEL_NAME", "en_core_web_sm")
REACTION_DB_FILE = os.getenv("REACTION_DB_FILE", "hybrid_reactions.json")
REACTION_DB_PATH = ROOT_DIR / REACTION_DB_FILE
FAST_TRACK_DEVICE = os.getenv("FAST_TRACK_DEVICE", "auto")
FAST_TRACK_MIN_RUNTIME_SCORE = float(os.getenv("FAST_TRACK_MIN_RUNTIME_SCORE", "0.0"))
FAST_TRACK_EVERYDAY_WEIGHT = float(os.getenv("FAST_TRACK_EVERYDAY_WEIGHT", "0.60"))
FAST_TRACK_STREAM_WEIGHT = float(os.getenv("FAST_TRACK_STREAM_WEIGHT", "0.40"))

# Slow Track: quality local LLM first, smaller local model second.
LOCAL_LLM_BASE_URL = os.getenv("LOCAL_LLM_BASE_URL", "http://127.0.0.1:8002/v1")
LOCAL_LLM_MODEL = os.getenv("LOCAL_LLM_MODEL", "llama3.3:70b-awq")
LOCAL_LLM_API_KEY = os.getenv("LOCAL_LLM_API_KEY", "EMPTY")
LOCAL_LLM_TIMEOUT = float(os.getenv("LOCAL_LLM_TIMEOUT", "45.0"))
LOCAL_LLM_TEMPERATURE = float(os.getenv("LOCAL_LLM_TEMPERATURE", "0.7"))
LOCAL_LLM_MAX_TOKENS = int(os.getenv("LOCAL_LLM_MAX_TOKENS", "180"))

FALLBACK_LOCAL_LLM_BASE_URL = os.getenv(
    "FALLBACK_LOCAL_LLM_BASE_URL",
    "http://127.0.0.1:8001/v1",
)
FALLBACK_LOCAL_LLM_MODEL = os.getenv("FALLBACK_LOCAL_LLM_MODEL", "qwen2.5:7b")
FALLBACK_LOCAL_LLM_API_KEY = os.getenv("FALLBACK_LOCAL_LLM_API_KEY", "EMPTY")
FALLBACK_LOCAL_LLM_TIMEOUT = float(os.getenv("FALLBACK_LOCAL_LLM_TIMEOUT", "20.0"))

# Backward-compatible aliases for older scripts.
OLLAMA_URL = FALLBACK_LOCAL_LLM_BASE_URL
OLLAMA_MODEL = FALLBACK_LOCAL_LLM_MODEL

# Slow lane ETA hint for clients/logging.
EXPECTED_SLOW_LANE_MS = int(os.getenv("EXPECTED_SLOW_LANE_MS", "3500"))
