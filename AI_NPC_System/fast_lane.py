"""Compatibility wrapper for the current hybrid Fast Track runtime.

`server.py` and `main.py` still import `fast_lane.analyze_and_react()`.
The actual implementation now lives in `hybrid_fast_track.py` and uses
`hybrid_reactions.json`.
"""

from __future__ import annotations

import config
from hybrid_fast_track import HybridFastTrack, HybridFastTrackConfig


print("[Fast Track] Loading hybrid reaction engine...")

_ENGINE: HybridFastTrack | None = None
_ENGINE_ERROR: str | None = None


def _get_engine() -> HybridFastTrack | None:
    global _ENGINE, _ENGINE_ERROR
    if _ENGINE is not None:
        return _ENGINE
    if _ENGINE_ERROR is not None:
        return None

    try:
        _ENGINE = HybridFastTrack(
            HybridFastTrackConfig(
                reaction_path=config.REACTION_DB_PATH,
                model_name=config.EMOTION_MODEL_NAME,
                device=config.FAST_TRACK_DEVICE,
                min_runtime_score=config.FAST_TRACK_MIN_RUNTIME_SCORE,
                spacy_model=config.SPACY_MODEL_NAME,
                everyday_weight=config.FAST_TRACK_EVERYDAY_WEIGHT,
                stream_weight=config.FAST_TRACK_STREAM_WEIGHT,
            )
        )
        return _ENGINE
    except Exception as exc:
        _ENGINE_ERROR = str(exc)
        print("[Fast Track] Hybrid engine load failed; neutral fallback enabled.")
        print(f"  error: {_ENGINE_ERROR}")
        return None


def _fallback_response(text: str) -> dict:
    del text
    return {
        "emotion_label": "neutral",
        "emotion_detail": "neutral",
        "reaction": "I see.",
        "keyword": None,
        "echo_text": "",
        "strategy": "fallback",
        "reaction_source": "fallback",
        "top1": 0.0,
        "margin": 0.0,
        "entropy": 0.0,
        "confidence_band": "fallback",
        "action_probs": {},
        "strategy_scores": {},
        "calibration_temp": None,
        "effective_temperature": None,
        "category_scores": {
            "positive": 0.0,
            "negative": 0.0,
            "ambiguous": 0.0,
            "neutral": 1.0,
        },
        "bert_time": "0.0000s",
        "spacy_time": "0.0000s",
        "latency_ms": 0.0,
    }


def analyze_and_react(text: str) -> dict:
    engine = _get_engine()
    if engine is None:
        return _fallback_response(text)

    result = engine.generate(text)
    category = str(result["emotion"])
    category_key = category.lower()
    score = float(result["emotion_score"])
    margin = float(result["emotion_margin"])
    category_scores = {
        str(key).lower(): value for key, value in result.get("category_scores", {}).items()
    }

    return {
        "emotion_label": category_key,
        "emotion_detail": result["emotion_label"],
        "reaction": result["reaction"],
        "tts_text": result["tts_text"],
        "keyword": result["keyword"],
        "keywords": result["keywords"],
        "echo_text": result["tts_text"] if result["keyword"] else "",
        "strategy": f"hybrid_{result['reaction_source']}",
        "reaction_source": result["reaction_source"],
        "top1": round(score, 4),
        "margin": round(margin, 4),
        "entropy": 0.0,
        "confidence_band": "top1",
        "action_probs": {
            "everyday": config.FAST_TRACK_EVERYDAY_WEIGHT,
            "stream": config.FAST_TRACK_STREAM_WEIGHT,
        },
        "strategy_scores": {},
        "calibration_temp": None,
        "effective_temperature": None,
        "category_scores": category_scores,
        "bert_time": f"{result['emotion_ms'] / 1000.0:.4f}s",
        "spacy_time": f"{result['keyword_ms'] / 1000.0:.4f}s",
        "latency_ms": result["latency_ms"],
    }


if __name__ == "__main__":
    print(analyze_and_react("I passed the exam! I am so happy!"))
