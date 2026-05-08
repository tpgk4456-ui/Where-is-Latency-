"""
Runtime hybrid Fast Track reaction generator.

Install:
  python3 -m pip install -r AI_NPC_System/reaction_pipeline/requirements_hybrid_reactions.txt
  python3 -m spacy download en_core_web_sm

Build the reaction JSON first:
  python3 AI_NPC_System/reaction_pipeline/build_hybrid_reactions.py

CLI smoke test:
  python3 AI_NPC_System/hybrid_fast_track.py --text "I passed the exam today"
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_REACTION_PATH = ROOT / "hybrid_reactions.json"
MODEL_NAME = "joeddav/distilbert-base-uncased-go-emotions-student"

LABEL_TO_CATEGORY = {
    "admiration": "Positive",
    "amusement": "Positive",
    "approval": "Positive",
    "caring": "Positive",
    "desire": "Positive",
    "excitement": "Positive",
    "gratitude": "Positive",
    "joy": "Positive",
    "love": "Positive",
    "optimism": "Positive",
    "pride": "Positive",
    "relief": "Positive",
    "anger": "Negative",
    "annoyance": "Negative",
    "disappointment": "Negative",
    "disapproval": "Negative",
    "disgust": "Negative",
    "embarrassment": "Negative",
    "fear": "Negative",
    "grief": "Negative",
    "nervousness": "Negative",
    "remorse": "Negative",
    "sadness": "Negative",
    "confusion": "Ambiguous",
    "curiosity": "Ambiguous",
    "realization": "Ambiguous",
    "surprise": "Ambiguous",
    "neutral": "Neutral",
}

FALLBACK_REACTIONS = {
    "Positive": ["Nice.", "Huge W.", "Glad to hear it."],
    "Negative": ["That's rough.", "I hear you.", "That hurts."],
    "Ambiguous": ["Wait, really?", "Interesting.", "What happened?"],
    "Neutral": ["Got it.", "I see.", "Okay."],
}


@dataclass(frozen=True)
class HybridFastTrackConfig:
    reaction_path: Path = DEFAULT_REACTION_PATH
    model_name: str = MODEL_NAME
    device: str = "auto"
    seed: int | None = None
    min_runtime_score: float = 0.0
    spacy_model: str = "en_core_web_sm"
    everyday_weight: float = 0.60
    stream_weight: float = 0.40


def require_runtime_deps() -> tuple[Any, Any, Any]:
    try:
        import spacy
        import torch
        from transformers import pipeline
    except ImportError as exc:
        print("Missing dependency. Install with:")
        print("  python3 -m pip install -r AI_NPC_System/reaction_pipeline/requirements_hybrid_reactions.txt")
        print("  python3 -m spacy download en_core_web_sm")
        raise SystemExit(2) from exc
    return spacy, torch, pipeline


def choose_device(torch: Any, requested: str) -> int:
    if requested == "cpu":
        return -1
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("CUDA was requested, but torch.cuda.is_available() is false.")
        return 0
    return 0 if torch.cuda.is_available() else -1


def load_reaction_db(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_score_list(raw: Any) -> list[dict[str, Any]]:
    if isinstance(raw, list):
        if raw and isinstance(raw[0], list):
            return raw[0] if raw[0] else [{"label": "neutral", "score": 0.0}]
        return raw if raw else [{"label": "neutral", "score": 0.0}]
    return [raw]


def aggregate_category_scores(score_items: list[dict[str, Any]]) -> dict[str, float]:
    scores = {"Positive": 0.0, "Negative": 0.0, "Ambiguous": 0.0, "Neutral": 0.0}
    for item in score_items:
        label = str(item.get("label", "neutral")).lower()
        category = LABEL_TO_CATEGORY.get(label)
        if category:
            scores[category] += float(item.get("score", 0.0))
    return scores


def top_label(score_items: list[dict[str, Any]]) -> str:
    if not score_items:
        return "neutral"
    return str(max(score_items, key=lambda item: float(item.get("score", 0.0))).get("label", "neutral")).lower()


class HybridFastTrack:
    def __init__(self, config: HybridFastTrackConfig | None = None) -> None:
        self.config = config or HybridFastTrackConfig()
        spacy, torch, pipeline = require_runtime_deps()

        self.rng = random.Random(self.config.seed)
        self.reactions = load_reaction_db(self.config.reaction_path)
        self.device = choose_device(torch, self.config.device)
        self.classifier = pipeline(
            "text-classification",
            model=self.config.model_name,
            tokenizer=self.config.model_name,
            top_k=None,
            device=self.device,
        )
        self.nlp = spacy.load(
            self.config.spacy_model,
            disable=["parser", "ner", "lemmatizer"],
        )
        self._warmup()

    def _warmup(self) -> None:
        self.classifier("warm up", truncation=True)
        self.nlp("warm up")

    def classify_emotion(self, text: str) -> dict[str, Any]:
        raw = self.classifier(text[:512], truncation=True)
        score_items = normalize_score_list(raw)
        label = top_label(score_items)
        category_scores = aggregate_category_scores(score_items)
        category = max(category_scores, key=category_scores.get)
        score = category_scores[category]
        if score < self.config.min_runtime_score:
            category = "Neutral"
        return {"category": category, "label": label, "score": score, "category_scores": category_scores}

    def extract_keywords(self, text: str) -> list[str]:
        doc = self.nlp(text)
        keywords = []
        for token in doc:
            if token.pos_ in {"NOUN", "PROPN"} and not token.is_stop:
                keywords.append(token.text)
        return keywords

    def choose_reaction(self, category: str) -> tuple[str, str]:
        bucket = self.reactions.get(category, {})
        if isinstance(bucket, dict):
            everyday = bucket.get("everyday") or []
            stream = bucket.get("stream") or []
            if everyday and stream:
                total_weight = max(0.0, self.config.everyday_weight) + max(0.0, self.config.stream_weight)
                everyday_probability = self.config.everyday_weight / total_weight if total_weight else 0.5
                source = "everyday" if self.rng.random() < everyday_probability else "stream"
                candidates = everyday if source == "everyday" else stream
                return self.rng.choice(candidates), source
            if everyday:
                return self.rng.choice(everyday), "everyday"
            if stream:
                return self.rng.choice(stream), "stream"

        candidates = FALLBACK_REACTIONS.get(category, FALLBACK_REACTIONS["Neutral"])
        return self.rng.choice(candidates), "fallback"

    def make_tts_text(self, reaction: str, keywords: list[str]) -> str:
        reaction = reaction.strip()
        if not keywords:
            return reaction
        keyword = keywords[-1].strip(".,!?;:\"'()")
        if not keyword:
            return reaction
        return f"{reaction} {keyword}?"

    def generate(self, user_text: str) -> dict[str, Any]:
        started = time.perf_counter()

        emotion_started = time.perf_counter()
        emotion = self.classify_emotion(user_text)
        emotion_ms = (time.perf_counter() - emotion_started) * 1000.0

        keyword_started = time.perf_counter()
        keywords = self.extract_keywords(user_text)
        keyword_ms = (time.perf_counter() - keyword_started) * 1000.0

        reaction, source = self.choose_reaction(emotion["category"])
        tts_text = self.make_tts_text(reaction, keywords)
        total_ms = (time.perf_counter() - started) * 1000.0

        return {
            "tts_text": tts_text,
            "reaction": reaction,
            "reaction_source": source,
            "keyword": keywords[-1] if keywords else None,
            "keywords": keywords,
            "emotion": emotion["category"],
            "emotion_label": emotion["label"],
            "emotion_score": round(emotion["score"], 4),
            "category_scores": {key: round(value, 4) for key, value in emotion["category_scores"].items()},
            "latency_ms": round(total_ms, 3),
            "emotion_ms": round(emotion_ms, 3),
            "keyword_ms": round(keyword_ms, 3),
        }


_DEFAULT_ENGINE: HybridFastTrack | None = None


def get_default_engine() -> HybridFastTrack:
    global _DEFAULT_ENGINE
    if _DEFAULT_ENGINE is None:
        _DEFAULT_ENGINE = HybridFastTrack()
    return _DEFAULT_ENGINE


def generate_fast_tts_text(user_text: str) -> str:
    return get_default_engine().generate(user_text)["tts_text"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run hybrid Fast Track once.")
    parser.add_argument("--text", required=True)
    parser.add_argument("--reaction-path", type=Path, default=DEFAULT_REACTION_PATH)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-runtime-score", type=float, default=0.0)
    parser.add_argument("--everyday-weight", type=float, default=0.60)
    parser.add_argument("--stream-weight", type=float, default=0.40)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    engine = HybridFastTrack(
        HybridFastTrackConfig(
            reaction_path=args.reaction_path,
            device=args.device,
            seed=args.seed,
            min_runtime_score=args.min_runtime_score,
            everyday_weight=args.everyday_weight,
            stream_weight=args.stream_weight,
        )
    )
    print(json.dumps(engine.generate(args.text), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
