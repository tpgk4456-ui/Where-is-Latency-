"""
Build a hybrid reaction list for the Fast Track path.

Data sources:
- daily_dialog: everyday short conversational reactions
- google-research-datasets/go_emotions: stream-style short raw texts

Output shape:
{
  "Positive": {"everyday": [...], "stream": [...]},
  "Negative": {"everyday": [...], "stream": [...]},
  "Ambiguous": {"everyday": [...], "stream": [...]},
  "Neutral": {"everyday": [...], "stream": [...]}
}

Install:
  python3 -m pip install -r AI_NPC_System/reaction_pipeline/requirements_hybrid_reactions.txt
  python3 -m spacy download en_core_web_sm

Run:
  python3 AI_NPC_System/reaction_pipeline/build_hybrid_reactions.py

Use --max-everyday-candidates 0 --max-stream-candidates 0 to classify every
short candidate found in both datasets. The defaults cap candidate counts so
the first experiment can finish quickly on a CPU machine.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "hybrid_reactions.json"
MODEL_NAME = "joeddav/distilbert-base-uncased-go-emotions-student"

CATEGORIES = ("Positive", "Negative", "Ambiguous", "Neutral")

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

WORD_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
SPACE_RE = re.compile(r"\s+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build hybrid Fast Track reaction JSON.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--score-threshold", type=float, default=0.70)
    parser.add_argument("--max-words", type=int, default=5)
    parser.add_argument("--daily-split", default="train")
    parser.add_argument("--goemotions-split", default="train")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-everyday-candidates", type=int, default=12000)
    parser.add_argument("--max-stream-candidates", type=int, default=20000)
    parser.add_argument("--max-per-bucket", type=int, default=400)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--torch-threads", type=int, default=0)
    return parser.parse_args()


def require_runtime_deps() -> tuple[Any, Any, Any]:
    try:
        import torch
        from datasets import load_dataset
        from transformers import pipeline
    except ImportError as exc:
        print("Missing dependency. Install with:")
        print("  python3 -m pip install -r AI_NPC_System/reaction_pipeline/requirements_hybrid_reactions.txt")
        print("  python3 -m spacy download en_core_web_sm")
        raise SystemExit(2) from exc
    return torch, load_dataset, pipeline


def normalize_text(text: str) -> str | None:
    text = URL_RE.sub("", text)
    text = text.replace("\n", " ").replace("\r", " ")
    text = SPACE_RE.sub(" ", text).strip()
    if not text:
        return None
    if len(text) > 96:
        return None
    if any(mark in text for mark in ("<", ">", "{", "}", "[", "]")):
        return None
    try:
        text.encode("ascii")
    except UnicodeEncodeError:
        return None
    if not WORD_RE.search(text):
        return None
    return text


def word_count(text: str) -> int:
    return len(WORD_RE.findall(text))


def is_short_reaction(text: str, max_words: int) -> bool:
    count = word_count(text)
    return 1 <= count <= max_words


def dedupe_keep_order(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        key = item.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def cap_candidates(items: list[str], cap: int, seed: int) -> list[str]:
    if cap <= 0 or len(items) <= cap:
        return items
    rng = random.Random(seed)
    sampled = rng.sample(items, cap)
    order = {item: index for index, item in enumerate(items)}
    return sorted(sampled, key=lambda x: order[x])


def load_daily_dialog_candidates(load_dataset: Any, split: str, max_words: int) -> list[str]:
    dataset = load_dataset("daily_dialog", split=split)
    candidates: list[str] = []
    for row in dataset:
        dialogue = row.get("dialog") or []
        for utterance in dialogue:
            if not isinstance(utterance, str):
                continue
            text = normalize_text(utterance)
            if text and is_short_reaction(text, max_words):
                candidates.append(text)
    return dedupe_keep_order(candidates)


def load_goemotions_dataset(load_dataset: Any, split: str) -> Any:
    try:
        return load_dataset("google-research-datasets/go_emotions", "simplified", split=split)
    except Exception:
        return load_dataset("go_emotions", "simplified", split=split)


def load_goemotions_candidates(load_dataset: Any, split: str, max_words: int) -> list[str]:
    dataset = load_goemotions_dataset(load_dataset, split)
    candidates: list[str] = []
    for row in dataset:
        raw_text = row.get("text") or ""
        if not isinstance(raw_text, str):
            continue
        text = normalize_text(raw_text)
        if text and is_short_reaction(text, max_words):
            candidates.append(text)
    return dedupe_keep_order(candidates)


def choose_device(torch: Any, requested: str) -> int:
    if requested == "cpu":
        return -1
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("CUDA was requested, but torch.cuda.is_available() is false.")
        return 0
    return 0 if torch.cuda.is_available() else -1


def normalize_pipeline_result(item: Any) -> dict[str, Any]:
    if isinstance(item, list):
        if not item:
            return {"label": "neutral", "score": 0.0}
        return item[0]
    return item


def classify_candidates(
    classifier: Any,
    candidates: list[str],
    source_name: str,
    score_threshold: float,
    batch_size: int,
) -> tuple[dict[str, list[str]], dict[str, Any]]:
    buckets: dict[str, list[str]] = {category: [] for category in CATEGORIES}
    rejected_low_score = 0
    rejected_unmapped = 0
    label_counts: dict[str, int] = defaultdict(int)

    started = time.perf_counter()
    for start in range(0, len(candidates), batch_size):
        batch = candidates[start : start + batch_size]
        results = classifier(batch, truncation=True, batch_size=batch_size)

        for text, raw in zip(batch, results):
            result = normalize_pipeline_result(raw)
            label = str(result.get("label", "neutral")).lower()
            score = float(result.get("score", 0.0))
            label_counts[label] += 1

            category = LABEL_TO_CATEGORY.get(label)
            if category is None:
                rejected_unmapped += 1
                continue
            if score < score_threshold:
                rejected_low_score += 1
                continue
            buckets[category].append(text)

    elapsed = time.perf_counter() - started
    report = {
        "source": source_name,
        "input_candidates": len(candidates),
        "kept": sum(len(values) for values in buckets.values()),
        "rejected_low_score": rejected_low_score,
        "rejected_unmapped": rejected_unmapped,
        "elapsed_seconds": round(elapsed, 3),
        "label_counts": dict(sorted(label_counts.items())),
    }
    return buckets, report


def trim_buckets(buckets: dict[str, list[str]], max_per_bucket: int, seed: int) -> dict[str, list[str]]:
    rng = random.Random(seed)
    trimmed: dict[str, list[str]] = {}
    for category, values in buckets.items():
        unique_values = dedupe_keep_order(values)
        if max_per_bucket > 0 and len(unique_values) > max_per_bucket:
            unique_values = rng.sample(unique_values, max_per_bucket)
            unique_values.sort(key=str.casefold)
        trimmed[category] = unique_values
    return trimmed


def build_reaction_json(args: argparse.Namespace) -> dict[str, Any]:
    torch, load_dataset, pipeline = require_runtime_deps()
    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)

    device = choose_device(torch, args.device)
    classifier = pipeline(
        "text-classification",
        model=args.model,
        tokenizer=args.model,
        top_k=1,
        device=device,
    )

    print("Loading daily_dialog candidates...")
    everyday = load_daily_dialog_candidates(load_dataset, args.daily_split, args.max_words)
    everyday = cap_candidates(everyday, args.max_everyday_candidates, args.seed)
    print(f"daily_dialog short candidates: {len(everyday)}")

    print("Loading GoEmotions candidates...")
    stream = load_goemotions_candidates(load_dataset, args.goemotions_split, args.max_words)
    stream = cap_candidates(stream, args.max_stream_candidates, args.seed + 1)
    print(f"GoEmotions short candidates: {len(stream)}")

    everyday_buckets, everyday_report = classify_candidates(
        classifier=classifier,
        candidates=everyday,
        source_name="daily_dialog",
        score_threshold=args.score_threshold,
        batch_size=args.batch_size,
    )
    stream_buckets, stream_report = classify_candidates(
        classifier=classifier,
        candidates=stream,
        source_name="go_emotions",
        score_threshold=args.score_threshold,
        batch_size=args.batch_size,
    )

    everyday_buckets = trim_buckets(everyday_buckets, args.max_per_bucket, args.seed)
    stream_buckets = trim_buckets(stream_buckets, args.max_per_bucket, args.seed + 1)

    output: dict[str, Any] = {
        category: {
            "everyday": everyday_buckets.get(category, []),
            "stream": stream_buckets.get(category, []),
        }
        for category in CATEGORIES
    }
    output["meta"] = {
        "version": "hybrid-v01",
        "model": args.model,
        "score_threshold": args.score_threshold,
        "max_words": args.max_words,
        "sources": {
            "everyday": "daily_dialog",
            "stream": "google-research-datasets/go_emotions:simplified",
        },
        "reports": [everyday_report, stream_report],
        "bucket_counts": {
            category: {
                "everyday": len(output[category]["everyday"]),
                "stream": len(output[category]["stream"]),
            }
            for category in CATEGORIES
        },
    }
    return output


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    output = build_reaction_json(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    elapsed = time.perf_counter() - started
    print(f"Wrote {args.output}")
    print(f"Total elapsed seconds: {elapsed:.3f}")
    print(json.dumps(output["meta"]["bucket_counts"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
