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
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "hybrid_reactions.json"
MODEL_NAME = "joeddav/distilbert-base-uncased-go-emotions-student"
DAILY_DIALOG_URL = "http://yanran.li/files/ijcnlp_dailydialog.zip"
DAILY_DIALOG_HF_MIRROR = "roskoN/dailydialog"
RAW_CACHE_DIR = Path("/tmp/credo_hybrid_reaction_cache")

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
PUNCT_SPACE_RE = re.compile(r"\s+([?.!,;:])")
ARTICLE_START_RE = re.compile(r"^(a|an|the)\s+[a-z0-9]", re.IGNORECASE)
CONTEXT_BOUND_TERMS = {
    "he",
    "him",
    "his",
    "she",
    "hers",
    "they",
    "them",
    "their",
    "theirs",
}
BLOCKED_TERMS = {
    "fuck",
    "fucking",
    "fucked",
    "shit",
    "bullshit",
    "bitch",
    "bitches",
    "asshole",
    "dick",
    "cunt",
    "slur",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build hybrid Fast Track reaction JSON.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--min-top-score", type=float, default=0.45)
    parser.add_argument("--min-margin", type=float, default=0.50)
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
    text = PUNCT_SPACE_RE.sub(r"\1", text)
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
    words = {word.casefold() for word in WORD_RE.findall(text)}
    if words & CONTEXT_BOUND_TERMS:
        return None
    if words & BLOCKED_TERMS:
        return None
    if any(mark in text for mark in ("#", "@", "*", "`", "|")):
        return None
    if ARTICLE_START_RE.search(text):
        return None
    if "please" in words:
        return None
    return text


def is_category_suitable(text: str, category: str) -> bool:
    if category in {"Positive", "Negative"} and "?" in text:
        return False
    return True


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


def normalize_daily_dialog_split(split: str) -> str:
    if split == "validation":
        return "validation"
    if split in {"valid", "val", "dev"}:
        return "validation"
    if split in {"train", "test"}:
        return split
    raise ValueError(f"Unsupported daily_dialog split: {split}")


def download_daily_dialog_zip() -> Path:
    RAW_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = RAW_CACHE_DIR / "ijcnlp_dailydialog.zip"
    if zip_path.exists() and zip_path.stat().st_size > 0:
        return zip_path
    urllib.request.urlretrieve(DAILY_DIALOG_URL, zip_path)
    return zip_path


def download_daily_dialog_split_zip(split_name: str) -> Path:
    try:
        from huggingface_hub import hf_hub_download

        return Path(
            hf_hub_download(
                repo_id=DAILY_DIALOG_HF_MIRROR,
                repo_type="dataset",
                filename=f"{split_name}.zip",
            )
        )
    except Exception:
        return download_daily_dialog_zip()


def load_daily_dialog_candidates(load_dataset: Any, split: str, max_words: int) -> list[str]:
    del load_dataset
    split_name = normalize_daily_dialog_split(split)
    zip_path = download_daily_dialog_split_zip(split_name)
    candidates: list[str] = []
    dialog_path = f"{split_name}/dialogues_{split_name}.txt"
    with ZipFile(zip_path) as split_zip:
        if dialog_path not in split_zip.namelist():
            nested_zip_path = f"ijcnlp_dailydialog/{split_name}.zip"
            with split_zip.open(nested_zip_path) as data_zip_file:
                with ZipFile(data_zip_file) as nested_zip:
                    return parse_daily_dialog_zip(nested_zip, dialog_path, max_words)
        candidates = parse_daily_dialog_zip(split_zip, dialog_path, max_words)
    return dedupe_keep_order(candidates)


def parse_daily_dialog_zip(split_zip: ZipFile, dialog_path: str, max_words: int) -> list[str]:
    candidates: list[str] = []
    with split_zip.open(dialog_path) as dialog_file:
        for raw_line in dialog_file:
            line = raw_line.decode("utf-8", errors="ignore").strip()
            if not line:
                continue
            for utterance in line.split("__eou__"):
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


def normalize_score_list(item: Any) -> list[dict[str, Any]]:
    if isinstance(item, list):
        if not item:
            return [{"label": "neutral", "score": 0.0}]
        if isinstance(item[0], list):
            return item[0]
        return item
    return [item]


def aggregate_category_scores(score_items: list[dict[str, Any]]) -> dict[str, float]:
    scores = {category: 0.0 for category in CATEGORIES}
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


def classify_candidates(
    classifier: Any,
    candidates: list[str],
    source_name: str,
    min_top_score: float,
    min_margin: float,
    batch_size: int,
) -> tuple[dict[str, list[str]], dict[str, Any]]:
    buckets: dict[str, list[str]] = {category: [] for category in CATEGORIES}
    rejected_low_top_score = 0
    rejected_low_margin = 0
    rejected_unmapped = 0
    label_counts: dict[str, int] = defaultdict(int)

    started = time.perf_counter()
    for start in range(0, len(candidates), batch_size):
        batch = candidates[start : start + batch_size]
        results = classifier(batch, truncation=True, batch_size=batch_size)

        for text, raw in zip(batch, results):
            score_items = normalize_score_list(raw)
            label = top_label(score_items)
            category_scores = aggregate_category_scores(score_items)
            ranked_categories = sorted(category_scores.items(), key=lambda item: item[1], reverse=True)
            category, score = ranked_categories[0]
            second_score = ranked_categories[1][1] if len(ranked_categories) > 1 else 0.0
            margin = score - second_score
            label_counts[label] += 1

            if score < min_top_score:
                rejected_low_top_score += 1
                continue
            if margin < min_margin:
                rejected_low_margin += 1
                continue
            if not is_category_suitable(text, category):
                continue
            buckets[category].append(text)

    elapsed = time.perf_counter() - started
    report = {
        "source": source_name,
        "input_candidates": len(candidates),
        "kept": sum(len(values) for values in buckets.values()),
        "rejected_low_top_score": rejected_low_top_score,
        "rejected_low_margin": rejected_low_margin,
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
        top_k=None,
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
        min_top_score=args.min_top_score,
        min_margin=args.min_margin,
        batch_size=args.batch_size,
    )
    stream_buckets, stream_report = classify_candidates(
        classifier=classifier,
        candidates=stream,
        source_name="go_emotions",
        min_top_score=args.min_top_score,
        min_margin=args.min_margin,
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
        "min_top_score": args.min_top_score,
        "min_margin": args.min_margin,
        "max_words": args.max_words,
        "sources": {
            "everyday": f"daily_dialog raw train.zip mirror ({DAILY_DIALOG_HF_MIRROR})",
            "stream": "google-research-datasets/go_emotions:simplified",
        },
        "confidence_filter": "after mapping 28 GoEmotions labels into 4 categories, keep items where top_category_score >= min_top_score and top_category_score - second_category_score >= min_margin",
        "quality_filters": [
            "max 5 words",
            "ASCII text",
            "URL/bracket/hashtag/mention/markdown removal",
            "context-bound third-person pronoun removal",
            "basic profanity removal",
            "question removal for Positive and Negative categories",
        ],
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
