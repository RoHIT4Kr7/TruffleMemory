from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

from memory.storage.sqlite_store import SQLiteStore


def build_profile_snapshot(store: SQLiteStore, output_path: Path) -> dict[str, object]:
    contacts = store.fetch_top_contacts(limit=25)
    facts = store.fetch_top_facts(limit=80)
    languages = store.detect_languages(limit=5000)
    tone = _infer_tone(store.fetch_recent_message_samples(limit=1200))

    profile = {
        "user": {
            "contacts": {name: count for name, count in contacts},
            "facts": [fact for fact, _score in facts],
            "languages": languages,
            "tone": tone,
        }
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")
    return profile


def _infer_tone(samples: list[str]) -> str:
    if not samples:
        return "Neutral"

    joined = " ".join(samples).lower()
    markers = {
        "casual": ("bhai", "yaar", "bro", "lol", "lmao"),
        "formal": ("regards", "dear", "kindly", "please find attached"),
        "tech": ("python", "repo", "commit", "bug", "deploy"),
    }

    scores: Counter[str] = Counter()
    for tone, words in markers.items():
        for word in words:
            scores[tone] += joined.count(word)

    if not scores:
        return "Mixed"

    ordered = [name for name, score in scores.most_common() if score > 0]
    if not ordered:
        return "Mixed"
    return ", ".join(ordered[:2]).title()
