from __future__ import annotations

from datetime import datetime, timezone
import math
import re

from memory.models import Episode

HIGH_SIGNAL_TERMS = (
    "important",
    "urgent",
    "invoice",
    "meeting",
    "deadline",
    "offer",
    "problem",
    "issue",
    "password",
    "otp",
    "account",
)


def score_episode_salience(episode: Episode, now: datetime | None = None) -> float:
    now_dt = now or datetime.now(timezone.utc).replace(tzinfo=None)
    age_days = max((now_dt - episode.end_ts).days, 0)

    recency = math.exp(-age_days / 210.0)

    text = episode.text_raw.lower()
    keyword_hits = sum(1 for term in HIGH_SIGNAL_TERMS if term in text)
    keyword_score = min(keyword_hits / 4.0, 1.0)

    question_score = min(text.count("?") / 4.0, 1.0)
    size_score = min(len(episode.message_ids) / 15.0, 1.0)

    # Weighted mix keeps old but important events retrievable.
    score = (0.38 * recency) + (0.26 * keyword_score) + (0.18 * question_score) + (0.18 * size_score)
    return max(0.0, min(1.0, score))


def assign_tier(score: float, deep_threshold: float, medium_threshold: float) -> str:
    if score >= deep_threshold:
        return "deep"
    if score >= medium_threshold:
        return "medium"
    return "pass_through"
