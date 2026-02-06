from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(slots=True)
class ChatMessage:
    message_id: str
    chat_id: str
    timestamp: datetime
    sender: str | None
    content_raw: str
    is_system: bool = False
    has_media: bool = False


@dataclass(slots=True)
class Episode:
    episode_id: str
    chat_id: str
    start_ts: datetime
    end_ts: datetime
    message_ids: list[str]
    text_raw: str
    text_search: str
    salience: float
    tier: str
    sensitivity_tags: list[str] = field(default_factory=list)


@dataclass(slots=True)
class SummaryRecord:
    summary_id: str
    episode_id: str | None
    chat_id: str
    week_key: str
    summary_text: str
    summary_type: str
    confidence: float
    evidence_message_ids: list[str] = field(default_factory=list)


@dataclass(slots=True)
class FactRecord:
    fact_id: str
    fact_text: str
    category: str
    confidence: float
    first_seen_at: datetime
    last_seen_at: datetime
    evidence_message_ids: list[str] = field(default_factory=list)
    source_episode_id: str | None = None


@dataclass(slots=True)
class RetrievalContext:
    source_type: str
    source_id: str
    score: float
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RetrievalResult:
    profile_text: str
    contexts: list[RetrievalContext]
    trace: list[dict[str, Any]] = field(default_factory=list)
