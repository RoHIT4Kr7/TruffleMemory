from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Iterable

from config.settings import Settings, get_settings
from memory.storage.chroma_store import ChromaHit, ChromaStore
from memory.storage.sqlite_store import SQLiteStore


@dataclass(slots=True)
class RetrievalPack:
    profile_text: str
    contexts: list[dict[str, Any]]
    episode_ids: list[str]
    summary_ids: list[str]
    self_aliases: list[str]
    trace: list[dict[str, Any]]
    scope_mode: str
    scope_chat_id: str


class MemoryRetriever:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.sqlite = SQLiteStore(self.settings.sqlite_path)
        self.chroma = ChromaStore(self.settings.chroma_dir)
        self.known_chat_ids = _discover_known_chat_ids(self.settings.whatsapp_dir)

    def retrieve(
        self,
        query: str,
        token_budget: int,
        *,
        include_profile: bool = True,
        exclude_episode_ids: set[str] | None = None,
    ) -> RetrievalPack:
        excluded = exclude_episode_ids or set()
        indexed_chat_ids = set(self.sqlite.list_chat_ids(limit=500))
        scoped_chat_id = _resolve_chat_scope(query, self.known_chat_ids)
        scope_filter = {"chat_id": scoped_chat_id} if scoped_chat_id else None

        weekly_hits = self.chroma.query_weekly(query, n_results=8, where=scope_filter)
        raw_hits = self.chroma.query_raw(query, n_results=18, where=scope_filter)
        lexical_hits = self.sqlite.search_episodes_lexical(query, limit=12, chat_id=scoped_chat_id)

        summary_ids = [hit.doc_id for hit in weekly_hits]
        episode_scores: dict[str, float] = {}
        trace: list[dict[str, Any]] = []
        scope_mode = "global"
        if scoped_chat_id:
            scope_mode = "chat_scoped" if scoped_chat_id in indexed_chat_ids else "chat_scoped_pending"
        trace.append(
            {
                "source": "scope",
                "mode": scope_mode,
                "chat_id": scoped_chat_id or "",
            }
        )

        for hit in weekly_hits:
            linked_episode_ids = _parse_episode_ids(hit.metadata.get("episode_ids"))
            for episode_id in linked_episode_ids:
                episode_scores[episode_id] = episode_scores.get(episode_id, 0.0) + (0.50 * hit.score)
            trace.append(
                {
                    "source": "weekly_summaries",
                    "id": hit.doc_id,
                    "score": hit.score,
                    "week_key": hit.metadata.get("week_key", ""),
                }
            )

        for hit in raw_hits:
            episode_scores[hit.doc_id] = episode_scores.get(hit.doc_id, 0.0) + (0.72 * hit.score)
            trace.append(
                {
                    "source": "raw_episodes",
                    "id": hit.doc_id,
                    "score": hit.score,
                    "tier": hit.metadata.get("tier", ""),
                }
            )

        for row in lexical_hits:
            eid = str(row["id"])
            episode_scores[eid] = episode_scores.get(eid, 0.0) + (0.60 * float(row.get("score", 0.25)))
            trace.append(
                {
                    "source": "lexical",
                    "id": eid,
                    "score": float(row.get("score", 0.25)),
                }
            )

        ranked_episode_ids = [
            episode_id
            for episode_id, _score in sorted(
                episode_scores.items(),
                key=lambda pair: pair[1],
                reverse=True,
            )
            if episode_id not in excluded
        ]

        rows = self.sqlite.fetch_episodes_by_ids(ranked_episode_ids)
        contexts: list[dict[str, Any]] = []
        used_tokens = 0

        # Weekly summaries first: cheap semantic context.
        for hit in weekly_hits:
            text = hit.text.strip()
            if not text:
                continue
            estimate = _estimate_tokens(text)
            if used_tokens + estimate > int(token_budget * 0.35):
                break
            contexts.append(
                {
                    "source_type": "weekly_summary",
                    "source_id": hit.doc_id,
                    "score": hit.score,
                    "text": text,
                    "metadata": hit.metadata,
                }
            )
            used_tokens += estimate

        chosen_episode_ids: list[str] = []
        for row in rows:
            text = str(row["text_raw"])
            estimate = _estimate_tokens(text)
            if used_tokens + estimate > token_budget:
                continue

            chosen_episode_ids.append(str(row["id"]))
            contexts.append(
                {
                    "source_type": "episode",
                    "source_id": str(row["id"]),
                    "score": episode_scores.get(str(row["id"]), 0.0),
                    "text": text,
                    "metadata": {
                        "chat_id": row["chat_id"],
                        "start_ts": row["start_ts"],
                        "end_ts": row["end_ts"],
                        "tier": row.get("tier", "pass_through"),
                        "salience": row.get("salience", 0.0),
                    },
                }
            )
            used_tokens += estimate

        self.sqlite.log_retrieval(query, summary_ids=summary_ids, episode_ids=chosen_episode_ids)

        profile_text = ""
        if include_profile and not scoped_chat_id:
            profile_text = self._load_profile_text()
        self_aliases = self.sqlite.fetch_likely_self_senders(limit=3, min_chats=2)

        return RetrievalPack(
            profile_text=profile_text,
            contexts=contexts,
            episode_ids=chosen_episode_ids,
            summary_ids=summary_ids,
            self_aliases=self_aliases,
            trace=trace,
            scope_mode=scope_mode,
            scope_chat_id=scoped_chat_id or "",
        )

    def _load_profile_text(self) -> str:
        profile_path = Path(self.settings.profile_path)
        if not profile_path.exists():
            return "{}"
        return profile_path.read_text(encoding="utf-8")


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _parse_episode_ids(raw: object) -> list[str]:
    if isinstance(raw, list):
        return [str(item) for item in raw]

    if isinstance(raw, str):
        stripped = raw.strip()
        if not stripped:
            return []
        try:
            parsed = json.loads(stripped)
            if isinstance(parsed, list):
                return [str(item) for item in parsed]
        except json.JSONDecodeError:
            return [stripped]
    return []


def _resolve_chat_scope(query: str, chat_ids: list[str]) -> str | None:
    if not chat_ids:
        return None

    matches = _find_chat_mentions(query, chat_ids)
    query_norm = _norm_space(query)
    if _is_explicit_global_query(query_norm):
        return None

    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        return None

    if _is_cross_chat_query(query_norm):
        return None
    return None


def _find_chat_mentions(query: str, chat_ids: list[str]) -> list[str]:
    query_norm = _norm_space(query)
    query_compact = _norm_compact(query)
    matched: list[tuple[str, int]] = []

    for chat_id in chat_ids:
        aliases = _chat_aliases(chat_id)
        longest = 0
        for alias in aliases:
            alias = alias.strip()
            if not alias:
                continue
            if len(alias) < 3:
                continue
            if alias in query_norm:
                longest = max(longest, len(alias))
                continue
            alias_compact = alias.replace(" ", "")
            if alias_compact and alias_compact in query_compact:
                longest = max(longest, len(alias))
        if longest > 0:
            matched.append((chat_id, longest))

    if not matched:
        return []
    matched.sort(key=lambda item: item[1], reverse=True)
    best_len = matched[0][1]
    return [chat_id for chat_id, size in matched if size == best_len]


def _chat_aliases(chat_id: str) -> set[str]:
    raw = str(chat_id).strip()
    aliases = {_norm_space(raw)}
    # Common spoken shortcuts for WhatsApp exports.
    aliases.add(_norm_space(raw.replace("&", "and")))
    aliases.add(_norm_space(raw.replace("_", " ")))
    return {alias for alias in aliases if alias}


def _is_explicit_global_query(query_norm: str) -> bool:
    explicit_phrases = (
        "across all chats",
        "across chats",
        "across profiles",
        "all chats",
        "all profiles",
        "overall",
        "in general",
        "globally",
    )
    return any(phrase in query_norm for phrase in explicit_phrases)


def _is_cross_chat_query(query_norm: str) -> bool:
    phrases = (
        "compare",
        "comparison",
        "between",
        "versus",
        " vs ",
        "connection",
        "connect",
        "relation",
        "cross chat",
    )
    return any(phrase in query_norm for phrase in phrases)


def _norm_space(text: str) -> str:
    lowered = str(text).lower()
    cleaned = re.sub(r"[^a-z0-9]+", " ", lowered)
    return re.sub(r"\s+", " ", cleaned).strip()


def _norm_compact(text: str) -> str:
    return _norm_space(text).replace(" ", "")


def _discover_known_chat_ids(whatsapp_dir: Path) -> list[str]:
    chats: list[str] = []
    prefix = "WhatsApp Chat with "
    for file_path in sorted(whatsapp_dir.glob("*.txt")):
        stem = file_path.stem.strip()
        if stem.startswith(prefix):
            stem = stem[len(prefix) :].strip()
        if stem:
            chats.append(stem)
    return chats
