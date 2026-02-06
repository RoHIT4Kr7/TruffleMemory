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
        query_norm = _norm_space(query)
        scope_filter = {"chat_id": scoped_chat_id} if scoped_chat_id else None
        lexical_queries = _lexical_query_variants(query)

        weekly_hits = self.chroma.query_weekly(query, n_results=8, where=scope_filter)
        raw_hits = self.chroma.query_raw(query, n_results=18, where=scope_filter)
        lexical_hits = _collect_episode_lexical_hits(
            sqlite=self.sqlite,
            chat_id=scoped_chat_id,
            lexical_queries=lexical_queries,
            limit=12,
        )

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

        # For global person/profile questions, blend a few timeline anchors so
        # retrieval does not collapse to only recent episodes.
        if (
            not scoped_chat_id
            and _needs_historical_anchors(query_norm)
            and used_tokens < token_budget
        ):
            anchor_rows = self.sqlite.fetch_earliest_episode_per_chat(limit=16)
            anchor_budget = int(token_budget * 0.25)
            used_anchor_tokens = 0
            existing_ids = set(chosen_episode_ids)

            for row in anchor_rows:
                anchor_id = str(row["id"])
                if anchor_id in excluded or anchor_id in existing_ids:
                    continue
                text = str(row.get("text_raw", "")).strip()
                if not text:
                    continue
                estimate = _estimate_tokens(text)
                if estimate <= 0:
                    continue
                if used_anchor_tokens + estimate > anchor_budget:
                    continue
                if used_tokens + estimate > token_budget:
                    break

                chosen_episode_ids.append(anchor_id)
                existing_ids.add(anchor_id)
                contexts.append(
                    {
                        "source_type": "episode_anchor",
                        "source_id": anchor_id,
                        "score": 0.22,
                        "text": text,
                        "metadata": {
                            "chat_id": row.get("chat_id", ""),
                            "start_ts": row.get("start_ts", ""),
                            "end_ts": row.get("end_ts", ""),
                            "tier": row.get("tier", "pass_through"),
                            "kind": "historical_anchor",
                        },
                    }
                )
                used_tokens += estimate
                used_anchor_tokens += estimate
                trace.append(
                    {
                        "source": "historical_anchor",
                        "id": anchor_id,
                        "chat_id": str(row.get("chat_id", "")),
                    }
                )

        # Warmup fallback: when a specific chat is asked before its episodes are indexed,
        # pull message-level evidence so basic chat-grounded answers can still work.
        if scoped_chat_id and (scope_mode == "chat_scoped_pending" or len(contexts) < 4):
            message_hits = _collect_message_lexical_hits(
                sqlite=self.sqlite,
                chat_id=scoped_chat_id,
                lexical_queries=lexical_queries[:2],
                limit=28,
            )
            message_budget = int(token_budget * (0.45 if scope_mode == "chat_scoped_pending" else 0.22))
            message_budget = max(0, min(message_budget, token_budget - used_tokens))
            used_message_tokens = 0

            for hit in message_hits:
                text = _message_context_text(hit)
                estimate = _estimate_tokens(text)
                if estimate <= 0:
                    continue
                if used_message_tokens + estimate > message_budget:
                    break
                contexts.append(
                    {
                        "source_type": "message",
                        "source_id": str(hit["id"]),
                        "score": float(hit.get("score", 0.0)),
                        "text": text,
                        "metadata": {
                            "chat_id": scoped_chat_id,
                            "timestamp": str(hit.get("timestamp", "")),
                            "sender": str(hit.get("sender") or "unknown"),
                            "kind": "message_lexical_fallback",
                        },
                    }
                )
                used_message_tokens += estimate
                used_tokens += estimate

            if message_hits:
                for row in message_hits[:8]:
                    trace.append(
                        {
                            "source": "message_fallback",
                            "id": str(row["id"]),
                            "score": float(row.get("score", 0.0)),
                        }
                    )

        self.sqlite.log_retrieval(query, summary_ids=summary_ids, episode_ids=chosen_episode_ids)

        profile_text = ""
        if include_profile and not scoped_chat_id:
            profile_text = self._load_profile_text()

        return RetrievalPack(
            profile_text=profile_text,
            contexts=contexts,
            episode_ids=chosen_episode_ids,
            summary_ids=summary_ids,
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


def _collect_episode_lexical_hits(
    *,
    sqlite: SQLiteStore,
    chat_id: str | None,
    lexical_queries: list[str],
    limit: int,
) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for idx, lexical_query in enumerate(lexical_queries):
        weight = 1.0 if idx == 0 else 0.82
        rows = sqlite.search_episodes_lexical(lexical_query, limit=limit, chat_id=chat_id)
        for row in rows:
            eid = str(row["id"])
            score = float(row.get("score", 0.0)) * weight
            existing = by_id.get(eid)
            if existing is None or score > float(existing.get("score", 0.0)):
                by_id[eid] = {**dict(row), "score": score}
    ranked = sorted(by_id.values(), key=lambda item: float(item.get("score", 0.0)), reverse=True)
    return ranked[: limit * 2]


def _collect_message_lexical_hits(
    *,
    sqlite: SQLiteStore,
    chat_id: str,
    lexical_queries: list[str],
    limit: int,
) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    queries = lexical_queries or [""]
    for idx, lexical_query in enumerate(queries):
        weight = 1.0 if idx == 0 else 0.85
        rows = sqlite.search_messages_lexical(lexical_query, chat_id=chat_id, limit=limit)
        for row in rows:
            mid = str(row["id"])
            score = float(row.get("score", 0.0)) * weight
            existing = by_id.get(mid)
            if existing is None or score > float(existing.get("score", 0.0)):
                by_id[mid] = {**dict(row), "score": score}
    ranked = sorted(by_id.values(), key=lambda item: float(item.get("score", 0.0)), reverse=True)
    return ranked[: limit]


def _message_context_text(row: dict[str, Any]) -> str:
    ts = str(row.get("timestamp", "")).strip()
    sender = str(row.get("sender") or "unknown").strip() or "unknown"
    content = str(row.get("content_raw", "")).strip()
    if len(content) > 360:
        content = content[:357] + "..."
    return f"[{ts}] {sender}: {content}"


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


def _lexical_query_variants(query: str) -> list[str]:
    base = str(query).strip()
    if not base:
        return [""]

    variants = [base]
    qn = _norm_space(base)

    if any(term in qn for term in ("best friend", "closest friend", "closest contact", "most interacted")):
        variants.append(f"{base} friend close trust support")

    if any(term in qn for term in ("owe", "owed", "borrow", "money", "pay", "payment", "upi", "loan")):
        variants.append(f"{base} money owe pay transfer upi")

    if any(term in qn for term in ("subject", "study", "syllabus", "exam", "class", "do")):
        variants.append(f"{base} subject syllabus study exam")

    if any(term in qn for term in ("foul", "abuse", "gaali", "worst language", "bad language")):
        variants.append(f"{base} abuse foul language gaali")

    if any(term in qn for term in ("summarise", "summarize", "conversation", "talked")):
        variants.append(f"{base} conversation chat discussed talked")

    deduped: list[str] = []
    seen: set[str] = set()
    for variant in variants:
        key = _norm_space(variant)
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(variant)
    return deduped[:3]


def _needs_historical_anchors(query_norm: str) -> bool:
    if not query_norm:
        return False
    phrases = (
        "all contacts",
        "among all contacts",
        "all friends",
        "among friends",
        "best friend",
        "closest friend",
        "what do you know about me",
        "about me",
        "who am i",
        "across all chats",
        "overall",
    )
    return any(phrase in query_norm for phrase in phrases)


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
