from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
import hashlib
import logging
from pathlib import Path
from typing import Any

from config.settings import Settings, get_settings
from memory.chunker import chunk_messages
from memory.llm_client import OpenRouterLLM
from memory.models import Episode, FactRecord, SummaryRecord
from memory.parser import parse_chat_file
from memory.pii import detect_pii_tags, redact_sensitive_text
from memory.profile import build_profile_snapshot
from memory.rosetta import normalize_for_search
from memory.salience import assign_tier, score_episode_salience
from memory.storage.chroma_store import ChromaStore
from memory.storage.sqlite_store import SQLiteStore
from memory.summarizer import EpisodeSummarizer, SummaryBundle, make_weekly_summary

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class IngestionStats:
    files_seen: int = 0
    files_ingested: int = 0
    files_skipped: int = 0
    messages_ingested: int = 0
    episodes_ingested: int = 0
    weekly_summaries_created: int = 0
    facts_created: int = 0


@dataclass(slots=True)
class EnrichmentStats:
    deep_episodes_seen: int = 0
    deep_episodes_enriched: int = 0
    facts_created: int = 0


@dataclass(slots=True)
class ConsolidationStats:
    weeks_seen: int = 0
    weeks_consolidated: int = 0


class IngestionPipeline:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.sqlite = SQLiteStore(self.settings.sqlite_path)
        self.chroma = ChromaStore(self.settings.chroma_dir)
        self.llm = OpenRouterLLM(self.settings)
        self.summarizer = EpisodeSummarizer(self.llm if self.llm.enabled else None)

    def ingest_all(self, force: bool = False) -> IngestionStats:
        stats = IngestionStats()
        files = sorted(self.settings.whatsapp_dir.glob("*.txt"))
        stats.files_seen = len(files)

        for file_path in files:
            result = self.ingest_file(file_path, force=force)
            if result["status"] == "skipped":
                stats.files_skipped += 1
                continue

            stats.files_ingested += 1
            stats.messages_ingested += int(result["messages"])
            stats.episodes_ingested += int(result["episodes"])
            stats.weekly_summaries_created += int(result["weekly_summaries"])
            stats.facts_created += int(result["facts"])

        build_profile_snapshot(self.sqlite, self.settings.profile_path)
        return stats

    def ingest_file(self, file_path: str | Path, force: bool = False) -> dict[str, Any]:
        path = Path(file_path)
        file_hash = self._sha256(path)

        existing_job = self.sqlite.get_ingestion_job(str(path))
        if (
            not force
            and existing_job
            and existing_job.get("status") == "completed"
            and existing_job.get("file_hash") == file_hash
        ):
            logger.info("Skipping unchanged file: %s", path.name)
            return {
                "status": "skipped",
                "messages": int(existing_job.get("processed_messages", 0)),
                "episodes": int(existing_job.get("processed_episodes", 0)),
                "weekly_summaries": 0,
                "facts": 0,
            }

        self.sqlite.upsert_ingestion_job(
            file_path=str(path),
            file_hash=file_hash,
            status="running",
            processed_messages=0,
            processed_episodes=0,
        )

        try:
            messages = parse_chat_file(path)
            search_by_id: dict[str, str] = {}
            tags_by_id: dict[str, list[str]] = {}

            for msg in messages:
                tags = detect_pii_tags(msg.content_raw)
                tags_by_id[msg.message_id] = tags

                content = redact_sensitive_text(msg.content_raw) if tags else msg.content_raw
                normalized = normalize_for_search(content)
                search_by_id[msg.message_id] = str(normalized["normalized"])

            self.sqlite.upsert_messages(messages, search_by_id, tags_by_id)

            episodes = chunk_messages(
                messages,
                gap_hours=self.settings.chunk_gap_hours,
                max_chars=self.settings.chunk_max_chars,
            )

            prepared_episodes = self._prepare_episodes(episodes)
            self.sqlite.upsert_episodes(prepared_episodes)

            raw_docs: list[tuple[str, str, dict[str, Any]]] = []
            for episode in prepared_episodes:
                raw_docs.append(
                    (
                        episode.episode_id,
                        episode.text_search,
                        {
                            "chat_id": episode.chat_id,
                            "start_ts": episode.start_ts.isoformat(),
                            "end_ts": episode.end_ts.isoformat(),
                            "tier": episode.tier,
                            "salience": float(episode.salience),
                            "week_key": _week_key(episode.end_ts),
                        },
                    )
                )
            self.chroma.add_raw_episodes(raw_docs)

            self.sqlite.upsert_ingestion_job(
                file_path=str(path),
                file_hash=file_hash,
                status="completed",
                processed_messages=len(messages),
                processed_episodes=len(prepared_episodes),
            )

            return {
                "status": "completed",
                "messages": len(messages),
                "episodes": len(prepared_episodes),
                "weekly_summaries": 0,
                "facts": 0,
            }
        except Exception as exc:
            self.sqlite.upsert_ingestion_job(
                file_path=str(path),
                file_hash=file_hash,
                status="failed",
                processed_messages=0,
                processed_episodes=0,
                last_error=str(exc),
            )
            raise

    def enrich_deep_episodes(self, limit: int = 50) -> EnrichmentStats:
        bounded_limit = max(1, limit)
        episodes = self.sqlite.fetch_unenriched_deep_episodes(limit=bounded_limit)
        stats = EnrichmentStats(deep_episodes_seen=len(episodes))
        if not episodes:
            return stats

        facts_buffer: list[FactRecord] = []
        for episode in episodes:
            summary_bundle = self.summarizer.summarize(episode)
            episode_week = _week_key(episode.end_ts)
            episodic_summary = SummaryRecord(
                summary_id=_hash_id("episodic", episode.episode_id),
                episode_id=episode.episode_id,
                chat_id=episode.chat_id,
                week_key=episode_week,
                summary_text=summary_bundle.summary_text,
                summary_type="episodic",
                confidence=summary_bundle.confidence,
                evidence_message_ids=episode.message_ids[:20],
            )
            self.sqlite.insert_summary(episodic_summary)
            facts_buffer.extend(self._build_facts(episode, summary_bundle))
            stats.deep_episodes_enriched += 1

        self.sqlite.insert_facts(facts_buffer)
        stats.facts_created = len(facts_buffer)

        if stats.deep_episodes_enriched:
            build_profile_snapshot(self.sqlite, self.settings.profile_path)
        return stats

    def consolidate_weekly_summaries(self, limit: int = 20) -> ConsolidationStats:
        bounded_limit = max(1, limit)
        groups = self.sqlite.fetch_pending_weekly_groups(limit=bounded_limit)
        stats = ConsolidationStats(weeks_seen=len(groups))

        for group in groups:
            chat_id = str(group["chat_id"])
            week_key = str(group["week_key"])
            rows = self.sqlite.fetch_episodic_summaries_for_week(chat_id, week_key)
            if not rows:
                continue

            summary_texts = [str(row["summary_text"]) for row in rows if str(row["summary_text"]).strip()]
            week_text = make_weekly_summary(week_key, summary_texts, self.llm if self.llm.enabled else None)
            if not week_text:
                continue

            episode_ids = [str(row["episode_id"]) for row in rows if row.get("episode_id")]
            evidence_ids: list[str] = []
            for row in rows:
                evidence_ids.extend([str(item) for item in row.get("evidence_message_ids", [])[:8]])

            weekly_id = _hash_id("weekly", f"{chat_id}|{week_key}")
            weekly_record = SummaryRecord(
                summary_id=weekly_id,
                episode_id=None,
                chat_id=chat_id,
                week_key=week_key,
                summary_text=week_text,
                summary_type="weekly",
                confidence=0.7,
                evidence_message_ids=_compact_unique(evidence_ids, limit=100),
            )
            self.sqlite.insert_summary(weekly_record)
            self.chroma.add_weekly_summary(
                summary_id=weekly_id,
                text=week_text,
                metadata={
                    "chat_id": chat_id,
                    "week_key": week_key,
                    "episode_ids": _compact_unique(episode_ids, limit=150),
                },
            )
            stats.weeks_consolidated += 1

        return stats

    def _prepare_episodes(self, episodes: list[Episode]) -> list[Episode]:
        if not episodes:
            return []

        # Parallel preprocessing keeps ingestion throughput healthy while model calls remain bounded.
        max_workers = max(1, self.settings.max_workers)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            return list(executor.map(self._prepare_episode, episodes))

    def _prepare_episode(self, episode: Episode) -> Episode:
        score = score_episode_salience(episode)
        tier = assign_tier(
            score,
            deep_threshold=self.settings.deep_salience_threshold,
            medium_threshold=self.settings.medium_salience_threshold,
        )
        episode.salience = score
        episode.tier = tier

        tags = detect_pii_tags(episode.text_raw)
        episode.sensitivity_tags = tags
        if tags:
            episode.text_search = redact_sensitive_text(episode.text_search)
        return episode

    def _build_facts(self, episode: Episode, summary_bundle: SummaryBundle) -> list[FactRecord]:
        facts: list[FactRecord] = []
        for item in summary_bundle.facts:
            fact_text = str(item.get("fact", "")).strip()
            if not fact_text:
                continue
            category = str(item.get("category", "event")).strip() or "event"
            confidence = float(item.get("confidence", 0.65))
            facts.append(
                FactRecord(
                    fact_id=_hash_id("fact", f"{episode.episode_id}|{fact_text}"),
                    fact_text=fact_text,
                    category=category,
                    confidence=max(0.0, min(1.0, confidence)),
                    first_seen_at=episode.start_ts,
                    last_seen_at=episode.end_ts,
                    evidence_message_ids=episode.message_ids[:12],
                    source_episode_id=episode.episode_id,
                )
            )
        return facts

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()


def _week_key(dt: datetime) -> str:
    year, week, _weekday = dt.isocalendar()
    return f"{year}-W{week:02d}"


def _hash_id(prefix: str, payload: str) -> str:
    raw = f"{prefix}|{payload}".encode("utf-8")
    return hashlib.sha1(raw).hexdigest()


def _compact_unique(items: list[str], limit: int) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        clean = str(item).strip()
        if not clean or clean in seen:
            continue
        seen.add(clean)
        result.append(clean)
        if len(result) >= limit:
            break
    return result
