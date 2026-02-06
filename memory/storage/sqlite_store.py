from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from datetime import datetime
import json
from pathlib import Path
import sqlite3
from typing import Any, Iterable, Iterator, Sequence

from memory.models import ChatMessage, Episode, FactRecord, SummaryRecord


class SQLiteStore:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._fts_enabled = False
        self._initialize()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(str(self.db_path), timeout=30)
        conn.execute("PRAGMA busy_timeout=30000;")
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _initialize(self) -> None:
        with self.connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    chat_id TEXT NOT NULL,
                    sender TEXT,
                    timestamp TEXT NOT NULL,
                    content_raw TEXT NOT NULL,
                    content_search TEXT NOT NULL,
                    is_system INTEGER NOT NULL DEFAULT 0,
                    has_media INTEGER NOT NULL DEFAULT 0,
                    sensitivity_tags TEXT NOT NULL DEFAULT '[]',
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                CREATE INDEX IF NOT EXISTS idx_messages_chat_ts ON messages(chat_id, timestamp);
                CREATE INDEX IF NOT EXISTS idx_messages_sender ON messages(sender);

                CREATE TABLE IF NOT EXISTS episodes (
                    id TEXT PRIMARY KEY,
                    chat_id TEXT NOT NULL,
                    start_ts TEXT NOT NULL,
                    end_ts TEXT NOT NULL,
                    message_count INTEGER NOT NULL,
                    message_ids TEXT NOT NULL,
                    text_raw TEXT NOT NULL,
                    text_search TEXT NOT NULL,
                    salience REAL NOT NULL,
                    tier TEXT NOT NULL,
                    sensitivity_tags TEXT NOT NULL DEFAULT '[]',
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                CREATE INDEX IF NOT EXISTS idx_episodes_chat_ts ON episodes(chat_id, end_ts);
                CREATE INDEX IF NOT EXISTS idx_episodes_salience ON episodes(salience DESC);

                CREATE TABLE IF NOT EXISTS summaries (
                    id TEXT PRIMARY KEY,
                    episode_id TEXT,
                    chat_id TEXT NOT NULL,
                    week_key TEXT NOT NULL,
                    summary_text TEXT NOT NULL,
                    summary_type TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    evidence_message_ids TEXT NOT NULL DEFAULT '[]',
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                CREATE INDEX IF NOT EXISTS idx_summaries_week ON summaries(week_key);
                CREATE INDEX IF NOT EXISTS idx_summaries_chat ON summaries(chat_id);

                CREATE TABLE IF NOT EXISTS facts (
                    id TEXT PRIMARY KEY,
                    fact_text TEXT NOT NULL,
                    category TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    first_seen_at TEXT NOT NULL,
                    last_seen_at TEXT NOT NULL,
                    evidence_message_ids TEXT NOT NULL DEFAULT '[]',
                    source_episode_id TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                CREATE INDEX IF NOT EXISTS idx_facts_conf ON facts(confidence DESC);

                CREATE TABLE IF NOT EXISTS ingestion_jobs (
                    file_path TEXT PRIMARY KEY,
                    file_hash TEXT NOT NULL,
                    status TEXT NOT NULL,
                    processed_messages INTEGER NOT NULL DEFAULT 0,
                    processed_episodes INTEGER NOT NULL DEFAULT 0,
                    last_error TEXT,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS retrieval_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query TEXT NOT NULL,
                    retrieved_summary_ids TEXT NOT NULL DEFAULT '[]',
                    retrieved_episode_ids TEXT NOT NULL DEFAULT '[]',
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                """
            )

            try:
                conn.execute(
                    """
                    CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts
                    USING fts5(message_id UNINDEXED, chat_id UNINDEXED, content);
                    """
                )
                conn.execute(
                    """
                    CREATE VIRTUAL TABLE IF NOT EXISTS episodes_fts
                    USING fts5(episode_id UNINDEXED, chat_id UNINDEXED, content);
                    """
                )
                self._fts_enabled = True
            except sqlite3.OperationalError:
                self._fts_enabled = False

            conn.commit()

    def get_ingestion_job(self, file_path: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT file_path, file_hash, status, processed_messages, processed_episodes, last_error, updated_at "
                "FROM ingestion_jobs WHERE file_path = ?",
                (file_path,),
            ).fetchone()
            return dict(row) if row else None

    def upsert_ingestion_job(
        self,
        file_path: str,
        file_hash: str,
        status: str,
        processed_messages: int,
        processed_episodes: int,
        last_error: str | None = None,
    ) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO ingestion_jobs(file_path, file_hash, status, processed_messages, processed_episodes, last_error, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(file_path) DO UPDATE SET
                    file_hash = excluded.file_hash,
                    status = excluded.status,
                    processed_messages = excluded.processed_messages,
                    processed_episodes = excluded.processed_episodes,
                    last_error = excluded.last_error,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (file_path, file_hash, status, processed_messages, processed_episodes, last_error),
            )
            conn.commit()

    def upsert_messages(
        self,
        messages: Sequence[ChatMessage],
        search_by_id: dict[str, str],
        tags_by_id: dict[str, list[str]],
    ) -> None:
        rows = []
        fts_rows = []
        for msg in messages:
            tags = tags_by_id.get(msg.message_id, [])
            search_text = search_by_id.get(msg.message_id, msg.content_raw)
            rows.append(
                (
                    msg.message_id,
                    msg.chat_id,
                    msg.sender,
                    msg.timestamp.isoformat(),
                    msg.content_raw,
                    search_text,
                    int(msg.is_system),
                    int(msg.has_media),
                    json.dumps(tags, ensure_ascii=False),
                )
            )
            fts_rows.append((msg.message_id, msg.chat_id, search_text))

        with self.connect() as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO messages(
                    id, chat_id, sender, timestamp, content_raw, content_search,
                    is_system, has_media, sensitivity_tags
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )

            if self._fts_enabled and fts_rows:
                conn.executemany(
                    "DELETE FROM messages_fts WHERE message_id = ?",
                    [(item[0],) for item in fts_rows],
                )
                conn.executemany(
                    "INSERT INTO messages_fts(message_id, chat_id, content) VALUES (?, ?, ?)",
                    fts_rows,
                )
            conn.commit()

    def upsert_episode(self, episode: Episode) -> None:
        self.upsert_episodes([episode])

    def upsert_episodes(self, episodes: Sequence[Episode]) -> None:
        if not episodes:
            return

        rows = [
            (
                episode.episode_id,
                episode.chat_id,
                episode.start_ts.isoformat(),
                episode.end_ts.isoformat(),
                len(episode.message_ids),
                json.dumps(episode.message_ids, ensure_ascii=False),
                episode.text_raw,
                episode.text_search,
                float(episode.salience),
                episode.tier,
                json.dumps(episode.sensitivity_tags, ensure_ascii=False),
            )
            for episode in episodes
        ]
        fts_rows = [
            (episode.episode_id, episode.chat_id, episode.text_search) for episode in episodes
        ]

        with self.connect() as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO episodes(
                    id, chat_id, start_ts, end_ts, message_count, message_ids,
                    text_raw, text_search, salience, tier, sensitivity_tags
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )

            if self._fts_enabled and fts_rows:
                conn.executemany(
                    "DELETE FROM episodes_fts WHERE episode_id = ?",
                    [(item[0],) for item in fts_rows],
                )
                conn.executemany(
                    "INSERT INTO episodes_fts(episode_id, chat_id, content) VALUES (?, ?, ?)",
                    fts_rows,
                )

            conn.commit()

    def insert_summary(self, summary: SummaryRecord) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO summaries(
                    id, episode_id, chat_id, week_key, summary_text,
                    summary_type, confidence, evidence_message_ids
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    summary.summary_id,
                    summary.episode_id,
                    summary.chat_id,
                    summary.week_key,
                    summary.summary_text,
                    summary.summary_type,
                    summary.confidence,
                    json.dumps(summary.evidence_message_ids, ensure_ascii=False),
                ),
            )
            conn.commit()

    def insert_facts(self, facts: Sequence[FactRecord]) -> None:
        if not facts:
            return

        rows = [
            (
                fact.fact_id,
                fact.fact_text,
                fact.category,
                float(fact.confidence),
                fact.first_seen_at.isoformat(),
                fact.last_seen_at.isoformat(),
                json.dumps(fact.evidence_message_ids, ensure_ascii=False),
                fact.source_episode_id,
            )
            for fact in facts
        ]

        with self.connect() as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO facts(
                    id, fact_text, category, confidence, first_seen_at,
                    last_seen_at, evidence_message_ids, source_episode_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            conn.commit()

    def fetch_unenriched_deep_episodes(self, limit: int = 50) -> list[Episode]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT e.id, e.chat_id, e.start_ts, e.end_ts, e.message_ids,
                       e.text_raw, e.text_search, e.salience, e.tier, e.sensitivity_tags
                FROM episodes e
                LEFT JOIN summaries s
                    ON s.episode_id = e.id AND s.summary_type = 'episodic'
                WHERE e.tier = 'deep' AND s.id IS NULL
                ORDER BY e.salience DESC, e.end_ts DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

        episodes: list[Episode] = []
        for row in rows:
            episodes.append(
                Episode(
                    episode_id=str(row["id"]),
                    chat_id=str(row["chat_id"]),
                    start_ts=datetime.fromisoformat(str(row["start_ts"])),
                    end_ts=datetime.fromisoformat(str(row["end_ts"])),
                    message_ids=_as_string_list(row["message_ids"]),
                    text_raw=str(row["text_raw"]),
                    text_search=str(row["text_search"]),
                    salience=float(row["salience"]),
                    tier=str(row["tier"]),
                    sensitivity_tags=_as_string_list(row["sensitivity_tags"]),
                )
            )
        return episodes

    def count_unenriched_deep_episodes(self) -> int:
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT COUNT(*) AS count
                FROM episodes e
                LEFT JOIN summaries s
                    ON s.episode_id = e.id AND s.summary_type = 'episodic'
                WHERE e.tier = 'deep' AND s.id IS NULL
                """
            ).fetchone()
        if not row:
            return 0
        return int(row["count"])

    def fetch_pending_weekly_groups(self, limit: int = 20) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT s.chat_id, s.week_key, COUNT(*) AS episodic_count
                FROM summaries s
                WHERE s.summary_type = 'episodic'
                  AND NOT EXISTS (
                    SELECT 1
                    FROM summaries w
                    WHERE w.summary_type = 'weekly'
                      AND w.chat_id = s.chat_id
                      AND w.week_key = s.week_key
                  )
                GROUP BY s.chat_id, s.week_key
                ORDER BY s.week_key DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def count_pending_weekly_groups(self) -> int:
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT COUNT(*) AS count
                FROM (
                    SELECT s.chat_id, s.week_key
                    FROM summaries s
                    WHERE s.summary_type = 'episodic'
                      AND NOT EXISTS (
                        SELECT 1
                        FROM summaries w
                        WHERE w.summary_type = 'weekly'
                          AND w.chat_id = s.chat_id
                          AND w.week_key = s.week_key
                      )
                    GROUP BY s.chat_id, s.week_key
                ) pending
                """
            ).fetchone()
        if not row:
            return 0
        return int(row["count"])

    def count_running_ingestion_jobs(self) -> int:
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT COUNT(*) AS count
                FROM ingestion_jobs
                WHERE status = 'running'
                """
            ).fetchone()
        if not row:
            return 0
        return int(row["count"])

    def fetch_episodic_summaries_for_week(
        self,
        chat_id: str,
        week_key: str,
        limit: int = 120,
    ) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT id, episode_id, summary_text, confidence, evidence_message_ids
                FROM summaries
                WHERE summary_type = 'episodic' AND chat_id = ? AND week_key = ?
                ORDER BY confidence DESC, created_at DESC
                LIMIT ?
                """,
                (chat_id, week_key, limit),
            ).fetchall()

        results: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            item["evidence_message_ids"] = _as_string_list(item.get("evidence_message_ids"))
            results.append(item)
        return results

    def fetch_episodes_by_ids(self, episode_ids: Sequence[str]) -> list[dict[str, Any]]:
        if not episode_ids:
            return []

        placeholders = ",".join("?" for _ in episode_ids)
        query = (
            "SELECT id, chat_id, start_ts, end_ts, text_raw, text_search, salience, tier, sensitivity_tags "
            f"FROM episodes WHERE id IN ({placeholders})"
        )

        with self.connect() as conn:
            rows = conn.execute(query, tuple(episode_ids)).fetchall()

        by_id = {row["id"]: dict(row) for row in rows}
        return [by_id[item] for item in episode_ids if item in by_id]

    def list_chat_ids(self, limit: int = 500) -> list[str]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT chat_id, COUNT(*) AS c
                FROM episodes
                GROUP BY chat_id
                ORDER BY c DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [str(row["chat_id"]) for row in rows]

    def search_episodes_lexical(
        self,
        query: str,
        limit: int = 12,
        chat_id: str | None = None,
    ) -> list[dict[str, Any]]:
        with self.connect() as conn:
            if self._fts_enabled:
                try:
                    if chat_id:
                        rows = conn.execute(
                            """
                            SELECT e.id, e.chat_id, e.start_ts, e.end_ts, e.text_raw, e.salience,
                                   bm25(episodes_fts) AS rank
                            FROM episodes_fts
                            JOIN episodes e ON e.id = episodes_fts.episode_id
                            WHERE episodes_fts MATCH ? AND e.chat_id = ?
                            ORDER BY rank
                            LIMIT ?
                            """,
                            (query, chat_id, limit),
                        ).fetchall()
                    else:
                        rows = conn.execute(
                            """
                            SELECT e.id, e.chat_id, e.start_ts, e.end_ts, e.text_raw, e.salience,
                                   bm25(episodes_fts) AS rank
                            FROM episodes_fts
                            JOIN episodes e ON e.id = episodes_fts.episode_id
                            WHERE episodes_fts MATCH ?
                            ORDER BY rank
                            LIMIT ?
                            """,
                            (query, limit),
                        ).fetchall()
                    return [
                        {
                            **dict(row),
                            "score": 1.0 / (1.0 + max(float(row["rank"]), 0.0)),
                        }
                        for row in rows
                    ]
                except sqlite3.OperationalError:
                    pass

            like = f"%{query}%"
            if chat_id:
                rows = conn.execute(
                    """
                    SELECT id, chat_id, start_ts, end_ts, text_raw, salience
                    FROM episodes
                    WHERE chat_id = ? AND text_search LIKE ?
                    ORDER BY salience DESC, end_ts DESC
                    LIMIT ?
                    """,
                    (chat_id, like, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT id, chat_id, start_ts, end_ts, text_raw, salience
                    FROM episodes
                    WHERE text_search LIKE ?
                    ORDER BY salience DESC, end_ts DESC
                    LIMIT ?
                    """,
                    (like, limit),
                ).fetchall()
            return [{**dict(row), "score": 0.25} for row in rows]

    def fetch_top_contacts(self, limit: int = 25) -> list[tuple[str, int]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT sender, COUNT(*) AS count
                FROM messages
                WHERE sender IS NOT NULL
                GROUP BY sender
                ORDER BY count DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [(str(row["sender"]), int(row["count"])) for row in rows]

    def fetch_top_facts(self, limit: int = 80) -> list[tuple[str, float]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT fact_text, AVG(confidence) AS score
                FROM facts
                GROUP BY fact_text
                ORDER BY score DESC, COUNT(*) DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [(str(row["fact_text"]), float(row["score"])) for row in rows]

    def fetch_recent_message_samples(self, limit: int = 1200) -> list[str]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT content_raw
                FROM messages
                WHERE is_system = 0
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [str(row["content_raw"]) for row in rows]

    def detect_languages(self, limit: int = 5000) -> list[str]:
        samples = self.fetch_recent_message_samples(limit=limit)
        counts: Counter[str] = Counter()
        for text in samples:
            has_devanagari = any("\u0900" <= ch <= "\u097F" for ch in text)
            has_bengali = any("\u0980" <= ch <= "\u09FF" for ch in text)
            has_latin = any(("A" <= ch <= "Z") or ("a" <= ch <= "z") for ch in text)
            if has_latin:
                counts["Latin"] += 1
            if has_devanagari:
                counts["Devanagari"] += 1
            if has_bengali:
                counts["Bengali"] += 1

        if not counts:
            return ["Unknown"]
        return [name for name, _ in counts.most_common(3)]

    def log_retrieval(
        self,
        query: str,
        summary_ids: Sequence[str],
        episode_ids: Sequence[str],
    ) -> None:
        try:
            with self.connect() as conn:
                conn.execute(
                    """
                    INSERT INTO retrieval_logs(query, retrieved_summary_ids, retrieved_episode_ids)
                    VALUES (?, ?, ?)
                    """,
                    (
                        query,
                        json.dumps(list(summary_ids), ensure_ascii=False),
                        json.dumps(list(episode_ids), ensure_ascii=False),
                    ),
                )
                conn.commit()
        except sqlite3.OperationalError:
            # Best-effort telemetry; never fail the user response path.
            return


def _as_string_list(raw: object) -> list[str]:
    if isinstance(raw, str):
        stripped = raw.strip()
        if not stripped:
            return []
        try:
            decoded = json.loads(stripped)
        except json.JSONDecodeError:
            return [stripped]
    elif isinstance(raw, Iterable):
        decoded = list(raw)
    else:
        return []

    if not isinstance(decoded, list):
        return []
    return [str(item) for item in decoded]
