from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

from memory.models import Episode, SummaryRecord
from memory.storage.sqlite_store import SQLiteStore


def _episode(episode_id: str, *, tier: str, salience: float, offset_hours: int) -> Episode:
    start = datetime(2025, 1, 1, 9, 0, 0) + timedelta(hours=offset_hours)
    end = start + timedelta(minutes=45)
    return Episode(
        episode_id=episode_id,
        chat_id="chat_alpha",
        start_ts=start,
        end_ts=end,
        message_ids=[f"{episode_id}_m1", f"{episode_id}_m2"],
        text_raw=f"raw {episode_id}",
        text_search=f"search {episode_id}",
        salience=salience,
        tier=tier,
    )


def test_fetch_unenriched_deep_episodes(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "memory.db")
    deep_pending = _episode("deep_pending", tier="deep", salience=0.93, offset_hours=0)
    deep_done = _episode("deep_done", tier="deep", salience=0.88, offset_hours=3)
    medium = _episode("medium", tier="medium", salience=0.62, offset_hours=6)

    store.upsert_episodes([deep_pending, deep_done, medium])
    store.insert_summary(
        SummaryRecord(
            summary_id="episodic_deep_done",
            episode_id=deep_done.episode_id,
            chat_id=deep_done.chat_id,
            week_key="2025-W01",
            summary_text="already processed",
            summary_type="episodic",
            confidence=0.8,
            evidence_message_ids=[deep_done.message_ids[0]],
        )
    )

    pending = store.fetch_unenriched_deep_episodes(limit=10)
    assert [episode.episode_id for episode in pending] == ["deep_pending"]


def test_pending_weekly_groups_and_fetch_episodic_rows(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "memory.db")

    store.insert_summary(
        SummaryRecord(
            summary_id="ep_w05_1",
            episode_id="e1",
            chat_id="chat_alpha",
            week_key="2025-W05",
            summary_text="first",
            summary_type="episodic",
            confidence=0.91,
            evidence_message_ids=["m1", "m2"],
        )
    )
    store.insert_summary(
        SummaryRecord(
            summary_id="ep_w05_2",
            episode_id="e2",
            chat_id="chat_alpha",
            week_key="2025-W05",
            summary_text="second",
            summary_type="episodic",
            confidence=0.72,
            evidence_message_ids=["m3"],
        )
    )
    store.insert_summary(
        SummaryRecord(
            summary_id="ep_w06_1",
            episode_id="e3",
            chat_id="chat_alpha",
            week_key="2025-W06",
            summary_text="third",
            summary_type="episodic",
            confidence=0.66,
            evidence_message_ids=["m4"],
        )
    )
    store.insert_summary(
        SummaryRecord(
            summary_id="wk_w06",
            episode_id=None,
            chat_id="chat_alpha",
            week_key="2025-W06",
            summary_text="weekly done",
            summary_type="weekly",
            confidence=0.7,
            evidence_message_ids=["m4"],
        )
    )

    pending = store.fetch_pending_weekly_groups(limit=10)
    assert len(pending) == 1
    assert pending[0]["chat_id"] == "chat_alpha"
    assert pending[0]["week_key"] == "2025-W05"
    assert int(pending[0]["episodic_count"]) == 2

    rows = store.fetch_episodic_summaries_for_week("chat_alpha", "2025-W05")
    assert [row["id"] for row in rows] == ["ep_w05_1", "ep_w05_2"]
    assert rows[0]["evidence_message_ids"] == ["m1", "m2"]
