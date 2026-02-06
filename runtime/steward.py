from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
import logging
import os
import threading
import time
from typing import Any, Callable

from config.settings import Settings
from memory.ingestion import IngestionPipeline
from memory.storage.sqlite_store import SQLiteStore

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class StewardSnapshot:
    pending_deep: int
    pending_weekly: int
    running_ingestion_jobs: int


class MemorySteward:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.sqlite = SQLiteStore(settings.sqlite_path)
        self.pipeline = IngestionPipeline(settings)
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="memory-steward")
        self._lock = threading.Lock()
        self._active: dict[str, Future[Any]] = {}

        self.auto_bootstrap = _env_bool("STEWARD_AUTO_BOOTSTRAP", True)
        self.cooldown_seconds = _env_int("STEWARD_COOLDOWN_SECONDS", 45)
        self.enrich_batch = max(1, _env_int("STEWARD_ENRICH_BATCH", 8))
        self.consolidate_batch = max(1, _env_int("STEWARD_CONSOLIDATE_BATCH", 3))
        self.min_context_threshold = max(1, _env_int("STEWARD_MIN_CONTEXTS", 5))
        self.score_threshold = max(0.0, min(1.0, _env_float("STEWARD_SCORE_THRESHOLD", 0.58)))
        self.low_confidence_threshold = max(
            0.0,
            min(1.0, _env_float("STEWARD_LOW_CONFIDENCE_THRESHOLD", 0.45)),
        )

        self._last_action_at = 0.0
        self._bootstrap_submitted = False
        self._session_started = False

    def on_session_start(self) -> list[dict[str, Any]]:
        with self._lock:
            if self._session_started:
                return []
            self._session_started = True

        reports: list[dict[str, Any]] = []
        bootstrap_event = self.maybe_bootstrap(reason="chat_session_started")
        if bootstrap_event:
            reports.append(bootstrap_event)

        snapshot = self.snapshot()
        if snapshot.running_ingestion_jobs > 0:
            return reports

        if snapshot.pending_deep > 0:
            scheduled = self._submit(
                action="enrich_deep",
                fn=self._run_enrich_deep,
                reason="chat_session_started",
                bypass_cooldown=True,
            )
            if scheduled:
                reports.append(
                    {
                        "source": "memory_steward",
                        "action": "enrich_deep",
                        "scheduled": True,
                        "reason": "chat_session_started",
                    }
                )
            return reports

        if snapshot.pending_weekly > 0:
            scheduled = self._submit(
                action="consolidate_weekly",
                fn=self._run_consolidate_weekly,
                reason="chat_session_started",
                bypass_cooldown=True,
            )
            if scheduled:
                reports.append(
                    {
                        "source": "memory_steward",
                        "action": "consolidate_weekly",
                        "scheduled": True,
                        "reason": "chat_session_started",
                    }
                )
        return reports

    def maybe_bootstrap(self, reason: str = "chat_session_started") -> dict[str, Any] | None:
        if not self.auto_bootstrap:
            return None
        if self.sqlite.count_running_ingestion_jobs() > 0:
            return None
        with self._lock:
            if self._bootstrap_submitted:
                return None
            self._bootstrap_submitted = True
        scheduled = self._submit(
            action="fast_ingest",
            fn=self._run_fast_ingest,
            reason=reason,
            bypass_cooldown=True,
        )
        if not scheduled:
            return None
        return {"source": "memory_steward", "action": "fast_ingest", "scheduled": True, "reason": reason}

    def consider_retrieval(
        self,
        *,
        context_count: int,
        scores: list[float],
        recall_count: int,
        reason: str,
    ) -> list[dict[str, Any]]:
        snapshot = self.snapshot()
        reports: list[dict[str, Any]] = []

        if snapshot.running_ingestion_jobs > 0:
            return reports

        peak_score = max(scores) if scores else 0.0
        low_context = context_count < self.min_context_threshold
        weak_grounding = peak_score < self.score_threshold
        needs_boost = low_context or weak_grounding or recall_count > 0

        if snapshot.pending_deep > 0 and needs_boost:
            scheduled = self._submit(
                action="enrich_deep",
                fn=self._run_enrich_deep,
                reason=reason,
            )
            if scheduled:
                reports.append(
                    {
                        "source": "memory_steward",
                        "action": "enrich_deep",
                        "scheduled": True,
                        "reason": reason,
                        "pending_deep": snapshot.pending_deep,
                        "context_count": context_count,
                        "peak_score": round(peak_score, 3),
                    }
                )

        if snapshot.pending_weekly > 0 and (snapshot.pending_deep == 0 or recall_count > 0):
            scheduled = self._submit(
                action="consolidate_weekly",
                fn=self._run_consolidate_weekly,
                reason=reason,
            )
            if scheduled:
                reports.append(
                    {
                        "source": "memory_steward",
                        "action": "consolidate_weekly",
                        "scheduled": True,
                        "reason": reason,
                        "pending_weekly": snapshot.pending_weekly,
                    }
                )

        return reports

    def consider_answer_confidence(
        self,
        *,
        confidence_label: str,
        confidence_score: float,
        context_count: int,
        recall_count: int,
        signals: list[str],
        reason: str,
    ) -> list[dict[str, Any]]:
        snapshot = self.snapshot()
        reports: list[dict[str, Any]] = []
        if snapshot.running_ingestion_jobs > 0:
            return reports

        is_low = confidence_label == "low" or confidence_score <= self.low_confidence_threshold
        if is_low and snapshot.pending_deep > 0:
            scheduled = self._submit(
                action="enrich_deep",
                fn=self._run_enrich_deep,
                reason=reason,
            )
            if scheduled:
                reports.append(
                    {
                        "source": "memory_steward",
                        "action": "enrich_deep",
                        "scheduled": True,
                        "reason": reason,
                        "trigger": "low_confidence_answer",
                        "confidence_label": confidence_label,
                        "confidence_score": round(confidence_score, 3),
                        "context_count": context_count,
                        "recall_count": recall_count,
                        "pending_deep": snapshot.pending_deep,
                        "signals": signals[:6],
                    }
                )

        should_consolidate = (
            snapshot.pending_weekly > 0 and (is_low or (confidence_label == "medium" and recall_count > 0))
        )
        if should_consolidate and snapshot.pending_deep == 0:
            scheduled = self._submit(
                action="consolidate_weekly",
                fn=self._run_consolidate_weekly,
                reason=reason,
            )
            if scheduled:
                reports.append(
                    {
                        "source": "memory_steward",
                        "action": "consolidate_weekly",
                        "scheduled": True,
                        "reason": reason,
                        "trigger": "low_confidence_answer",
                        "pending_weekly": snapshot.pending_weekly,
                    }
                )

        return reports

    def snapshot(self) -> StewardSnapshot:
        return StewardSnapshot(
            pending_deep=self.sqlite.count_unenriched_deep_episodes(),
            pending_weekly=self.sqlite.count_pending_weekly_groups(),
            running_ingestion_jobs=self.sqlite.count_running_ingestion_jobs(),
        )

    def _submit(
        self,
        *,
        action: str,
        fn: Callable[[], None],
        reason: str,
        bypass_cooldown: bool = False,
    ) -> bool:
        with self._lock:
            self._cleanup_locked()

            now = time.monotonic()
            if not bypass_cooldown and now - self._last_action_at < self.cooldown_seconds:
                return False

            existing = self._active.get(action)
            if existing is not None and not existing.done():
                return False

            future = self.executor.submit(self._run_action, action, fn, reason)
            self._active[action] = future
            self._last_action_at = now
            return True

    def _run_action(self, action: str, fn: Callable[[], None], reason: str) -> None:
        try:
            logger.info("MemorySteward running action=%s reason=%s", action, reason)
            fn()
        except Exception:
            logger.exception("MemorySteward action failed: %s", action)

    def _cleanup_locked(self) -> None:
        done = [name for name, fut in self._active.items() if fut.done()]
        for name in done:
            self._active.pop(name, None)

    def _run_fast_ingest(self) -> None:
        self.pipeline.ingest_all(force=False)
        # One small autonomous follow-up pass keeps chat quality improving
        # without turning this into a permanent background process.
        deep_stats = self.pipeline.enrich_deep_episodes(limit=self.enrich_batch)
        if deep_stats.deep_episodes_enriched == 0:
            self.pipeline.consolidate_weekly_summaries(limit=self.consolidate_batch)

    def _run_enrich_deep(self) -> None:
        self.pipeline.enrich_deep_episodes(limit=self.enrich_batch)

    def _run_consolidate_weekly(self) -> None:
        self.pipeline.consolidate_weekly_summaries(limit=self.consolidate_batch)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default
