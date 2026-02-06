from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import chromadb


@dataclass(slots=True)
class ChromaHit:
    doc_id: str
    text: str
    metadata: dict[str, Any]
    score: float


class ChromaStore:
    def __init__(self, persist_path: str | Path) -> None:
        self.persist_path = Path(persist_path)
        self.persist_path.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(path=str(self.persist_path))
        self.weekly = self.client.get_or_create_collection("weekly_summaries")
        self.raw = self.client.get_or_create_collection("raw_episodes")

    def add_raw_episode(self, episode_id: str, text: str, metadata: dict[str, Any]) -> None:
        self.add_raw_episodes([(episode_id, text, metadata)])

    def add_raw_episodes(self, episodes: list[tuple[str, str, dict[str, Any]]]) -> None:
        if not episodes:
            return

        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict[str, Any]] = []
        for episode_id, text, metadata in episodes:
            ids.append(episode_id)
            documents.append(text)
            metadatas.append(self._normalize_metadata(metadata))

        self.raw.upsert(ids=ids, documents=documents, metadatas=metadatas)

    def add_weekly_summary(self, summary_id: str, text: str, metadata: dict[str, Any]) -> None:
        self.weekly.upsert(
            ids=[summary_id],
            documents=[text],
            metadatas=[self._normalize_metadata(metadata)],
        )

    def query_raw(
        self,
        query: str,
        n_results: int = 12,
        where: dict[str, Any] | None = None,
    ) -> list[ChromaHit]:
        return self._query_collection(self.raw, query, n_results=n_results, where=where)

    def query_weekly(
        self,
        query: str,
        n_results: int = 8,
        where: dict[str, Any] | None = None,
    ) -> list[ChromaHit]:
        return self._query_collection(self.weekly, query, n_results=n_results, where=where)

    def _query_collection(
        self,
        collection: Any,
        query: str,
        n_results: int,
        where: dict[str, Any] | None = None,
    ) -> list[ChromaHit]:
        try:
            payload = collection.query(query_texts=[query], n_results=n_results, where=where)
        except Exception:
            return []

        ids = payload.get("ids", [[]])
        docs = payload.get("documents", [[]])
        metas = payload.get("metadatas", [[]])
        dists = payload.get("distances", [[]])

        rows = []
        for index, doc_id in enumerate(ids[0] if ids else []):
            text = docs[0][index] if docs and docs[0] else ""
            metadata = metas[0][index] if metas and metas[0] else {}
            dist = dists[0][index] if dists and dists[0] else 0.0
            score = 1.0 / (1.0 + float(dist))
            rows.append(
                ChromaHit(
                    doc_id=str(doc_id),
                    text=str(text),
                    metadata=dict(metadata or {}),
                    score=score,
                )
            )
        return rows

    def _normalize_metadata(self, metadata: dict[str, Any]) -> dict[str, Any]:
        normalized: dict[str, Any] = {}
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)):
                normalized[key] = value
            elif value is None:
                normalized[key] = ""
            else:
                normalized[key] = json.dumps(value, ensure_ascii=False)
        return normalized
