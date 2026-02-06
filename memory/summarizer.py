from __future__ import annotations

from dataclasses import dataclass
import json

from memory.llm_client import OpenRouterLLM
from memory.models import Episode


@dataclass(slots=True)
class SummaryBundle:
    summary_text: str
    confidence: float
    sentiment: str
    facts: list[dict[str, object]]


def _extractive_summary(text: str, max_lines: int = 8, max_chars: int = 700) -> str:
    lines = []
    for line in text.splitlines():
        clean = line.strip()
        if not clean:
            continue
        if "<Media omitted>" in clean:
            continue
        lines.append(clean)
        if len(lines) >= max_lines:
            break

    summary = " ".join(lines)
    if not summary:
        summary = text[:max_chars]
    if len(summary) > max_chars:
        summary = summary[: max_chars - 3] + "..."
    return summary


class EpisodeSummarizer:
    def __init__(self, llm: OpenRouterLLM | None) -> None:
        self.llm = llm

    def summarize(self, episode: Episode) -> SummaryBundle:
        if episode.tier == "pass_through" or self.llm is None or not self.llm.enabled:
            return SummaryBundle(
                summary_text=_extractive_summary(episode.text_raw),
                confidence=0.55,
                sentiment="neutral",
                facts=[],
            )

        system_prompt = (
            "You summarize personal chat episodes for long-term memory retrieval. "
            "Return strict JSON only."
        )
        user_prompt = (
            "Summarize this chat episode for semantic memory retrieval. "
            "Keep concise and factual.\n"
            "Extract up to 5 grounded facts with confidence and category.\n"
            "JSON schema:\n"
            "{\n"
            '  "summary_text": "...",\n'
            '  "sentiment": "neutral|positive|negative|mixed",\n'
            '  "confidence": 0.0,\n'
            '  "facts": [\n'
            "    {\"fact\": \"...\", \"category\": \"identity|preference|relationship|event|goal\", \"confidence\": 0.0}\n"
            "  ]\n"
            "}\n\n"
            f"EPISODE:\n{episode.text_raw}"
        )

        payload = self.llm.complete_json(system_prompt, user_prompt, max_tokens=1200)
        if not payload:
            return SummaryBundle(
                summary_text=_extractive_summary(episode.text_raw),
                confidence=0.5,
                sentiment="neutral",
                facts=[],
            )

        summary_text = str(payload.get("summary_text", "")).strip() or _extractive_summary(
            episode.text_raw
        )
        sentiment = str(payload.get("sentiment", "neutral")).strip().lower()
        confidence = _coerce_confidence(payload.get("confidence", 0.65))

        facts_raw = payload.get("facts", [])
        facts: list[dict[str, object]] = []
        if isinstance(facts_raw, list):
            for item in facts_raw[:5]:
                if not isinstance(item, dict):
                    continue
                fact_text = str(item.get("fact", "")).strip()
                if not fact_text:
                    continue
                category = str(item.get("category", "event")).strip().lower()
                fact_conf = _coerce_confidence(item.get("confidence", 0.65))
                facts.append(
                    {
                        "fact": fact_text,
                        "category": category,
                        "confidence": fact_conf,
                    }
                )

        return SummaryBundle(
            summary_text=summary_text,
            confidence=confidence,
            sentiment=sentiment,
            facts=facts,
        )


def _coerce_confidence(value: object) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        score = 0.65
    return max(0.0, min(1.0, score))


def make_weekly_summary(
    week_key: str,
    summaries: list[str],
    llm: OpenRouterLLM | None,
) -> str:
    if not summaries:
        return ""

    joined = "\n".join(f"- {item}" for item in summaries[:40])

    if llm is None or not llm.enabled:
        return _extractive_summary(joined, max_lines=10, max_chars=850)

    system_prompt = "You consolidate weekly memory summaries. Return short plain text."
    user_prompt = (
        f"Week: {week_key}\n"
        "Create one dense weekly summary for retrieval. Keep it factual and non-redundant.\n"
        f"SUMMARIES:\n{joined}"
    )
    text = llm.complete(system_prompt, user_prompt, max_tokens=500)
    clean = text.strip()
    return clean if clean else _extractive_summary(joined, max_lines=10, max_chars=850)
