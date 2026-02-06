from __future__ import annotations

from runtime.graph import (
    _extract_explicit_self_aliases,
    _guard_self_as_contact_answer,
    _infer_answer_confidence,
    _is_followup_query,
    _query_is_contact_ranking,
    _sanitize_trace_for_user,
    _token_overlap_ratio,
    _user_reflection_note,
)


def test_answer_confidence_low_on_sparse_uncertain_output() -> None:
    confidence = _infer_answer_confidence(
        payload=None,
        answer="I am not sure about this from memory.",
        contexts=[],
        recall_count=1,
        max_recall_count=3,
        parse_ok=False,
        needs_recall=False,
    )
    assert confidence["label"] == "low"
    assert float(confidence["score"]) < 0.45


def test_answer_confidence_high_with_strong_grounding_and_model_signal() -> None:
    contexts = [{"score": 0.92} for _ in range(8)]
    confidence = _infer_answer_confidence(
        payload={"confidence_label": "high", "confidence": 0.9},
        answer="You discussed this with Pranav and shared two options.",
        contexts=contexts,
        recall_count=0,
        max_recall_count=3,
        parse_ok=True,
        needs_recall=False,
    )
    assert confidence["label"] == "high"
    assert float(confidence["score"]) >= 0.72


def test_user_reflection_note_low_confidence() -> None:
    note = _user_reflection_note(confidence_label="low", recall_count=2, needs_recall=True)
    assert note == "I might need to reflect on this. I'll come back stronger."


def test_trace_sanitization_hides_job_names_and_scores() -> None:
    raw_trace = [
        {"source": "memory_steward", "action": "enrich_deep", "scheduled": True},
        {"source": "answer_confidence", "label": "low", "score": 0.32, "signals": ["x"]},
        {"source": "raw_episodes", "id": "e1", "score": 0.7},
    ]
    sanitized = _sanitize_trace_for_user(raw_trace)
    assert sanitized[0]["source"] == "memory_steward"
    assert "action" not in sanitized[0]
    assert "note" in sanitized[0]
    assert sanitized[1]["source"] == "answer_confidence"
    assert "score" not in sanitized[1]
    assert sanitized[1]["confidence"] == "low"


def test_followup_query_detection() -> None:
    assert _is_followup_query("what about him")
    assert _is_followup_query("and in that case")
    assert not _is_followup_query("in reposting baddies who is most active")


def test_token_overlap_ratio() -> None:
    ratio = _token_overlap_ratio("who is most active there", "who is active in that chat")
    assert ratio > 0.25


def test_extract_explicit_self_aliases_from_user_statement() -> None:
    aliases = _extract_explicit_self_aliases("Rohit Kumar is me myself.")
    assert "Rohit Kumar" in aliases


def test_guard_self_as_contact_answer_for_contact_ranking_query() -> None:
    guarded, changed = _guard_self_as_contact_answer(
        query="who is my best friend amongst all my contacts",
        answer="Rohit Kumar is your closest contact based on interactions.",
        self_aliases=["Rohit Kumar"],
    )
    assert _query_is_contact_ranking("who is my best friend amongst all my contacts")
    assert changed is True
    assert "won't count Rohit Kumar as a contact" in guarded
