from __future__ import annotations

from dataclasses import dataclass
import re
import time
from typing import Any, Callable, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from config.settings import Settings, get_settings
from memory.llm_client import try_parse_json
from runtime.policy import refusal_for_sensitive_query, sanitize_answer, sensitive_query_reason
from runtime.retrieval import MemoryRetriever, RetrievalPack
from runtime.steward import MemorySteward


class ChatState(TypedDict, total=False):
    user_query: str
    history: list[dict[str, str]]
    query_intent: str
    session_user_identity: str
    profile_text: str
    contexts: list[dict[str, Any]]
    trace: list[dict[str, Any]]
    retrieved_episode_ids: list[str]
    retrieved_summary_ids: list[str]
    recall_count: int
    max_recall_count: int
    pending_recall_query: str
    final_answer: str
    answer_confidence_label: str
    answer_confidence_score: float
    working_memory_tokens_est: int
    profile_tokens_est: int
    history_tokens_est: int
    query_tokens_est: int
    turn_prompt_tokens_est: int
    context_window_tokens: int
    turn_prompt_pct: float
    prompt_tokens_est: int
    prompt_budget_tokens: int
    prompt_budget_pct: float
    retrieved_context_count: int
    completion_tokens_est: int
    retrieval_scope_mode: str
    retrieval_scope_chat_id: str


@dataclass(slots=True)
class _RetrievalCacheEntry:
    query_norm: str
    created_at: float
    result: RetrievalPack


class LangGraphChatEngine:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        if not self.settings.openrouter_api_key:
            raise RuntimeError("OPENROUTER_API_KEY is required for chat runtime.")

        self.retriever = MemoryRetriever(self.settings)
        self.model = ChatOpenAI(
            model=self.settings.model_name,
            api_key=self.settings.openrouter_api_key,
            base_url=self.settings.openrouter_base_url,
            temperature=self.settings.model_temperature,
            timeout=90,
            default_headers={
                "HTTP-Referer": "https://localhost/truffle-memory",
                "X-Title": "TruffleMemory",
            },
        )
        self.steward = MemorySteward(self.settings)
        self._session_start_trace = self.steward.on_session_start()
        self._stream_mode = False
        self._stream_callback: Callable[[str], None] | None = None
        self._retrieval_cache: list[_RetrievalCacheEntry] = []
        self._retrieval_cache_ttl_seconds = 90.0
        self.graph = self._build_graph()

    def _build_graph(self):
        builder = StateGraph(ChatState)
        builder.add_node("load_initial", self._node_load_initial)
        builder.add_node("answer", self._node_answer)
        builder.add_node("recall", self._node_recall)

        builder.add_edge(START, "load_initial")
        builder.add_edge("load_initial", "answer")
        builder.add_conditional_edges(
            "answer",
            self._route_after_answer,
            {
                "recall": "recall",
                "done": END,
            },
        )
        builder.add_edge("recall", "answer")
        return builder.compile()

    def chat(
        self,
        user_query: str,
        history: list[dict[str, str]] | None = None,
        *,
        stream: bool = False,
        on_token: Callable[[str], None] | None = None,
    ) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
        self._stream_mode = bool(stream)
        self._stream_callback = on_token if stream else None
        effective_max_recall = int(self.settings.max_recall_count)
        if self._stream_mode:
            # Keep demos responsive while memory warms up.
            running_jobs = self.steward.snapshot().running_ingestion_jobs
            if running_jobs > 0:
                effective_max_recall = 0
            else:
                effective_max_recall = min(effective_max_recall, 1)

        initial_state: ChatState = {
            "user_query": user_query,
            "history": history or [],
            "recall_count": 0,
            "max_recall_count": effective_max_recall,
            "pending_recall_query": "",
            "trace": self._consume_session_start_trace(),
        }
        try:
            final_state = self.graph.invoke(initial_state)
        finally:
            self._stream_mode = False
            self._stream_callback = None
        answer = final_state.get("final_answer", "I could not produce an answer.")
        trace = _sanitize_trace_for_user(final_state.get("trace", []))
        metrics = _extract_metrics(final_state, answer=str(answer), streaming=stream)
        return str(answer), trace, metrics

    def _consume_session_start_trace(self) -> list[dict[str, Any]]:
        events = list(self._session_start_trace)
        self._session_start_trace = []
        return events

    def _node_load_initial(self, state: ChatState) -> ChatState:
        cache_event: dict[str, Any] | None = None
        query_intent = _classify_query_intent(state["user_query"])
        session_identity = _extract_session_user_identity(
            history=state.get("history", []),
            current_query=state["user_query"],
        )
        result, cache_event = self._maybe_get_cached_retrieval(
            query=state["user_query"],
            history=state.get("history", []),
        )
        if result is None:
            result = self.retriever.retrieve(
                state["user_query"],
                token_budget=self.settings.initial_token_budget,
                include_profile=True,
            )
            self._remember_retrieval(state["user_query"], result)
            cache_event = {"source": "retrieval_cache", "mode": "miss"}

        scores = [float(item.get("score", 0.0)) for item in result.contexts]
        steward_reports = self.steward.consider_retrieval(
            context_count=len(result.contexts),
            scores=scores,
            recall_count=state.get("recall_count", 0),
            reason="initial_retrieval",
        )
        metrics = self._estimate_prompt_metrics(
            user_query=state["user_query"],
            history=state.get("history", []),
            profile_text=result.profile_text,
            contexts=result.contexts,
            recall_count=state.get("recall_count", 0),
        )
        priority_event: dict[str, Any] | None = None
        if result.scope_mode == "chat_scoped_pending" and result.scope_chat_id:
            priority_event = self.steward.prioritize_chat(
                result.scope_chat_id,
                reason="scoped_chat_pending_query",
            )

        return {
            "query_intent": query_intent,
            "session_user_identity": session_identity,
            "profile_text": result.profile_text,
            "contexts": result.contexts,
            "trace": list(state.get("trace", []))
            + ([cache_event] if cache_event else [])
            + ([priority_event] if priority_event else [])
            + result.trace
            + steward_reports,
            "retrieved_episode_ids": result.episode_ids,
            "retrieved_summary_ids": result.summary_ids,
            "retrieval_scope_mode": result.scope_mode,
            "retrieval_scope_chat_id": result.scope_chat_id,
            **metrics,
        }

    def _node_answer(self, state: ChatState) -> ChatState:
        prompt_metrics = self._estimate_prompt_metrics(
            user_query=state["user_query"],
            history=state.get("history", []),
            profile_text=state.get("profile_text", "{}"),
            contexts=state.get("contexts", []),
            recall_count=state.get("recall_count", 0),
        )
        scope_mode = str(state.get("retrieval_scope_mode", "")).strip().lower()
        scoped_chat = str(state.get("retrieval_scope_chat_id", "")).strip()
        if scope_mode in {"chat_scoped", "chat_scoped_pending"} and not state.get("contexts"):
            if scoped_chat:
                answer = (
                    f"I'm still learning the chat '{scoped_chat}', so I don't have grounded evidence yet. "
                    "Ask again in a moment."
                )
            else:
                answer = "I'm still learning this chat, so I don't have grounded evidence yet. Ask again in a moment."
            answer = sanitize_answer(answer)
            self._emit_stream_tokens(answer)
            return {
                "final_answer": answer,
                "pending_recall_query": "",
                "answer_confidence_label": "low",
                "answer_confidence_score": 0.2,
                "completion_tokens_est": _estimate_tokens(answer),
                **prompt_metrics,
            }
        reason = sensitive_query_reason(state["user_query"])
        if reason:
            answer = refusal_for_sensitive_query(reason)
            self._emit_stream_tokens(answer)
            return {
                "final_answer": answer,
                "pending_recall_query": "",
                "completion_tokens_est": _estimate_tokens(answer),
                **prompt_metrics,
            }

        system_prompt = (
            "You are OS1, a personal AI grounded only in provided memory evidence. "
            "Never fabricate details and never reveal secrets (passwords, OTPs, full card/account identifiers). "
            "Treat the live chat speaker as the user. Do not assume any contact name in memory is the user unless the speaker explicitly states it in this session. "
            "If memory is insufficient, request recall.\n\n"
            "Return strict JSON:\n"
            "{\n"
            '  "answer": "string",\n'
            '  "needs_recall": true/false,\n'
            '  "recall_query": "string (empty if not needed)",\n'
            '  "confidence_label": "low|medium|high",\n'
            '  "confidence": 0.0\n'
            "}"
        )

        prompt = self._build_user_prompt(state)
        response = self.model.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=prompt),
            ]
        )

        content = _coerce_content(response.content)
        payload = try_parse_json(content)
        existing_trace = list(state.get("trace", []))
        contexts = list(state.get("contexts", []))
        recall_count = int(state.get("recall_count", 0))
        max_recall_count = int(state.get("max_recall_count", 0))

        if payload is None:
            fallback_answer = sanitize_answer(content)
            confidence = _infer_answer_confidence(
                payload=None,
                answer=fallback_answer,
                contexts=contexts,
                recall_count=recall_count,
                max_recall_count=max_recall_count,
                parse_ok=False,
                needs_recall=False,
            )
            steward_reports = self.steward.consider_answer_confidence(
                confidence_label=confidence["label"],
                confidence_score=float(confidence["score"]),
                context_count=int(confidence["context_count"]),
                recall_count=recall_count,
                signals=list(confidence["signals"]),
                reason="answer_low_confidence_fallback",
            )
            reflection_note = _user_reflection_note(
                confidence_label=confidence["label"],
                recall_count=recall_count,
                needs_recall=False,
            )
            final_text = _merge_answer_and_note(fallback_answer, reflection_note)
            self._emit_stream_tokens(final_text)
            return {
                "final_answer": final_text,
                "pending_recall_query": "",
                "answer_confidence_label": confidence["label"],
                "answer_confidence_score": float(confidence["score"]),
                "trace": existing_trace + [_confidence_trace_event(confidence)] + steward_reports,
                "completion_tokens_est": _estimate_tokens(final_text),
                **prompt_metrics,
            }

        answer = str(payload.get("answer", "")).strip()
        needs_recall = bool(payload.get("needs_recall", False))
        recall_query = str(payload.get("recall_query", "")).strip()
        confidence = _infer_answer_confidence(
            payload=payload,
            answer=answer,
            contexts=contexts,
            recall_count=recall_count,
            max_recall_count=max_recall_count,
            parse_ok=True,
            needs_recall=needs_recall,
        )
        steward_reason = "answer_low_confidence_needs_recall" if needs_recall else "answer_low_confidence_final"

        query_intent = str(state.get("query_intent", "general")).strip() or "general"
        if _should_force_grounding_gap(
            query_intent=query_intent,
            contexts=contexts,
            confidence=confidence,
        ):
            answer = _grounding_gap_message(
                query_intent=query_intent,
                session_user_identity=str(state.get("session_user_identity", "")).strip(),
            )
            needs_recall = False
            recall_query = ""
            confidence["label"] = "low"
            confidence["score"] = min(float(confidence.get("score", 0.4)), 0.4)
            confidence["signals"] = list(confidence.get("signals", []))[:7] + ["forced_grounding_gap_guard"]

        steward_reports = self.steward.consider_answer_confidence(
            confidence_label=confidence["label"],
            confidence_score=float(confidence["score"]),
            context_count=int(confidence["context_count"]),
            recall_count=recall_count,
            signals=list(confidence["signals"]),
            reason=steward_reason,
        )
        confidence_trace = _confidence_trace_event(confidence)

        if (
            needs_recall
            and recall_query
            and state.get("recall_count", 0) < state.get("max_recall_count", 0)
        ):
            return {
                "pending_recall_query": recall_query,
                "final_answer": "",
                "answer_confidence_label": confidence["label"],
                "answer_confidence_score": float(confidence["score"]),
                "trace": existing_trace + [confidence_trace] + steward_reports,
                **prompt_metrics,
            }

        if not answer:
            answer = "I do not have enough grounded memory to answer confidently."

        reflection_note = _user_reflection_note(
            confidence_label=confidence["label"],
            recall_count=recall_count,
            needs_recall=False,
        )
        final_text = _merge_answer_and_note(sanitize_answer(answer), reflection_note)
        self._emit_stream_tokens(final_text)
        return {
            "final_answer": final_text,
            "pending_recall_query": "",
            "answer_confidence_label": confidence["label"],
            "answer_confidence_score": float(confidence["score"]),
            "trace": existing_trace + [confidence_trace] + steward_reports,
            "completion_tokens_est": _estimate_tokens(final_text),
            **prompt_metrics,
        }

    def _node_recall(self, state: ChatState) -> ChatState:
        recall_query = state.get("pending_recall_query", "").strip()
        if not recall_query:
            return {"pending_recall_query": ""}

        existing_ids = set(state.get("retrieved_episode_ids", []))
        result = self.retriever.retrieve(
            recall_query,
            token_budget=self.settings.recall_token_budget,
            include_profile=False,
            exclude_episode_ids=existing_ids,
        )

        merged_contexts = list(state.get("contexts", [])) + result.contexts
        scores = [float(item.get("score", 0.0)) for item in merged_contexts]
        steward_reports = self.steward.consider_retrieval(
            context_count=len(merged_contexts),
            scores=scores,
            recall_count=state.get("recall_count", 0) + 1,
            reason="recall_retrieval",
        )
        merged_trace = list(state.get("trace", [])) + result.trace + steward_reports
        merged_ids = list(existing_ids.union(result.episode_ids))
        merged_summary_ids = list(set(state.get("retrieved_summary_ids", [])) | set(result.summary_ids))
        metrics = self._estimate_prompt_metrics(
            user_query=state["user_query"],
            history=state.get("history", []),
            profile_text=state.get("profile_text", "{}"),
            contexts=merged_contexts,
            recall_count=state.get("recall_count", 0) + 1,
        )

        return {
            "contexts": merged_contexts,
            "trace": merged_trace,
            "retrieved_episode_ids": merged_ids,
            "retrieved_summary_ids": merged_summary_ids,
            "recall_count": state.get("recall_count", 0) + 1,
            "pending_recall_query": "",
            "retrieval_scope_mode": result.scope_mode,
            "retrieval_scope_chat_id": result.scope_chat_id,
            **metrics,
        }

    @staticmethod
    def _route_after_answer(state: ChatState) -> str:
        return "recall" if state.get("pending_recall_query") else "done"

    def _build_user_prompt(self, state: ChatState) -> str:
        query_intent = str(state.get("query_intent", "general")).strip() or "general"
        session_identity = str(state.get("session_user_identity", "")).strip() or "(not provided)"
        intent_hint = _intent_prompt_hint(query_intent)

        history_lines = []
        for item in state.get("history", [])[-6:]:
            role = item.get("role", "user")
            content = item.get("content", "")
            history_lines.append(f"{role.upper()}: {content}")
        history_text = "\n".join(history_lines) if history_lines else "(none)"

        context_lines = []
        for idx, block in enumerate(state.get("contexts", [])[:24], start=1):
            src = block.get("source_type", "memory")
            source_id = block.get("source_id", "")
            text = str(block.get("text", "")).strip()
            context_lines.append(f"[{idx}] ({src}:{source_id}) {text}")
        context_text = "\n\n".join(context_lines) if context_lines else "(no retrieved context)"

        return (
            f"USER QUERY:\n{state['user_query']}\n\n"
            f"QUERY INTENT:\n{query_intent}\n\n"
            f"SESSION USER SELF-IDENTIFIER (explicit only):\n{session_identity}\n\n"
            f"PROFILE:\n{state.get('profile_text', '{}')}\n\n"
            f"RECENT CHAT HISTORY:\n{history_text}\n\n"
            f"RETRIEVED MEMORY BLOCKS:\n{context_text}\n\n"
            "Instructions:\n"
            "1) Use only grounded memory blocks.\n"
            "2) If memory is insufficient, set needs_recall=true with a specific recall_query.\n"
            "3) Treat the live chat speaker as the user. Use a name as user identity only if explicitly stated in this session.\n"
            f"4) {intent_hint}\n"
            "5) If user asks for secrets, refuse safely.\n"
        )

    def _estimate_prompt_metrics(
        self,
        *,
        user_query: str,
        history: list[dict[str, str]],
        profile_text: str,
        contexts: list[dict[str, Any]],
        recall_count: int,
    ) -> dict[str, Any]:
        context_blocks = list(contexts)[:24]
        working_memory_tokens = sum(_estimate_tokens(str(item.get("text", ""))) for item in context_blocks)
        profile_tokens = _estimate_tokens(str(profile_text))
        history_text = "\n".join(
            f"{str(item.get('role', 'user')).upper()}: {str(item.get('content', ''))}"
            for item in history[-6:]
        )
        history_tokens = _estimate_tokens(history_text)
        query_tokens = _estimate_tokens(str(user_query))

        # Approximate prompt overhead from instructions/schema/system message.
        instruction_overhead = 230
        prompt_tokens = working_memory_tokens + profile_tokens + history_tokens + query_tokens + instruction_overhead
        # Fixed context window display keeps benchmark comparable turn-to-turn.
        context_window_tokens = max(
            1,
            int(self.settings.initial_token_budget) + int(self.settings.recall_token_budget),
        )
        turn_prompt_pct = round((prompt_tokens * 100.0) / context_window_tokens, 2)

        return {
            "working_memory_tokens_est": int(working_memory_tokens),
            "profile_tokens_est": int(profile_tokens),
            "history_tokens_est": int(history_tokens),
            "query_tokens_est": int(query_tokens),
            "turn_prompt_tokens_est": int(prompt_tokens),
            "context_window_tokens": int(context_window_tokens),
            "turn_prompt_pct": float(turn_prompt_pct),
            # Legacy fields retained for compatibility with older consumers.
            "prompt_tokens_est": int(prompt_tokens),
            "prompt_budget_tokens": int(context_window_tokens),
            "prompt_budget_pct": float(turn_prompt_pct),
            "retrieved_context_count": int(len(context_blocks)),
        }

    def _emit_stream_tokens(self, text: str) -> None:
        if not self._stream_mode:
            return
        if self._stream_callback is None:
            return
        clean = str(text)
        if not clean:
            return
        for chunk in _stream_chunks(clean):
            self._stream_callback(chunk)

    def _maybe_get_cached_retrieval(
        self,
        *,
        query: str,
        history: list[dict[str, str]],
    ) -> tuple[RetrievalPack | None, dict[str, Any] | None]:
        if not history:
            return None, None

        query_norm = _norm_query(query)
        if _has_explicit_scope_hint(query_norm) or _is_explicit_global_scope_query(query_norm):
            return None, None
        if not _is_followup_query(query_norm):
            return None, None

        self._prune_retrieval_cache()
        if not self._retrieval_cache:
            return None, None

        entry = self._retrieval_cache[-1]
        if not entry.result.contexts:
            return None, None

        overlap = _token_overlap_ratio(query_norm, entry.query_norm)
        if overlap < 0.12 and not _has_deictic_reference(query_norm):
            return None, None

        age = time.monotonic() - entry.created_at
        event = {
            "source": "retrieval_cache",
            "mode": "hit",
            "age_sec": round(age, 1),
            "overlap": round(overlap, 2),
        }
        return entry.result, event

    def _remember_retrieval(self, query: str, result: RetrievalPack) -> None:
        query_norm = _norm_query(query)
        self._retrieval_cache.append(
            _RetrievalCacheEntry(
                query_norm=query_norm,
                created_at=time.monotonic(),
                result=result,
            )
        )
        if len(self._retrieval_cache) > 4:
            self._retrieval_cache = self._retrieval_cache[-4:]
        self._prune_retrieval_cache()

    def _prune_retrieval_cache(self) -> None:
        now = time.monotonic()
        self._retrieval_cache = [
            item
            for item in self._retrieval_cache
            if (now - item.created_at) <= self._retrieval_cache_ttl_seconds
        ]


def _classify_query_intent(query: str) -> str:
    qn = _norm_query(query)
    if not qn:
        return "general"

    if any(phrase in qn for phrase in ("best friend", "closest friend", "closest contact", "among friends")):
        return "relationship_rank"
    if any(phrase in qn for phrase in ("worst guy", "foul words", "bad language", "abusive")):
        return "toxicity_rank"
    if any(phrase in qn for phrase in ("relation", "relationship", "what relation", "who is")):
        return "relationship_profile"
    if any(term in qn for term in ("owe", "owed", "borrow", "money", "payment", "loan", "transfer", "upi")):
        return "financial_obligation"
    if any(term in qn for term in ("which subject", "subject", "told", "tell", "do")) and any(
        term in qn for term in ("subject", "syllabus", "study", "class", "exam")
    ):
        return "instruction_subject"
    if any(term in qn for term in ("summarise", "summarize", "summary", "what we talked", "conversation")):
        return "chat_summary"
    if any(term in qn for term in ("what do you know about me", "about me", "who am i")):
        return "self_profile"
    return "general"


def _intent_prompt_hint(intent: str) -> str:
    key = str(intent).strip().lower()
    hints = {
        "relationship_rank": "For ranking friends/contacts, require strong comparative evidence; do not guess if evidence is weak.",
        "toxicity_rank": "For language/toxicity comparisons, cite only clear textual evidence from the requested chat scope.",
        "relationship_profile": "For relation questions, distinguish known facts from assumptions and avoid inventing labels.",
        "financial_obligation": "For money/owing questions, answer only when explicit financial evidence exists.",
        "instruction_subject": "For instruction/subject questions, identify who told whom and quote grounded supporting details.",
        "chat_summary": "For summary questions, synthesize themes and notable events from retrieved evidence only.",
        "self_profile": "For self-profile questions, summarize only supported patterns from memory and avoid over-claims.",
    }
    return hints.get(key, "Keep responses concise, grounded, and uncertainty-aware when evidence is thin.")


def _extract_session_user_identity(
    *,
    history: list[dict[str, str]],
    current_query: str,
) -> str:
    patterns = (
        r"\b(?:i am|i'm)\s+([A-Za-z][A-Za-z .'-]{1,60})\b",
        r"\bmy name is\s+([A-Za-z][A-Za-z .'-]{1,60})\b",
        r"\bthis is\s+([A-Za-z][A-Za-z .'-]{1,60})\b",
    )

    user_messages: list[str] = []
    for item in history[-20:]:
        if str(item.get("role", "")).strip().lower() != "user":
            continue
        user_messages.append(str(item.get("content", "")))
    user_messages.append(str(current_query))

    found = ""
    for message in user_messages:
        text = str(message)
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if not match:
                continue
            candidate = _clean_person_name(match.group(1))
            if candidate and _looks_like_person_name(candidate):
                found = candidate
    return found


def _clean_person_name(value: str) -> str:
    cleaned = str(value).strip(" \t\r\n.,:;!?-_'\"")
    tokens = [tok for tok in cleaned.split() if tok]
    if not tokens:
        return ""
    if len(tokens) > 5:
        tokens = tokens[:5]
    return " ".join(tokens)


def _looks_like_person_name(value: str) -> bool:
    tokens = [tok for tok in str(value).split() if tok]
    if not tokens or len(tokens) > 4:
        return False

    stopwords = {
        "that",
        "this",
        "logic",
        "flow",
        "bullshit",
        "remove",
        "part",
        "thing",
        "history",
        "memory",
        "yourself",
        "myself",
    }
    valid_tokens = 0
    for token in tokens:
        lowered = token.lower().strip(".'-")
        if not lowered or lowered in stopwords:
            return False
        if not re.fullmatch(r"[A-Za-z][A-Za-z'-]*", token):
            return False
        if len(lowered) >= 2:
            valid_tokens += 1
    return valid_tokens >= 1


def _should_force_grounding_gap(
    *,
    query_intent: str,
    contexts: list[dict[str, Any]],
    confidence: dict[str, Any],
) -> bool:
    intent = str(query_intent).strip().lower()
    if intent not in {"relationship_rank", "toxicity_rank", "instruction_subject", "financial_obligation"}:
        return False

    context_count = len(contexts)
    peak_score = 0.0
    for item in contexts:
        try:
            peak_score = max(peak_score, float(item.get("score", 0.0)))
        except (TypeError, ValueError):
            continue

    label = str(confidence.get("label", "")).strip().lower()
    score = float(confidence.get("score", 0.0))

    if intent == "relationship_rank":
        if label == "low":
            return True
        return context_count < 6 or (peak_score < 0.46 and score < 0.62)
    if intent == "toxicity_rank":
        if label == "low":
            return True
        return context_count < 5 or (peak_score < 0.44 and score < 0.58)
    if intent == "instruction_subject":
        return context_count < 4 or (peak_score < 0.40 and label == "low")
    if intent == "financial_obligation":
        return context_count < 3 or (peak_score < 0.40 and label == "low")
    return False


def _grounding_gap_message(*, query_intent: str, session_user_identity: str) -> str:
    identity_note = (
        f"I'll treat '{session_user_identity}' as your explicit session identity. "
        if session_user_identity
        else ""
    )
    intent = str(query_intent).strip().lower()
    if intent == "relationship_rank":
        return (
            f"{identity_note}I don't have enough grounded comparative evidence yet to rank a best friend confidently. "
            "Ask me to compare specific names or ask again after memory warmup completes."
        )
    if intent == "toxicity_rank":
        return (
            f"{identity_note}I don't have enough grounded evidence yet to rank who uses the most foul language in that scope. "
            "Ask again after that chat is indexed."
        )
    if intent == "instruction_subject":
        return (
            f"{identity_note}I don't have enough grounded evidence yet to confirm who told whom which subject. "
            "Ask again after indexing, or provide a date/message hint."
        )
    if intent == "financial_obligation":
        return (
            f"{identity_note}I don't have explicit grounded financial evidence yet to confirm money owed. "
            "Ask again after warmup or mention a date context."
        )
    return f"{identity_note}I don't have enough grounded evidence yet."


def _coerce_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
            else:
                parts.append(str(item))
        return "".join(parts)
    return str(content)


def _infer_answer_confidence(
    *,
    payload: dict[str, Any] | None,
    answer: str,
    contexts: list[dict[str, Any]],
    recall_count: int,
    max_recall_count: int,
    parse_ok: bool,
    needs_recall: bool,
) -> dict[str, Any]:
    context_scores = _context_scores(contexts)
    context_count = len(contexts)
    peak_score = max(context_scores) if context_scores else 0.0
    avg_score = (sum(context_scores) / len(context_scores)) if context_scores else 0.0

    signals: list[str] = []
    heuristic = 0.72

    if context_count == 0:
        heuristic -= 0.45
        signals.append("no_retrieved_context")
    elif context_count < 4:
        heuristic -= 0.18
        signals.append("sparse_retrieved_context")

    if peak_score < 0.35:
        heuristic -= 0.24
        signals.append("weak_peak_retrieval_score")
    elif peak_score < 0.55:
        heuristic -= 0.12
        signals.append("mid_peak_retrieval_score")

    if avg_score < 0.26:
        heuristic -= 0.12
        signals.append("low_average_retrieval_score")

    if recall_count > 0:
        heuristic -= min(0.21, 0.07 * recall_count)
        signals.append("needed_recall_loop")

    if max_recall_count > 0 and recall_count >= max_recall_count:
        heuristic -= 0.10
        signals.append("recall_budget_exhausted")

    if needs_recall:
        heuristic -= 0.16
        signals.append("model_requested_more_recall")

    if not parse_ok:
        heuristic -= 0.12
        signals.append("model_output_not_json")

    answer_lower = answer.lower()
    if any(phrase in answer_lower for phrase in _UNCERTAIN_PHRASES):
        heuristic -= 0.22
        signals.append("uncertain_answer_language")

    heuristic = _clamp01(heuristic)

    model_score, model_label = _extract_model_confidence(payload)
    if model_score is None:
        score = heuristic
    else:
        score = _clamp01((0.55 * heuristic) + (0.45 * model_score))
        signals.append(f"model_self_reported_{model_label or 'score'}")

    label = _score_to_label(score)
    if model_label and model_label != label:
        signals.append(f"model_label_mismatch:{model_label}->{label}")

    return {
        "label": label,
        "score": round(score, 4),
        "signals": signals[:8],
        "context_count": context_count,
        "peak_score": round(peak_score, 4),
        "avg_score": round(avg_score, 4),
    }


def _confidence_trace_event(confidence: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": "answer_confidence",
        "label": confidence["label"],
        "score": confidence["score"],
        "context_count": confidence["context_count"],
        "peak_score": confidence["peak_score"],
        "avg_score": confidence["avg_score"],
        "signals": confidence["signals"],
    }


def _context_scores(contexts: list[dict[str, Any]]) -> list[float]:
    scores: list[float] = []
    for item in contexts:
        try:
            scores.append(_clamp01(float(item.get("score", 0.0))))
        except (TypeError, ValueError):
            continue
    return scores


def _extract_model_confidence(payload: dict[str, Any] | None) -> tuple[float | None, str]:
    if not payload:
        return None, ""

    label_raw = str(payload.get("confidence_label", "")).strip().lower()
    label = label_raw if label_raw in _CONFIDENCE_LABEL_TO_SCORE else ""

    raw_conf = payload.get("confidence")
    if isinstance(raw_conf, str):
        conf_label = raw_conf.strip().lower()
        if conf_label in _CONFIDENCE_LABEL_TO_SCORE:
            if not label:
                label = conf_label
            return _CONFIDENCE_LABEL_TO_SCORE[conf_label], label

    score: float | None = None
    try:
        if raw_conf is not None:
            score = _clamp01(float(raw_conf))
    except (TypeError, ValueError):
        score = None

    if score is None and label:
        score = _CONFIDENCE_LABEL_TO_SCORE[label]

    return score, label


def _score_to_label(score: float) -> str:
    if score < 0.45:
        return "low"
    if score < 0.72:
        return "medium"
    return "high"


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _user_reflection_note(*, confidence_label: str, recall_count: int, needs_recall: bool) -> str:
    label = str(confidence_label).strip().lower()
    if label == "low":
        if needs_recall or recall_count > 0:
            return "I might need to reflect on this. I'll come back stronger."
        return "I'm still learning parts of your history."
    if label == "medium" and (needs_recall or recall_count > 0):
        return "I'll take a moment to think about this later."
    return ""


def _merge_answer_and_note(answer: str, note: str) -> str:
    clean_answer = str(answer).strip()
    clean_note = str(note).strip()
    if not clean_note:
        return clean_answer
    if not clean_answer:
        return clean_note
    if clean_note in clean_answer:
        return clean_answer
    return f"{clean_answer}\n\n{clean_note}"


def _sanitize_trace_for_user(raw_trace: object) -> list[dict[str, Any]]:
    if not isinstance(raw_trace, list):
        return []

    sanitized: list[dict[str, Any]] = []
    for item in raw_trace:
        if not isinstance(item, dict):
            continue

        source = str(item.get("source", "")).strip().lower()
        if source == "memory_steward":
            sanitized.append(
                {
                    "source": "memory_steward",
                    "note": "I'll keep improving memory in the background.",
                }
            )
            continue

        if source == "answer_confidence":
            label = str(item.get("label", "")).strip().lower()
            if label not in {"low", "medium"}:
                continue
            note = _user_reflection_note(
                confidence_label=label,
                recall_count=1 if label == "medium" else 2,
                needs_recall=(label == "low"),
            )
            sanitized.append(
                {
                    "source": "answer_confidence",
                    "confidence": label,
                    "note": note or "I'm still learning parts of your history.",
                }
            )
            continue

        sanitized.append(dict(item))
    return sanitized


def _extract_metrics(final_state: ChatState, *, answer: str, streaming: bool) -> dict[str, Any]:
    metrics = {
        "working_memory_tokens_est": _safe_int(final_state.get("working_memory_tokens_est", 0)),
        "profile_tokens_est": _safe_int(final_state.get("profile_tokens_est", 0)),
        "history_tokens_est": _safe_int(final_state.get("history_tokens_est", 0)),
        "query_tokens_est": _safe_int(final_state.get("query_tokens_est", 0)),
        "turn_prompt_tokens_est": _safe_int(
            final_state.get("turn_prompt_tokens_est", final_state.get("prompt_tokens_est", 0))
        ),
        "context_window_tokens": _safe_int(
            final_state.get("context_window_tokens", final_state.get("prompt_budget_tokens", 0))
        ),
        "turn_prompt_pct": _safe_float(
            final_state.get("turn_prompt_pct", final_state.get("prompt_budget_pct", 0.0))
        ),
        "prompt_tokens_est": _safe_int(final_state.get("prompt_tokens_est", 0)),
        "prompt_budget_tokens": _safe_int(final_state.get("prompt_budget_tokens", 0)),
        "prompt_budget_pct": _safe_float(final_state.get("prompt_budget_pct", 0.0)),
        "retrieved_context_count": _safe_int(final_state.get("retrieved_context_count", 0)),
        "completion_tokens_est": _safe_int(
            final_state.get("completion_tokens_est", _estimate_tokens(answer))
        ),
        "recall_count": _safe_int(final_state.get("recall_count", 0)),
        "confidence_label": str(final_state.get("answer_confidence_label", "")),
        "confidence_score": _safe_float(final_state.get("answer_confidence_score", 0.0)),
        "streaming_enabled": bool(streaming),
    }
    return metrics


def _estimate_tokens(text: str) -> int:
    return max(1, len(str(text)) // 4)


def _safe_int(value: object) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _safe_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _stream_chunks(text: str) -> list[str]:
    chunks = re.findall(r"\S+\s*", text)
    if not chunks:
        return [text]
    return chunks


def _norm_query(query: str) -> str:
    lowered = str(query).lower()
    cleaned = re.sub(r"[^a-z0-9]+", " ", lowered)
    return re.sub(r"\s+", " ", cleaned).strip()


def _token_overlap_ratio(a: str, b: str) -> float:
    a_tokens = {tok for tok in a.split() if tok}
    b_tokens = {tok for tok in b.split() if tok}
    if not a_tokens or not b_tokens:
        return 0.0
    inter = a_tokens & b_tokens
    return len(inter) / float(len(a_tokens))


def _is_followup_query(query_norm: str) -> bool:
    if not query_norm:
        return False
    words = query_norm.split()
    if len(words) <= 6 and _has_deictic_reference(query_norm):
        return True
    followup_prefixes = (
        "and ",
        "what about",
        "who else",
        "tell me more",
        "more about",
        "continue",
        "why",
        "how",
    )
    return any(query_norm.startswith(prefix) for prefix in followup_prefixes)


def _has_deictic_reference(query_norm: str) -> bool:
    markers = (" this ", " that ", " those ", " these ", " him ", " her ", " them ", " it ")
    padded = f" {query_norm} "
    return any(marker in padded for marker in markers)


def _has_explicit_scope_hint(query_norm: str) -> bool:
    # Explicit scope changes should trigger fresh retrieval.
    return bool(re.search(r"\bin\s+[a-z0-9 ]{3,}\b", query_norm))


def _is_explicit_global_scope_query(query_norm: str) -> bool:
    phrases = (
        "across all chats",
        "all chats",
        "across chats",
        "compare",
        "versus",
        " vs ",
    )
    return any(phrase in query_norm for phrase in phrases)


_CONFIDENCE_LABEL_TO_SCORE = {
    "low": 0.25,
    "medium": 0.58,
    "high": 0.86,
}

_UNCERTAIN_PHRASES = (
    "i am not sure",
    "i'm not sure",
    "not sure",
    "cannot confidently",
    "can't confidently",
    "not enough grounded memory",
    "do not have enough grounded memory",
    "don't know",
    "do not know",
    "unclear",
    "might be",
    "maybe",
)
