from __future__ import annotations

from typing import Any, Callable

from runtime.graph import LangGraphChatEngine


class ChatSession:
    def __init__(self, engine: LangGraphChatEngine | None = None) -> None:
        self.engine = engine or LangGraphChatEngine()
        self.history: list[dict[str, str]] = []
        self._turn_count = 0
        self._cumulative_prompt_tokens_est = 0

    def ask(
        self,
        query: str,
        *,
        stream: bool = False,
        on_token: Callable[[str], None] | None = None,
    ) -> tuple[str, list[dict[str, object]], dict[str, Any]]:
        answer, trace, metrics = self.engine.chat(
            query,
            history=self.history,
            stream=stream,
            on_token=on_token,
        )
        turn_prompt_tokens = _safe_int(
            metrics.get("turn_prompt_tokens_est", metrics.get("prompt_tokens_est", 0))
        )
        self._cumulative_prompt_tokens_est += max(0, turn_prompt_tokens)
        self._turn_count += 1

        self.history.append({"role": "user", "content": query})
        self.history.append({"role": "assistant", "content": answer})
        # Keep session history bounded for stable prompts.
        self.history = self.history[-20:]

        context_window_tokens = max(
            1,
            _safe_int(metrics.get("context_window_tokens", metrics.get("prompt_budget_tokens", 1))),
        )
        retained_history_tokens = _estimate_history_tokens(self.history)
        retained_pct = round((retained_history_tokens * 100.0) / context_window_tokens, 2)
        metrics["session_turn_count"] = int(self._turn_count)
        metrics["session_history_messages"] = int(len(self.history))
        metrics["session_retained_tokens_est"] = int(retained_history_tokens)
        metrics["session_retained_pct"] = float(retained_pct)
        metrics["session_cumulative_prompt_tokens_est"] = int(self._cumulative_prompt_tokens_est)
        return answer, trace, metrics


def _estimate_history_tokens(history: list[dict[str, str]]) -> int:
    if not history:
        return 0
    text = "\n".join(
        f"{str(item.get('role', 'user')).upper()}: {str(item.get('content', ''))}"
        for item in history
    )
    return _estimate_tokens(text)


def _estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, len(str(text)) // 4)


def _safe_int(value: object) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0
