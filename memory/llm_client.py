from __future__ import annotations

import json
import re
from typing import Any

from openai import OpenAI

from config.settings import Settings

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


class OpenRouterLLM:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.model_name = settings.model_name
        self.enabled = bool(settings.openrouter_api_key)
        self.client: OpenAI | None = None
        if self.enabled:
            self.client = OpenAI(
                api_key=settings.openrouter_api_key,
                base_url=settings.openrouter_base_url,
            )

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float | None = None,
        max_tokens: int = 900,
    ) -> str:
        if not self.client:
            raise RuntimeError("OPENROUTER_API_KEY is missing. Cannot call model.")

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature if temperature is not None else self.settings.model_temperature,
            max_tokens=max_tokens,
            extra_headers={
                "HTTP-Referer": "https://localhost/truffle-memory",
                "X-Title": "TruffleMemory",
            },
        )
        message = response.choices[0].message
        return message.content or ""

    def complete_json(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float | None = None,
        max_tokens: int = 900,
    ) -> dict[str, Any] | None:
        content = self.complete(
            system_prompt,
            user_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return try_parse_json(content)


def try_parse_json(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    if not stripped:
        return None

    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    match = _JSON_BLOCK_RE.search(stripped)
    if not match:
        return None

    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None
