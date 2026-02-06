from __future__ import annotations

from datetime import timedelta
import hashlib
import re

from memory.models import ChatMessage, Episode
from memory.rosetta import normalize_for_search

SYSTEM_NOISE_PATTERNS = (
    re.compile(r"messages and calls are end-to-end encrypted", re.IGNORECASE),
    re.compile(r"joined using (this )?group", re.IGNORECASE),
    re.compile(r"created (this )?group", re.IGNORECASE),
    re.compile(r"added .+", re.IGNORECASE),
)


def _is_ignorable_system_message(message: ChatMessage) -> bool:
    if not message.is_system:
        return False
    content = message.content_raw.strip()
    return any(pattern.search(content) for pattern in SYSTEM_NOISE_PATTERNS)


def _episode_id(chat_id: str, message_ids: list[str]) -> str:
    payload = f"{chat_id}|{message_ids[0]}|{message_ids[-1]}|{len(message_ids)}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def chunk_messages(
    messages: list[ChatMessage],
    gap_hours: float = 2.0,
    max_chars: int = 2200,
) -> list[Episode]:
    filtered = [msg for msg in messages if not _is_ignorable_system_message(msg)]
    if not filtered:
        return []

    episodes: list[Episode] = []
    current: list[ChatMessage] = []
    current_chars = 0
    gap_delta = timedelta(hours=gap_hours)

    for msg in filtered:
        msg_chars = len(msg.content_raw) + (len(msg.sender) if msg.sender else 6)

        should_split = False
        if current:
            if msg.timestamp - current[-1].timestamp > gap_delta:
                should_split = True
            elif current_chars + msg_chars > max_chars:
                should_split = True

        if should_split:
            episodes.append(_build_episode(current))
            current = []
            current_chars = 0

        current.append(msg)
        current_chars += msg_chars

    if current:
        episodes.append(_build_episode(current))

    return episodes


def _build_episode(messages: list[ChatMessage]) -> Episode:
    first = messages[0]
    last = messages[-1]
    lines = []
    for msg in messages:
        sender = msg.sender or "SYSTEM"
        lines.append(f"[{msg.timestamp.isoformat()}] {sender}: {msg.content_raw}")

    text_raw = "\n".join(lines)
    rosetta = normalize_for_search(text_raw)

    return Episode(
        episode_id=_episode_id(first.chat_id, [m.message_id for m in messages]),
        chat_id=first.chat_id,
        start_ts=first.timestamp,
        end_ts=last.timestamp,
        message_ids=[m.message_id for m in messages],
        text_raw=text_raw,
        text_search=str(rosetta["normalized"]),
        salience=0.0,
        tier="pass_through",
        sensitivity_tags=[],
    )
