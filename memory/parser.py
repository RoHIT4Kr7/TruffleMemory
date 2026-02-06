from __future__ import annotations

from datetime import datetime
import hashlib
from pathlib import Path
import re

from memory.models import ChatMessage

# Example: 2/25/23, 9:39 PM - Rohit Kumar: hello
LINE_RE = re.compile(
    r"^(?P<date>\d{1,2}/\d{1,2}/\d{2,4}),\s+"
    r"(?P<time>\d{1,2}:\d{2}(?:[\u202f\s]?[APMapm]{2})?)\s+-\s+"
    r"(?P<body>.*)$"
)

TIME_FORMATS = (
    "%m/%d/%y, %I:%M %p",
    "%m/%d/%Y, %I:%M %p",
    "%d/%m/%y, %I:%M %p",
    "%d/%m/%Y, %I:%M %p",
)


def _normalize_datetime_parts(date_text: str, time_text: str) -> str:
    clean_time = time_text.replace("\u202f", " ").replace("\u200e", " ").strip()
    clean_time = re.sub(r"\s+", " ", clean_time)
    clean_time = clean_time.upper()
    return f"{date_text}, {clean_time}"


def _parse_datetime(date_text: str, time_text: str) -> datetime:
    normalized = _normalize_datetime_parts(date_text, time_text)
    for fmt in TIME_FORMATS:
        try:
            return datetime.strptime(normalized, fmt)
        except ValueError:
            continue
    raise ValueError(f"Unable to parse datetime: {normalized}")


def _derive_chat_id(path: Path) -> str:
    stem = path.stem
    prefix = "WhatsApp Chat with "
    if stem.startswith(prefix):
        return stem[len(prefix) :].strip()
    return stem.strip()


def _message_id(chat_id: str, index: int, timestamp: datetime, sender: str | None, content: str) -> str:
    payload = f"{chat_id}|{index}|{timestamp.isoformat()}|{sender or ''}|{content}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def parse_chat_file(path: str | Path, chat_id: str | None = None) -> list[ChatMessage]:
    file_path = Path(path)
    chat = chat_id or _derive_chat_id(file_path)

    messages: list[ChatMessage] = []
    current: dict[str, object] | None = None

    with file_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.rstrip("\n")
            match = LINE_RE.match(line)

            if match:
                if current is not None:
                    messages.append(_finalize_message(chat, len(messages), current))

                date_text = match.group("date")
                time_text = match.group("time")
                body = match.group("body").strip()

                try:
                    timestamp = _parse_datetime(date_text, time_text)
                except ValueError:
                    # Keep raw line as continuation if datetime parse fails.
                    if current is not None:
                        current["content"] = f"{current['content']}\n{line}".strip()
                    continue

                sender: str | None = None
                content = body
                is_system = True

                if ":" in body:
                    sender_candidate, content_candidate = body.split(":", 1)
                    sender_candidate = sender_candidate.strip()
                    content_candidate = content_candidate.strip()
                    if sender_candidate:
                        sender = sender_candidate
                        content = content_candidate
                        is_system = False

                current = {
                    "timestamp": timestamp,
                    "sender": sender,
                    "content": content,
                    "is_system": is_system,
                    "line_no": line_no,
                }
            else:
                if current is None:
                    continue
                continuation = line.strip()
                if continuation:
                    current["content"] = f"{current['content']}\n{continuation}".strip()

    if current is not None:
        messages.append(_finalize_message(chat, len(messages), current))

    return messages


def _finalize_message(chat_id: str, index: int, current: dict[str, object]) -> ChatMessage:
    timestamp = current["timestamp"]
    sender = current["sender"]
    content = str(current["content"])
    is_system = bool(current["is_system"])
    message_id = _message_id(chat_id, index, timestamp, sender, content)

    return ChatMessage(
        message_id=message_id,
        chat_id=chat_id,
        timestamp=timestamp,
        sender=sender,
        content_raw=content,
        is_system=is_system,
        has_media="<Media omitted>" in content,
    )
