from __future__ import annotations

from pathlib import Path

from memory.parser import parse_chat_file


def test_parser_handles_multiline(tmp_path: Path) -> None:
    content = (
        "2/25/23, 9:39 PM - Rohit Kumar: hello\n"
        "continuation line\n"
        "2/25/23, 9:40 PM - Tanishq: hi\n"
    )
    path = tmp_path / "chat.txt"
    path.write_text(content, encoding="utf-8")

    messages = parse_chat_file(path, chat_id="test_chat")
    assert len(messages) == 2
    assert "continuation line" in messages[0].content_raw
    assert messages[1].sender == "Tanishq"
