from __future__ import annotations

import re


def _detect_scripts(text: str) -> list[str]:
    has_devanagari = any("\u0900" <= ch <= "\u097F" for ch in text)
    has_bengali = any("\u0980" <= ch <= "\u09FF" for ch in text)
    has_latin = any(("A" <= ch <= "Z") or ("a" <= ch <= "z") for ch in text)

    langs: list[str] = []
    if has_latin:
        langs.append("Latin")
    if has_devanagari:
        langs.append("Devanagari")
    if has_bengali:
        langs.append("Bengali")
    if not langs:
        langs.append("Unknown")
    return langs


def normalize_for_search(raw_text: str) -> dict[str, object]:
    # Search stream is optimized for retrieval; raw stream preserves exact user text.
    normalized = raw_text.replace("\u202f", " ").replace("\u200e", " ")
    normalized = re.sub(r"https?://\S+", " [URL] ", normalized)
    normalized = re.sub(r"<Media omitted>", " ", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\s+", " ", normalized).strip()

    return {
        "normalized": normalized,
        "raw": raw_text,
        "languages": _detect_scripts(raw_text),
    }
