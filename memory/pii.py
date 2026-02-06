from __future__ import annotations

import re

PII_PATTERNS: dict[str, re.Pattern[str]] = {
    "ifsc": re.compile(r"\b[A-Z]{4}0[A-Z0-9]{6}\b", re.IGNORECASE),
    "upi_id": re.compile(r"\b[a-zA-Z0-9._-]{2,}@[a-zA-Z]{2,}\b", re.IGNORECASE),
    "aadhaar": re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b"),
    "pan": re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b", re.IGNORECASE),
    "password": re.compile(r"\bpassword\b\s*[:=-]?\s*\S+", re.IGNORECASE),
    "otp": re.compile(r"\botp\b[^\n]{0,20}\b\d{4,8}\b|\b\d{4,8}\b(?=[^\n]{0,20}\botp\b)", re.IGNORECASE),
    "account_number": re.compile(
        r"\b(?:a/c|ac|account(?:\s+number)?|ac(?:count)?\s+no)\b[^\n]{0,16}\b\d{6,18}\b",
        re.IGNORECASE,
    ),
    "card_number": re.compile(r"\b(?:\d[ -]?){13,19}\b"),
    "phone_number": re.compile(r"\b(?:\+?\d{1,3}[- ]?)?\d{10}\b"),
}


def detect_pii_tags(text: str) -> list[str]:
    tags: list[str] = []
    for tag, pattern in PII_PATTERNS.items():
        if pattern.search(text):
            tags.append(tag)
    return sorted(tags)


def redact_sensitive_text(text: str) -> str:
    redacted = text
    for tag, pattern in PII_PATTERNS.items():
        redacted = pattern.sub(f"[REDACTED_{tag.upper()}]", redacted)
    return redacted
