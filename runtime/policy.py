from __future__ import annotations

import re

from memory.pii import redact_sensitive_text

_SENSITIVE_QUERIES = {
    "password": re.compile(r"\b(password|passcode|pin)\b", re.IGNORECASE),
    "otp": re.compile(r"\botp\b", re.IGNORECASE),
    "card_number": re.compile(r"\b(card number|credit card|cvv|debit card)\b", re.IGNORECASE),
    "bank_account": re.compile(r"\b(account number|ifsc|bank details|upi id)\b", re.IGNORECASE),
    "identity_numbers": re.compile(r"\b(aadhaar|aadhar|pan number)\b", re.IGNORECASE),
}


def sensitive_query_reason(query: str) -> str | None:
    for reason, pattern in _SENSITIVE_QUERIES.items():
        if pattern.search(query):
            return reason
    return None


def refusal_for_sensitive_query(reason: str) -> str:
    return (
        "I cannot provide sensitive credentials or financial identity details from memory "
        f"({reason}). I can help with a safe summary instead."
    )


def sanitize_answer(text: str) -> str:
    return redact_sensitive_text(text)
