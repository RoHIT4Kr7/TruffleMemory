from __future__ import annotations

from runtime.retrieval import _resolve_chat_scope


def test_retrieval_scope_single_chat_match() -> None:
    chat_ids = ["Reposting Baddies", "Pranav", "Mridul mishra"]
    query = "in reposting baddies who talks the most?"
    assert _resolve_chat_scope(query, chat_ids) == "Reposting Baddies"


def test_retrieval_scope_explicit_global_query() -> None:
    chat_ids = ["Reposting Baddies", "Pranav", "Mridul mishra"]
    query = "compare across all chats who is most active"
    assert _resolve_chat_scope(query, chat_ids) is None


def test_retrieval_scope_group_name_not_split_into_member_profile() -> None:
    chat_ids = ["Reposting Baddies", "HRITHIK GHOSH", "Pranav"]
    query = "in reposting baddies who talks too much?"
    assert _resolve_chat_scope(query, chat_ids) == "Reposting Baddies"
