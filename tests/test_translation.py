"""Tests unitaires pour la traduction chat <-> completions de gateway.py."""
import pytest
from fastapi import HTTPException

from gateway import (
    chat_to_completion_payload,
    chat_to_sse,
    completion_to_chat,
    messages_to_prompt,
)


def test_messages_to_prompt_single_user_message_passthrough():
    # Cas spécial : Evo reçoit une séquence ADN brute, pas un "User: ...".
    messages = [{"role": "user", "content": "ATTCCGATTCCG"}]
    assert messages_to_prompt(messages) == "ATTCCGATTCCG"


def test_messages_to_prompt_multi_turn_flattened():
    messages = [
        {"role": "system", "content": "Tu es utile."},
        {"role": "user", "content": "Bonjour"},
        {"role": "assistant", "content": "Salut"},
        {"role": "user", "content": "Ça va ?"},
    ]
    prompt = messages_to_prompt(messages)
    assert prompt == (
        "System: Tu es utile.\n"
        "User: Bonjour\n"
        "Assistant: Salut\n"
        "User: Ça va ?\n"
        "Assistant:"
    )


def test_messages_to_prompt_empty_list_rejected():
    with pytest.raises(HTTPException) as exc_info:
        messages_to_prompt([])
    assert exc_info.value.status_code == 400


def test_messages_to_prompt_not_a_list_rejected():
    with pytest.raises(HTTPException) as exc_info:
        messages_to_prompt("not a list")
    assert exc_info.value.status_code == 400


def test_messages_to_prompt_missing_content_rejected():
    with pytest.raises(HTTPException) as exc_info:
        messages_to_prompt([{"role": "user"}])
    assert exc_info.value.status_code == 400


def test_messages_to_prompt_multimodal_content_rejected():
    messages = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
    with pytest.raises(HTTPException) as exc_info:
        messages_to_prompt(messages)
    assert exc_info.value.status_code == 400


def test_chat_to_completion_payload_drops_chat_only_fields():
    payload = {
        "model": "evo-1.5-8k-base",
        "messages": [{"role": "user", "content": "ATCG"}],
        "stream": True,
        "stream_options": {"include_usage": True},
        "max_tokens": None,
        "temperature": 0.5,
    }
    result = chat_to_completion_payload(payload)
    assert result == {
        "model": "evo-1.5-8k-base",
        "temperature": 0.5,
        "prompt": "ATCG",
    }


def test_completion_to_chat_maps_choices_and_usage():
    data = {
        "id": "cmpl-123",
        "created": 42,
        "model": "evo-1.5-8k-base",
        "choices": [{"text": "GATTACA", "finish_reason": "stop"}],
        "usage": {"total_tokens": 7},
    }
    chat = completion_to_chat(data)
    assert chat["id"] == "cmpl-123"
    assert chat["object"] == "chat.completion"
    assert chat["choices"][0]["message"] == {"role": "assistant", "content": "GATTACA"}
    assert chat["choices"][0]["finish_reason"] == "stop"
    assert chat["usage"] == {"total_tokens": 7}


def test_completion_to_chat_generates_id_when_absent():
    chat = completion_to_chat({"choices": []})
    assert chat["id"].startswith("chatcmpl-")


def test_chat_to_sse_ends_with_done_sentinel():
    chat = {
        "id": "chatcmpl-1",
        "created": 1,
        "model": "evo-1.5-8k-base",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "GATTACA"},
                "finish_reason": "stop",
            }
        ],
    }
    events = list(chat_to_sse(chat))
    assert events[-1] == "data: [DONE]\n\n"
    assert "GATTACA" in events[0]
    assert '"finish_reason": "stop"' in events[1]
