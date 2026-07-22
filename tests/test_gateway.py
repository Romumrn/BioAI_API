"""
Tests d'intégration des endpoints de la gateway (auth, quotas, routage).

Aucun test ici ne parle à un vrai serveur de modèle : `requests.post` et
`requests.get` sont mockés, la gateway étant le seul composant testé.
"""
import requests
import pytest
from fastapi.testclient import TestClient

import gateway
from gateway import app

client = TestClient(app)


class FakeResponse:
    def __init__(self, status_code=200, json_data=None):
        self.status_code = status_code
        self.ok = status_code < 400
        self._json_data = json_data or {}

    def json(self):
        return self._json_data


def test_root():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json()["name"] == "BioAI Gateway"


def test_health_reports_unreachable_backends(monkeypatch):
    def fake_get(*args, **kwargs):
        raise requests.RequestException("connection refused")

    monkeypatch.setattr(gateway.requests, "get", fake_get)
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert all(s == {"status": "unreachable"} for s in body["backends"].values())


def test_list_models_requires_auth():
    r = client.get("/v1/models")
    assert r.status_code == 401


def test_list_models_with_valid_key(make_user):
    api_key = make_user()
    r = client.get("/v1/models", headers={"Authorization": f"Bearer {api_key}"})
    assert r.status_code == 200
    model_ids = {m["id"] for m in r.json()["data"]}
    assert "dnabert2-117m" in model_ids


# ============ /v1/embeddings, /v1/completions : auth, modèle, quota ============


def test_embeddings_missing_auth():
    r = client.post("/v1/embeddings", json={"model": "dnabert2-117m", "input": "ATCG"})
    assert r.status_code == 401


def test_embeddings_invalid_json_body():
    r = client.post(
        "/v1/embeddings",
        content=b"not json",
        headers={"Authorization": "Bearer sk-whatever", "Content-Type": "application/json"},
    )
    assert r.status_code == 400


def test_embeddings_unknown_model(make_user):
    api_key = make_user()
    r = client.post(
        "/v1/embeddings",
        json={"model": "no-such-model", "input": "ATCG"},
        headers={"Authorization": f"Bearer {api_key}"},
    )
    assert r.status_code == 400


def test_embeddings_wrong_kind_for_model(make_user):
    # med42-8b est "completions", pas "embeddings"
    api_key = make_user()
    r = client.post(
        "/v1/embeddings",
        json={"model": "med42-8b", "input": "ATCG"},
        headers={"Authorization": f"Bearer {api_key}"},
    )
    assert r.status_code == 400


def test_embeddings_quota_exceeded(make_user):
    # MIN_REMAINING["embeddings"] == 10
    api_key = make_user(quota_tokens=5)
    r = client.post(
        "/v1/embeddings",
        json={"model": "dnabert2-117m", "input": "ATCG"},
        headers={"Authorization": f"Bearer {api_key}"},
    )
    assert r.status_code == 429


def test_embeddings_success_deducts_tokens(make_user, token_manager, monkeypatch):
    api_key = make_user(quota_tokens=1000)

    def fake_post(url, json=None, headers=None, timeout=None):
        assert url == f"{gateway.BACKENDS['dnabert2-117m']['url']}/v1/embeddings"
        return FakeResponse(
            200,
            {
                "data": [{"object": "embedding", "embedding": [0.1, 0.2], "index": 0}],
                "usage": {"prompt_tokens": 3, "total_tokens": 3},
            },
        )

    monkeypatch.setattr(gateway.requests, "post", fake_post)

    r = client.post(
        "/v1/embeddings",
        json={"model": "dnabert2-117m", "input": "ATCG"},
        headers={"Authorization": f"Bearer {api_key}"},
    )
    assert r.status_code == 200
    assert r.json()["usage"]["total_tokens"] == 3
    assert token_manager.verify_key(api_key)["used_tokens"] == 3


def test_completions_backend_unreachable_returns_500(make_user, monkeypatch):
    api_key = make_user(quota_tokens=1000)

    def fake_post(*args, **kwargs):
        raise requests.RequestException("connection refused")

    monkeypatch.setattr(gateway.requests, "post", fake_post)

    r = client.post(
        "/v1/completions",
        json={"model": "evo-1.5-8k-base", "prompt": "ATCG"},
        headers={"Authorization": f"Bearer {api_key}"},
    )
    assert r.status_code == 500


def test_completions_backend_error_is_unwrapped(make_user, monkeypatch):
    api_key = make_user(quota_tokens=1000)

    def fake_post(*args, **kwargs):
        return FakeResponse(422, {"detail": {"error": {"message": "bad prompt"}}})

    monkeypatch.setattr(gateway.requests, "post", fake_post)

    r = client.post(
        "/v1/completions",
        json={"model": "evo-1.5-8k-base", "prompt": "ATCG"},
        headers={"Authorization": f"Bearer {api_key}"},
    )
    assert r.status_code == 422
    assert r.json()["detail"] == {"error": {"message": "bad prompt"}}


# ============ /v1/chat/completions : traduction selon le backend ============


def test_chat_completions_translates_for_non_chat_capable_backend(make_user, monkeypatch):
    api_key = make_user(quota_tokens=1000)
    captured = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        captured["url"] = url
        captured["payload"] = json
        return FakeResponse(
            200,
            {
                "id": "cmpl-1",
                "created": 1,
                "model": "evo-1.5-8k-base",
                "choices": [{"index": 0, "text": "GATTACA", "finish_reason": "stop"}],
                "usage": {"total_tokens": 5},
            },
        )

    monkeypatch.setattr(gateway.requests, "post", fake_post)

    r = client.post(
        "/v1/chat/completions",
        json={
            "model": "evo-1.5-8k-base",
            "messages": [{"role": "user", "content": "ATCG"}],
        },
        headers={"Authorization": f"Bearer {api_key}"},
    )
    assert r.status_code == 200
    # Evo n'est pas chat_capable : la gateway doit avoir traduit vers /v1/completions.
    assert captured["url"].endswith("/v1/completions")
    assert captured["payload"]["prompt"] == "ATCG"
    assert "messages" not in captured["payload"]

    body = r.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["content"] == "GATTACA"


def test_chat_completions_passthrough_for_chat_capable_backend(make_user, monkeypatch):
    api_key = make_user(quota_tokens=1000)
    captured = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        captured["url"] = url
        captured["payload"] = json
        return FakeResponse(
            200,
            {
                "id": "chatcmpl-1",
                "object": "chat.completion",
                "created": 1,
                "model": "med42-8b",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Bonjour"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"total_tokens": 4},
            },
        )

    monkeypatch.setattr(gateway.requests, "post", fake_post)

    r = client.post(
        "/v1/chat/completions",
        json={
            "model": "med42-8b",
            "messages": [{"role": "user", "content": "Salut"}],
        },
        headers={"Authorization": f"Bearer {api_key}"},
    )
    assert r.status_code == 200
    # med42-8b est chat_capable : les messages doivent être relayés tels quels.
    assert captured["url"].endswith("/v1/chat/completions")
    assert captured["payload"]["messages"] == [{"role": "user", "content": "Salut"}]
    assert r.json()["choices"][0]["message"]["content"] == "Bonjour"


def test_chat_completions_stream_returns_sse(make_user, monkeypatch):
    api_key = make_user(quota_tokens=1000)

    def fake_post(*args, **kwargs):
        return FakeResponse(
            200,
            {
                "id": "cmpl-1",
                "created": 1,
                "model": "evo-1.5-8k-base",
                "choices": [{"index": 0, "text": "GATTACA", "finish_reason": "stop"}],
                "usage": {"total_tokens": 5},
            },
        )

    monkeypatch.setattr(gateway.requests, "post", fake_post)

    r = client.post(
        "/v1/chat/completions",
        json={
            "model": "evo-1.5-8k-base",
            "messages": [{"role": "user", "content": "ATCG"}],
            "stream": True,
        },
        headers={"Authorization": f"Bearer {api_key}"},
    )
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    assert r.text.strip().endswith("data: [DONE]")


# ============ /v1/api-keys : réservé à l'admin ============


def test_create_api_key_requires_admin_key():
    r = client.post("/v1/api-keys", params={"email": "a@test.com"})
    assert r.status_code == 401


def test_create_api_key_rejects_wrong_admin_key():
    r = client.post(
        "/v1/api-keys",
        params={"email": "a@test.com"},
        headers={"Authorization": "Bearer wrong-key"},
    )
    assert r.status_code == 401


def test_create_api_key_success(token_manager):
    r = client.post(
        "/v1/api-keys",
        params={"email": "a@test.com", "quota_tokens": 2000},
        headers={"Authorization": "Bearer test-admin-key"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["email"] == "a@test.com"
    assert body["quota_tokens"] == 2000
    assert token_manager.verify_key(body["api_key"]) is not None


def test_create_api_key_unlimited_ignores_quota(token_manager):
    r = client.post(
        "/v1/api-keys",
        params={"email": "a@test.com", "quota_tokens": 2000, "unlimited": True},
        headers={"Authorization": "Bearer test-admin-key"},
    )
    assert r.status_code == 200
    assert r.json()["quota_tokens"] is None


def test_api_key_status(make_user, token_manager):
    api_key = make_user(quota_tokens=1000)
    token_manager.deduct_tokens(api_key, 100)

    r = client.get("/v1/api-keys/status", headers={"Authorization": f"Bearer {api_key}"})
    assert r.status_code == 200
    body = r.json()
    assert body["used_tokens"] == 100
    assert body["remaining_tokens"] == 900


def test_api_key_status_requires_valid_key():
    r = client.get("/v1/api-keys/status", headers={"Authorization": "Bearer sk-unknown"})
    assert r.status_code == 401
