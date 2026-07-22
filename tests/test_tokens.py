"""Tests unitaires pour common/tokens.py (utilisateurs et quotas)."""
import pytest
from fastapi import HTTPException

from common import TokenManager, authenticate, extract_bearer_token, remaining_tokens


def test_remaining_tokens_unlimited_account():
    user = {"quota_tokens": None, "used_tokens": 500}
    assert remaining_tokens(user) is None


def test_remaining_tokens_limited_account():
    user = {"quota_tokens": 1000, "used_tokens": 300}
    assert remaining_tokens(user) == 700


def test_extract_bearer_token_valid():
    assert extract_bearer_token("Bearer sk-abc123") == "sk-abc123"


@pytest.mark.parametrize("header", [None, "", "sk-abc123", "Basic sk-abc123"])
def test_extract_bearer_token_missing_or_malformed(header):
    with pytest.raises(HTTPException) as exc_info:
        extract_bearer_token(header)
    assert exc_info.value.status_code == 401


def test_create_user_generates_unique_prefixed_keys(tmp_path):
    tm = TokenManager(str(tmp_path / "tokens.json"))
    key1 = tm.create_user("a@test.com")
    key2 = tm.create_user("b@test.com")
    assert key1 != key2
    assert key1.startswith("sk-")
    assert key2.startswith("sk-")


def test_create_user_persists_to_disk(tmp_path):
    db_file = tmp_path / "tokens.json"
    tm = TokenManager(str(db_file))
    api_key = tm.create_user("a@test.com", quota_tokens=5000)

    reloaded = TokenManager(str(db_file))
    assert reloaded.verify_key(api_key)["email"] == "a@test.com"
    assert reloaded.verify_key(api_key)["quota_tokens"] == 5000


def test_verify_key_unknown_returns_none(tmp_path):
    tm = TokenManager(str(tmp_path / "tokens.json"))
    assert tm.verify_key("sk-does-not-exist") is None


def test_deduct_tokens_success(tmp_path):
    tm = TokenManager(str(tmp_path / "tokens.json"))
    api_key = tm.create_user("a@test.com", quota_tokens=1000)

    assert tm.deduct_tokens(api_key, 100) is True

    user = tm.verify_key(api_key)
    assert user["used_tokens"] == 100
    assert user["requests"] == 1


def test_deduct_tokens_insufficient_quota(tmp_path):
    tm = TokenManager(str(tmp_path / "tokens.json"))
    api_key = tm.create_user("a@test.com", quota_tokens=50)

    assert tm.deduct_tokens(api_key, 100) is False
    # le quota n'a pas dû bouger sur un refus
    assert tm.verify_key(api_key)["used_tokens"] == 0


def test_deduct_tokens_unlimited_account_never_refused(tmp_path):
    tm = TokenManager(str(tmp_path / "tokens.json"))
    api_key = tm.create_user("a@test.com", quota_tokens=None)

    assert tm.deduct_tokens(api_key, 10 ** 9) is True
    assert tm.verify_key(api_key)["used_tokens"] == 10 ** 9


def test_deduct_tokens_unknown_key(tmp_path):
    tm = TokenManager(str(tmp_path / "tokens.json"))
    assert tm.deduct_tokens("sk-unknown", 10) is False


def test_get_status_unknown_key(tmp_path):
    tm = TokenManager(str(tmp_path / "tokens.json"))
    assert tm.get_status("sk-unknown") is None


def test_get_status_reports_remaining(tmp_path):
    tm = TokenManager(str(tmp_path / "tokens.json"))
    api_key = tm.create_user("a@test.com", quota_tokens=1000)
    tm.deduct_tokens(api_key, 300)

    status = tm.get_status(api_key)
    assert status["quota_tokens"] == 1000
    assert status["used_tokens"] == 300
    assert status["remaining_tokens"] == 700
    assert status["requests_made"] == 1


def test_authenticate_valid_key(tmp_path):
    tm = TokenManager(str(tmp_path / "tokens.json"))
    api_key = tm.create_user("a@test.com")

    returned_key, user = authenticate(tm, f"Bearer {api_key}")
    assert returned_key == api_key
    assert user["email"] == "a@test.com"


def test_authenticate_invalid_key(tmp_path):
    tm = TokenManager(str(tmp_path / "tokens.json"))
    with pytest.raises(HTTPException) as exc_info:
        authenticate(tm, "Bearer sk-does-not-exist")
    assert exc_info.value.status_code == 401


def test_authenticate_missing_header(tmp_path):
    tm = TokenManager(str(tmp_path / "tokens.json"))
    with pytest.raises(HTTPException) as exc_info:
        authenticate(tm, None)
    assert exc_info.value.status_code == 401
