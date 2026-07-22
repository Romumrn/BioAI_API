"""
Configuration partagée des tests de la gateway.

Les variables d'environnement doivent être posées avant le premier import de
`gateway` ou de `common` : `common.admin.get_admin_key` et
`common.internal.get_internal_key` mettent leur valeur en cache au premier
appel, et `gateway.py` instancie `token_manager` au chargement du module. Un
conftest.py est importé par pytest avant tout module de test, donc avant ces
imports — c'est ce qui garantit que les tests ne touchent jamais les fichiers
réels du repo (tokens_db.json, .admin_key, .internal_key).
"""
import os
import tempfile

_tmp_dir = tempfile.mkdtemp(prefix="bioai_test_")
os.environ["BIOAI_TOKENS_FILE"] = os.path.join(_tmp_dir, "tokens_db.json")
os.environ["BIOAI_ADMIN_KEY"] = "test-admin-key"
os.environ["BIOAI_INTERNAL_KEY"] = "test-internal-key"

import pytest  # noqa: E402

import gateway  # noqa: E402
from common import TokenManager  # noqa: E402


@pytest.fixture
def token_manager(tmp_path, monkeypatch):
    """Un TokenManager frais, isolé par test, branché à la place du singleton."""
    tm = TokenManager(str(tmp_path / "tokens.json"))
    monkeypatch.setattr(gateway, "token_manager", tm)
    return tm


@pytest.fixture
def make_user(token_manager):
    """Crée un utilisateur de test et retourne sa clé API."""

    def _make_user(email="user@test.com", quota_tokens=10000):
        return token_manager.create_user(email, quota_tokens)

    return _make_user
