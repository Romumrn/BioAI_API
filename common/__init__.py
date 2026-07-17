"""
Code partagé entre la gateway et les serveurs de modèles.

Ce package ne dépend que de la stdlib, fastapi et pydantic, pour rester
importable depuis les venvs isolés de certains sous-serveurs (.venv_nt,
.venv_dnabert2) qui n'ont pas les dépendances des autres.

Répartition des rôles :
  - tokens.py   : utilisateurs finaux et quotas. Gateway uniquement.
  - admin.py    : secret d'administration de la gateway. Gateway uniquement.
  - internal.py : secret gateway -> serveurs de modèles. Les deux côtés.
  - schemas.py  : corps de requête/réponse au format OpenAI.
  - errors.py   : erreurs HTTP au format OpenAI.
"""
from .admin import get_admin_key, require_admin_key
from .errors import (
    api_error,
    context_length_exceeded,
    insufficient_quota,
    invalid_api_key,
    server_error,
)
from .internal import get_internal_key, require_internal_key
from .schemas import (
    CompletionChoice,
    CompletionResponse,
    CompletionUsage,
    EmbeddingData,
    EmbeddingsRequest,
    EmbeddingsResponse,
    EmbeddingsUsage,
)
from .tokens import TokenManager, authenticate, extract_bearer_token, remaining_tokens

__all__ = [
    "api_error",
    "authenticate",
    "context_length_exceeded",
    "extract_bearer_token",
    "get_admin_key",
    "get_internal_key",
    "insufficient_quota",
    "invalid_api_key",
    "remaining_tokens",
    "require_admin_key",
    "require_internal_key",
    "server_error",
    "CompletionChoice",
    "CompletionResponse",
    "CompletionUsage",
    "EmbeddingData",
    "EmbeddingsRequest",
    "EmbeddingsResponse",
    "EmbeddingsUsage",
    "TokenManager",
]
