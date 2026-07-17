"""
Secret d'administration de la gateway.

Distinct du secret interne (internal.py) : celui-ci protège la gateway des
serveurs de modèles, celui-là protège les endpoints d'administration de la
gateway — aujourd'hui la création de clés API — de tout le reste du monde.

Les deux ne peuvent pas être confondus : le secret interne est connu des six
conteneurs, donc un serveur de modèle compromis pourrait s'auto-créer des
clés utilisateur s'il servait aussi d'admin.

Mêmes deux sources que le secret interne, dans le même ordre :
  1. BIOAI_ADMIN_KEY — utilisée par docker compose, injectée depuis .env.
  2. Le fichier .admin_key à la racine du repo, généré à la première
     utilisation (chemin bare-metal, rien à configurer).
"""
import os
import secrets
from pathlib import Path
from typing import Optional

from fastapi import Header

from .errors import invalid_api_key
from .tokens import extract_bearer_token

ENV_VAR = "BIOAI_ADMIN_KEY"
KEY_FILE = Path(__file__).resolve().parent.parent / ".admin_key"

_cached_key: Optional[str] = None


def get_admin_key() -> str:
    """
    Récupère le secret d'administration : depuis BIOAI_ADMIN_KEY si elle est
    définie, sinon depuis le fichier .admin_key, généré si absent.

    Returns:
        Le secret d'administration.
    """
    global _cached_key
    if _cached_key is not None:
        return _cached_key

    from_env = os.getenv(ENV_VAR)
    if from_env:
        _cached_key = from_env.strip()
        return _cached_key

    if not KEY_FILE.exists():
        key = secrets.token_hex(32)
        try:
            fd = os.open(KEY_FILE, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            with os.fdopen(fd, "w") as f:
                f.write(key)
        except FileExistsError:
            pass  # un autre processus a gagné la course, on lira le sien

    _cached_key = KEY_FILE.read_text().strip()
    return _cached_key


def require_admin_key(authorization: Optional[str] = Header(None)):
    """
    Dépendance FastAPI : refuse toute requête qui ne porte pas le secret
    d'administration.

    Le Header(None) explicite est indispensable : sans lui, FastAPI lirait
    `authorization` comme un paramètre de query et renverrait un 422 au lieu
    de vérifier le header.

    Args:
        authorization: Valeur brute du header Authorization.

    Raises:
        HTTPException: 401 si le secret est absent ou incorrect.
    """
    provided = extract_bearer_token(authorization)
    if not secrets.compare_digest(provided, get_admin_key()):
        raise invalid_api_key(
            "This endpoint requires the gateway admin key, not a user API key."
        )
