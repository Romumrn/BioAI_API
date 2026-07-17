"""
Secret partagé entre la gateway et les serveurs de modèles.

Un serveur de modèle n'est jamais censé être joignable par un utilisateur :
la gateway est la seule porte d'entrée publique. Ce qui l'en empêche dépend
du mode de déploiement, et ce secret est ce qui reste vrai dans les deux :

  - en bare-metal, les serveurs bindent 127.0.0.1 (voir BIOAI_BIND_HOST dans
    chaque serveur) ; le secret est une deuxième barrière, pour le cas où
    cette isolation tombe.
  - en conteneur, les serveurs DOIVENT binder 0.0.0.0 pour que la gateway les
    joigne depuis son propre namespace réseau ; ils ne publient simplement
    pas leur port vers l'hôte. Le secret devient alors la seule barrière entre
    ce qui est sur le réseau compose et les modèles.

Deux sources, dans cet ordre :
  1. BIOAI_INTERNAL_KEY — utilisé par docker compose, qui injecte la même
     valeur (depuis .env) dans les 6 conteneurs. Indispensable ici : chaque
     conteneur a son propre système de fichiers, un fichier partagé à la
     racine du repo n'existerait donc pas d'un conteneur à l'autre.
  2. Le fichier .internal_key à la racine du repo, généré à la première
     utilisation. C'est le chemin bare-metal : rien à configurer, un serveur
     lancé seul fonctionne.
"""
import os
import secrets
from pathlib import Path
from typing import Optional

from fastapi import Header

from .errors import invalid_api_key
from .tokens import extract_bearer_token

ENV_VAR = "BIOAI_INTERNAL_KEY"
KEY_FILE = Path(__file__).resolve().parent.parent / ".internal_key"

_cached_key: Optional[str] = None


def get_internal_key() -> str:
    """
    Récupère le secret interne : depuis BIOAI_INTERNAL_KEY si elle est
    définie, sinon depuis le fichier .internal_key, généré si absent.

    La création du fichier passe par O_EXCL : si plusieurs serveurs démarrent
    en même temps (ce que fait start_all.py), un seul l'écrit et les autres
    relisent le sien plutôt que de l'écraser.

    Returns:
        Le secret interne partagé.
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
            with os.fdopen(fd, 'w') as f:
                f.write(key)
        except FileExistsError:
            pass  # un autre processus a gagné la course, on lira le sien

    _cached_key = KEY_FILE.read_text().strip()
    return _cached_key


def require_internal_key(authorization: Optional[str] = Header(None)):
    """
    Dépendance FastAPI : refuse toute requête qui ne porte pas le secret
    interne. À utiliser sur les endpoints des serveurs de modèles, qui ne
    sont censés être appelés que par la gateway.

    Le Header(None) explicite est indispensable : sans lui, FastAPI lirait
    `authorization` comme un paramètre de query et renverrait un 422 au lieu
    de vérifier le header.

    Args:
        authorization: Valeur brute du header Authorization.

    Raises:
        HTTPException: 401 si le secret est absent ou incorrect.
    """
    provided = extract_bearer_token(authorization)
    if not secrets.compare_digest(provided, get_internal_key()):
        raise invalid_api_key(
            "This endpoint is internal and only accepts calls from the gateway. "
            "End users should call the gateway instead."
        )
