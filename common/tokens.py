"""
Gestion des utilisateurs finaux et de leur quota.

Ce module n'est utilisé que par la gateway : c'est elle, et elle seule, qui
authentifie les utilisateurs et décompte leur quota. Les serveurs de modèles
ne voient jamais de clé utilisateur (voir common/internal.py).
"""
import json
import secrets
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from .errors import invalid_api_key


class TokenManager:
    """Gère les utilisateurs et leur quota de tokens via un fichier JSON."""

    def __init__(self, filename: str):
        """
        Initialise le gestionnaire et charge la base existante.

        Args:
            filename: Chemin du fichier JSON servant de base de données.
        """
        self.filename = filename
        self.db = self._load()

    def _load(self) -> Dict:
        """
        Charge la base de tokens depuis le fichier JSON.

        Returns:
            Le contenu de la base sous forme de dict, ou un dict vide
            si le fichier n'existe pas encore.
        """
        if Path(self.filename).exists():
            with open(self.filename, 'r') as f:
                return json.load(f)
        return {}

    def _save(self):
        """Sauvegarde l'état actuel de la base dans le fichier JSON."""
        with open(self.filename, 'w') as f:
            json.dump(self.db, f, indent=2)

    def create_user(self, email: str, quota_tokens: int = 10000) -> str:
        """
        Crée un nouvel utilisateur avec une clé API générée aléatoirement.

        Args:
            email: Email de l'utilisateur.
            quota_tokens: Quota de tokens alloué à l'utilisateur.

        Returns:
            La clé API générée pour cet utilisateur (préfixe sk-).
        """
        api_key = f"sk-{secrets.token_hex(32)}"
        self.db[api_key] = {
            "email": email,
            "created_at": datetime.now().isoformat(),
            "quota_tokens": quota_tokens,
            "used_tokens": 0,
            "requests": 0
        }
        self._save()
        return api_key

    def verify_key(self, api_key: str) -> Optional[Dict]:
        """
        Vérifie si une clé API existe dans la base.

        Args:
            api_key: Clé API à vérifier.

        Returns:
            Les données de l'utilisateur si la clé est valide, sinon None.
        """
        return self.db.get(api_key)

    def deduct_tokens(self, api_key: str, tokens_used: int) -> bool:
        """
        Déduit des tokens du quota d'un utilisateur.

        Args:
            api_key: Clé API de l'utilisateur.
            tokens_used: Nombre de tokens à déduire.

        Returns:
            True si la déduction a réussi, False si la clé est invalide
            ou si le quota restant est insuffisant.
        """
        if api_key not in self.db:
            return False

        user = self.db[api_key]
        remaining = user["quota_tokens"] - user["used_tokens"]

        if remaining < tokens_used:
            return False

        user["used_tokens"] += tokens_used
        user["requests"] += 1
        self._save()
        return True

    def get_status(self, api_key: str) -> Optional[Dict]:
        """
        Récupère le statut d'utilisation d'un utilisateur.

        Args:
            api_key: Clé API de l'utilisateur.

        Returns:
            Un dict contenant email, quota, tokens utilisés/restants,
            nombre de requêtes et date de création, ou None si la clé
            est invalide.
        """
        if api_key not in self.db:
            return None

        user = self.db[api_key]
        return {
            "email": user["email"],
            "quota_tokens": user["quota_tokens"],
            "used_tokens": user["used_tokens"],
            "remaining_tokens": user["quota_tokens"] - user["used_tokens"],
            "requests_made": user["requests"],
            "created_at": user["created_at"]
        }


def extract_bearer_token(authorization: Optional[str]) -> str:
    """
    Extrait la clé API d'un header Authorization au format Bearer.

    Args:
        authorization: Valeur brute du header Authorization
            (attendu sous la forme "Bearer sk-...").

    Returns:
        La clé API extraite.

    Raises:
        HTTPException: 401 si le header est absent ou mal formé.
    """
    if not authorization or not authorization.startswith("Bearer "):
        raise invalid_api_key(
            "Missing or malformed Authorization header. "
            "Expected format: Bearer <api_key>"
        )
    return authorization.removeprefix("Bearer ").strip()


def authenticate(token_manager: TokenManager, authorization: Optional[str]) -> tuple[str, Dict]:
    """
    Authentifie une requête à partir du header Authorization.

    Args:
        token_manager: Base des utilisateurs dans laquelle chercher la clé.
        authorization: Valeur brute du header Authorization.

    Returns:
        Un tuple (clé API, données de l'utilisateur).

    Raises:
        HTTPException: 401 si la clé API est manquante ou invalide.
    """
    api_key = extract_bearer_token(authorization)
    user = token_manager.verify_key(api_key)
    if not user:
        raise invalid_api_key()
    return api_key, user
