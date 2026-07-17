"""Erreurs HTTP au format OpenAI, partagées par la gateway et les sous-serveurs."""
from fastapi import HTTPException


def api_error(status_code: int, message: str, type_: str, code: str) -> HTTPException:
    """
    Construit une HTTPException dont le corps suit le format d'erreur OpenAI.

    Args:
        status_code: Code HTTP à renvoyer.
        message: Message lisible destiné à l'appelant.
        type_: Catégorie d'erreur OpenAI (ex: "invalid_request_error").
        code: Code d'erreur OpenAI (ex: "invalid_api_key").

    Returns:
        L'exception à lever.
    """
    return HTTPException(
        status_code=status_code,
        detail={"error": {"message": message, "type": type_, "code": code}},
    )


def invalid_api_key(message: str = "Incorrect API key provided.") -> HTTPException:
    """Erreur 401 pour une clé API manquante, mal formée ou inconnue."""
    return api_error(401, message, "invalid_request_error", "invalid_api_key")


def insufficient_quota(message: str) -> HTTPException:
    """Erreur 429 pour un quota épuisé."""
    return api_error(429, message, "insufficient_quota", "insufficient_quota")


def context_length_exceeded(message: str) -> HTTPException:
    """Erreur 400 pour une entrée plus longue que la fenêtre de contexte du modèle."""
    return api_error(400, message, "invalid_request_error", "context_length_exceeded")


def server_error(message: str, code: str = "internal_error") -> HTTPException:
    """Erreur 500 générique."""
    return api_error(500, message, "server_error", code)
