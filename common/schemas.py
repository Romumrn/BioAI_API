"""Schémas de requête/réponse au format OpenAI, partagés par les serveurs de modèles."""
from typing import List, Union

from pydantic import BaseModel


# ============ /v1/embeddings ============

class EmbeddingsRequest(BaseModel):
    """Corps de requête pour l'endpoint /v1/embeddings, format OpenAI."""
    model: str
    input: Union[str, List[str]]


class EmbeddingData(BaseModel):
    """Un vecteur d'embedding, au format OpenAI."""
    object: str = "embedding"
    embedding: List[float]
    index: int


class EmbeddingsUsage(BaseModel):
    """Décompte des tokens consommés, au format OpenAI."""
    prompt_tokens: int
    total_tokens: int


class EmbeddingsResponse(BaseModel):
    """Corps de réponse pour l'endpoint /v1/embeddings, format OpenAI."""
    object: str = "list"
    data: List[EmbeddingData]
    model: str
    usage: EmbeddingsUsage


# ============ /v1/completions ============

class CompletionChoice(BaseModel):
    """Une complétion générée, au format OpenAI."""
    text: str
    index: int
    finish_reason: str


class CompletionUsage(BaseModel):
    """Décompte des tokens consommés, au format OpenAI."""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class CompletionResponse(BaseModel):
    """Corps de réponse pour l'endpoint /v1/completions, format OpenAI."""
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: List[CompletionChoice]
    usage: CompletionUsage
