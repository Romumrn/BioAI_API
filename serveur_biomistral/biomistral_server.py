import os
import sys
import time
import uuid
from pathlib import Path
from datetime import datetime
from typing import Dict
import requests
from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Le package common/ vit à la racine du repo, un cran au-dessus de ce dossier.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import (  # noqa: E402
    CompletionChoice,
    CompletionResponse,
    CompletionUsage,
    require_internal_key,
    server_error,
)

# ============ CONFIGURATION ============

MODEL_NAME = "biomistral-7b"

# BioMistral n'est pas chargé en mémoire ici : ce serveur ne fait que router
# vers un backend qui, lui, sait le faire tourner. Deux backends supportés :
#   - "ollama" : ollama pull cniongolo/biomistral (ou adrienbrault/biomistral-7b),
#                puis ollama serve (par défaut sur localhost:11434)
#   - "vllm"   : vllm serve BioMistral/BioMistral-7B --port 8000,
#                qui expose déjà une API compatible OpenAI qu'on relaie telle quelle
BACKEND = os.getenv("BIOMISTRAL_BACKEND", "ollama")  # "ollama" ou "vllm"

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "cniongolo/biomistral")

VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8000")
VLLM_MODEL = os.getenv("VLLM_MODEL", "BioMistral/BioMistral-7B")

# Fenêtre de contexte Ollama par défaut (OLLAMA_CONTEXT_LENGTH=4096). Au-delà,
# le backend tronque silencieusement ou rame ; on plafonne max_tokens ici.
MAX_TOKENS_LIMIT = 4096


# ============ APPEL DU BACKEND (Ollama ou vLLM) ============

def call_ollama(prompt: str, max_tokens: int, temperature: float, top_p: float) -> Dict:
    """
    Génère une complétion via l'API native d'Ollama.

    Args:
        prompt: Texte d'entrée.
        max_tokens: Nombre maximum de tokens à générer.
        temperature: Température d'échantillonnage.
        top_p: Top-p (nucleus sampling).

    Returns:
        Un dict avec les clés "text", "prompt_tokens" et "completion_tokens".

    Raises:
        HTTPException: 500 si l'appel à Ollama échoue (serveur non démarré,
            modèle non pull, etc.).
    """
    try:
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "top_p": top_p,
                    "num_predict": max_tokens,
                },
            },
            timeout=300,
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        raise server_error(
            f"Ollama backend unreachable ({OLLAMA_BASE_URL}): {e}",
            code="backend_unavailable",
        )

    return {
        "text": data.get("response", ""),
        "prompt_tokens": data.get("prompt_eval_count", 0),
        "completion_tokens": data.get("eval_count", 0),
    }


def call_vllm(prompt: str, max_tokens: int, temperature: float, top_p: float) -> Dict:
    """
    Génère une complétion via l'API compatible OpenAI exposée par vLLM.

    Args:
        prompt: Texte d'entrée.
        max_tokens: Nombre maximum de tokens à générer.
        temperature: Température d'échantillonnage.
        top_p: Top-p (nucleus sampling).

    Returns:
        Un dict avec les clés "text", "prompt_tokens" et "completion_tokens".

    Raises:
        HTTPException: 500 si l'appel à vLLM échoue.
    """
    try:
        resp = requests.post(
            f"{VLLM_BASE_URL}/v1/completions",
            json={
                "model": VLLM_MODEL,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
            },
            timeout=300,
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        raise server_error(
            f"vLLM backend unreachable ({VLLM_BASE_URL}): {e}",
            code="backend_unavailable",
        )

    choice = data["choices"][0]
    usage = data.get("usage", {})
    return {
        "text": choice.get("text", ""),
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
    }


def call_backend(prompt: str, max_tokens: int, temperature: float, top_p: float) -> Dict:
    """
    Dispatche l'appel de génération vers le backend configuré (BACKEND).

    Args:
        prompt: Texte d'entrée.
        max_tokens: Nombre maximum de tokens à générer.
        temperature: Température d'échantillonnage.
        top_p: Top-p (nucleus sampling).

    Returns:
        Un dict avec les clés "text", "prompt_tokens" et "completion_tokens".
    """
    if BACKEND == "vllm":
        return call_vllm(prompt, max_tokens, temperature, top_p)
    return call_ollama(prompt, max_tokens, temperature, top_p)


# ============ MODELS FASTAPI (format OpenAI) ============
# Les schémas de réponse sont partagés (common/schemas.py) ; seule la requête
# est spécifique, ses bornes dépendant du modèle.

class CompletionRequest(BaseModel):
    """Corps de requête pour l'endpoint /v1/completions, format OpenAI."""
    model: str = MODEL_NAME
    prompt: str
    max_tokens: int = Field(default=256, gt=0, le=MAX_TOKENS_LIMIT)
    temperature: float = 0.7
    top_p: float = 0.95
    n: int = 1                    # nombre de complétions à générer

# ============ FASTAPI APP ============

app = FastAPI(
    title="BioMistral API",
    description="API compatible OpenAI pour générer du texte biomédical avec BioMistral, "
                "servie via Ollama ou vLLM",
    version="0.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============ ENDPOINTS ============

@app.post(
    "/v1/completions",
    response_model=CompletionResponse,
    dependencies=[Depends(require_internal_key)],
)
def create_completion(request: CompletionRequest):
    """
    Génère du texte à partir d'un prompt (format OpenAI).

    Relaie la génération au backend configuré (Ollama ou vLLM).

    L'authentification de l'utilisateur final et son quota sont gérés par la
    gateway : cet endpoint n'accepte que le secret interne.

    Args:
        request: Paramètres de génération au format OpenAI
            (model, prompt, max_tokens, temperature, top_p, n).

    Returns:
        CompletionResponse au format OpenAI (id, choices, usage, etc.).

    Raises:
        HTTPException: 401 si le secret interne est absent ou incorrect,
            500 si le backend est injoignable ou en cas d'erreur interne.
    """
    choices = []
    total_prompt_tokens = 0
    total_completion_tokens = 0

    for i in range(request.n):
        result = call_backend(
            request.prompt, request.max_tokens, request.temperature, request.top_p
        )
        total_prompt_tokens += result["prompt_tokens"]
        total_completion_tokens += result["completion_tokens"]
        choices.append(
            CompletionChoice(text=result["text"], index=i, finish_reason="stop")
        )

    tokens_used = total_prompt_tokens + total_completion_tokens

    return CompletionResponse(
        id=f"cmpl-{uuid.uuid4().hex}",
        created=int(time.time()),
        model=request.model,
        choices=choices,
        usage=CompletionUsage(
            prompt_tokens=total_prompt_tokens,
            completion_tokens=total_completion_tokens,
            total_tokens=tokens_used
        )
    )


@app.get("/v1/models", dependencies=[Depends(require_internal_key)])
async def list_models():
    """
    Liste les modèles disponibles (format OpenAI).

    Returns:
        Un dict au format OpenAI listant les modèles disponibles.

    Raises:
        HTTPException: 401 si le secret interne est absent ou incorrect.
    """
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_NAME,
                "object": "model",
                "created": 0,
                "owned_by": f"biomistral-via-{BACKEND}"
            }
        ]
    }


@app.get("/health")
async def health():
    """
    Vérifie l'état du service.

    Ouvert sans authentification : start_all.py et la gateway s'en servent
    pour attendre que le service soit prêt.

    Returns:
        Un dict avec le statut, le nom du modèle, le backend utilisé
        et un timestamp.
    """
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "backend": BACKEND,
        "backend_url": OLLAMA_BASE_URL if BACKEND == "ollama" else VLLM_BASE_URL,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/")
async def root():
    """
    Retourne les informations générales de l'API.

    Returns:
        Un dict décrivant l'API et listant ses endpoints disponibles.
    """
    return {
        "name": "BioMistral API",
        "version": "0.1.0",
        "model": MODEL_NAME,
        "backend": BACKEND,
        "note": "Serveur interne : les utilisateurs passent par la gateway "
                "(port 8080), qui gère les clés API et les quotas.",
        "endpoints": {
            "POST /v1/completions": "Générer du texte (format OpenAI, secret interne requis)",
            "GET /v1/models": "Lister les modèles disponibles (secret interne requis)",
            "GET /health": "Health check"
        },
        "docs": "http://localhost:8002/docs"
    }

if __name__ == "__main__":
    import uvicorn
    # Défaut 127.0.0.1 : en bare-metal, ce serveur n'est pas censé être
    # joignable depuis l'extérieur, la gateway est la seule porte d'entrée
    # publique. En conteneur, docker compose passe 0.0.0.0 — indispensable
    # pour que la gateway le joigne depuis son propre namespace réseau — et
    # l'isolation vient de l'absence de publication du port vers l'hôte.
    uvicorn.run(app, host=os.getenv("BIOAI_BIND_HOST", "127.0.0.1"), port=8002)
