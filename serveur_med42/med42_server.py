import os
import sys
import time
import uuid
from pathlib import Path
from datetime import datetime
from typing import Dict, List
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

MODEL_NAME = "med42-8b"

# Le modèle n'est pas chargé en mémoire ici : ce serveur ne fait que router
# vers Ollama, qui le fait tourner (ollama pull thewindmom/llama3-med42-8b,
# puis ollama serve, par défaut sur localhost:11434). Pourquoi ce modèle en
# particulier : voir le README.
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "thewindmom/llama3-med42-8b")

# Fenêtre de contexte du modèle : 8K. Au-delà, Ollama tronque silencieusement
# ou rame ; on plafonne max_tokens ici.
MAX_TOKENS_LIMIT = 8192

# Un Modelfile Ollama mal packagé peut déclarer un stop token corrompu (par
# ex. des guillemets littéraux collés autour, vérifiable via `ollama show`)
# et faire dériver le modèle en fin de génération faute de jeton de fin de
# tour reconnu. On force ces stop tokens à chaque appel plutôt que de
# dépendre du Modelfile du paquet Ollama utilisé — ce sont les tokens
# standard de tout modèle Llama-3-Instruct, donc sans effet quand le
# Modelfile est correct.
LLAMA3_STOP_TOKENS = ["<|eot_id|>", "<|start_header_id|>", "<|end_header_id|>"]


# ============ APPEL DU BACKEND (Ollama) ============

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
                    "stop": LLAMA3_STOP_TOKENS,
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


def call_ollama_chat(messages: List[Dict], max_tokens: int, temperature: float, top_p: float) -> Dict:
    """
    Génère une réponse de chat via l'API native /api/chat d'Ollama.

    À la différence de call_ollama (/api/generate, un seul prompt texte),
    cette fonction transmet le tableau de messages tel quel. C'est important :
    le modèle attend un historique structuré (chaque tour balisé selon le
    template Llama-3-Instruct), pas une conversation aplatie en un unique
    bloc de texte. Aplatie, la conversation ressemble à un script que le
    modèle se contente d'imiter (réponses courtes, questions qui reviennent
    en boucle) plutôt qu'à un dialogue dont il tient l'état.

    Args:
        messages: Historique de la conversation, au format OpenAI
            ({"role": "system"|"user"|"assistant", "content": str}) — ce sont
            aussi les noms de rôles attendus par Ollama, aucune conversion
            n'est nécessaire.
        max_tokens: Nombre maximum de tokens à générer.
        temperature: Température d'échantillonnage.
        top_p: Top-p (nucleus sampling).

    Returns:
        Un dict avec les clés "content", "prompt_tokens" et "completion_tokens".

    Raises:
        HTTPException: 500 si l'appel à Ollama échoue.
    """
    try:
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json={
                "model": OLLAMA_MODEL,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "top_p": top_p,
                    "num_predict": max_tokens,
                    "stop": LLAMA3_STOP_TOKENS,
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
        "content": data.get("message", {}).get("content", ""),
        "prompt_tokens": data.get("prompt_eval_count", 0),
        "completion_tokens": data.get("eval_count", 0),
    }


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


class ChatMessage(BaseModel):
    """Un message de conversation, au format OpenAI."""
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    """Corps de requête pour l'endpoint /v1/chat/completions, format OpenAI."""
    model: str = MODEL_NAME
    messages: List[ChatMessage]
    max_tokens: int = Field(default=256, gt=0, le=MAX_TOKENS_LIMIT)
    temperature: float = 0.7
    top_p: float = 0.95

# ============ FASTAPI APP ============

app = FastAPI(
    title="Med42 API",
    description="API compatible OpenAI pour dialoguer en biomédical avec Med42-v2 8B, "
                "servie via Ollama",
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

    Relaie la génération à Ollama.

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
        result = call_ollama(
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


@app.post("/v1/chat/completions", dependencies=[Depends(require_internal_key)])
def create_chat_completion(request: ChatCompletionRequest):
    """
    Génère une réponse de chat à partir d'un historique de messages (format OpenAI).

    Contrairement à /v1/completions, qui reçoit un prompt déjà aplati par la
    gateway, cet endpoint reçoit l'historique structuré et le relaie tel quel
    à Ollama (voir call_ollama_chat) : c'est ce qui permet au modèle de tenir
    correctement une conversation à plusieurs tours.

    L'authentification de l'utilisateur final et son quota sont gérés par la
    gateway : cet endpoint n'accepte que le secret interne.

    Args:
        request: Historique de conversation et paramètres de génération
            (model, messages, max_tokens, temperature, top_p).

    Returns:
        Un dict au format OpenAI chat.completion (id, choices, usage, etc.).

    Raises:
        HTTPException: 401 si le secret interne est absent ou incorrect,
            500 si le backend est injoignable ou en cas d'erreur interne.
    """
    result = call_ollama_chat(
        [m.model_dump() for m in request.messages],
        request.max_tokens,
        request.temperature,
        request.top_p,
    )
    tokens_used = result["prompt_tokens"] + result["completion_tokens"]

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": result["content"]},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": result["prompt_tokens"],
            "completion_tokens": result["completion_tokens"],
            "total_tokens": tokens_used,
        },
    }


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
                "owned_by": "m42-health-via-ollama"
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
        Un dict avec le statut, le nom du modèle, l'URL d'Ollama
        et un timestamp.
    """
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "backend": "ollama",
        "backend_url": OLLAMA_BASE_URL,
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
        "name": "Med42 API",
        "version": "0.1.0",
        "model": MODEL_NAME,
        "backend": "ollama",
        "note": "Serveur interne : les utilisateurs passent par la gateway "
                "(port 8080), qui gère les clés API et les quotas.",
        "endpoints": {
            "POST /v1/completions": "Générer du texte à partir d'un prompt (format OpenAI, secret interne requis)",
            "POST /v1/chat/completions": "Générer une réponse à partir d'un historique de messages (format OpenAI, secret interne requis)",
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
