import os
import secrets
import json
import time
import uuid
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict
import requests
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ============ CONFIGURATION ============

MODEL_NAME = "biomistral-7b"
TOKENS_FILE = "tokens_db.json"

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


# ============ GESTION DES TOKENS ============
# (identique à serveur_evo/evo_server.py)

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

token_manager = TokenManager(TOKENS_FILE)


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
        raise HTTPException(
            status_code=401,
            detail={
                "error": {
                    "message": "Missing or malformed Authorization header. "
                                "Expected format: Bearer <api_key>",
                    "type": "invalid_request_error",
                    "code": "invalid_api_key"
                }
            }
        )
    return authorization.removeprefix("Bearer ").strip()


def authenticate(authorization: Optional[str]) -> Dict:
    """
    Authentifie une requête à partir du header Authorization.

    Args:
        authorization: Valeur brute du header Authorization.

    Returns:
        Les données de l'utilisateur associé à la clé API.

    Raises:
        HTTPException: 401 si la clé API est manquante ou invalide.
    """
    api_key = extract_bearer_token(authorization)
    user = token_manager.verify_key(api_key)
    if not user:
        raise HTTPException(
            status_code=401,
            detail={
                "error": {
                    "message": "Incorrect API key provided.",
                    "type": "invalid_request_error",
                    "code": "invalid_api_key"
                }
            }
        )
    return user


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
        raise HTTPException(
            status_code=500,
            detail={
                "error": {
                    "message": f"Ollama backend unreachable ({OLLAMA_BASE_URL}): {e}",
                    "type": "server_error",
                    "code": "backend_unavailable"
                }
            }
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
        raise HTTPException(
            status_code=500,
            detail={
                "error": {
                    "message": f"vLLM backend unreachable ({VLLM_BASE_URL}): {e}",
                    "type": "server_error",
                    "code": "backend_unavailable"
                }
            }
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

class CompletionRequest(BaseModel):
    """Corps de requête pour l'endpoint /v1/completions, format OpenAI."""
    model: str = MODEL_NAME
    prompt: str
    max_tokens: int = Field(default=256, gt=0, le=MAX_TOKENS_LIMIT)
    temperature: float = 0.7
    top_p: float = 0.95
    n: int = 1                    # nombre de complétions à générer

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
    choices: list[CompletionChoice]
    usage: CompletionUsage

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

@app.post("/v1/completions", response_model=CompletionResponse)
def create_completion(
    request: CompletionRequest,
    authorization: str = Header(None)
):
    """
    Génère du texte à partir d'un prompt (format OpenAI).

    Authentifie la requête via le header Authorization (Bearer <api_key>),
    vérifie le quota de tokens restant, relaie la génération au backend
    configuré (Ollama ou vLLM), puis déduit les tokens consommés.

    Args:
        request: Paramètres de génération au format OpenAI
            (model, prompt, max_tokens, temperature, top_p, n).
        authorization: Header Authorization au format "Bearer <api_key>".

    Returns:
        CompletionResponse au format OpenAI (id, choices, usage, etc.).

    Raises:
        HTTPException: 401 si la clé API est manquante ou invalide,
            429 si le quota est dépassé ou insuffisant,
            500 si le backend est injoignable ou en cas d'erreur interne.
    """
    user = authenticate(authorization)
    api_key = extract_bearer_token(authorization)

    remaining = user["quota_tokens"] - user["used_tokens"]
    if remaining < 50:
        raise HTTPException(
            status_code=429,
            detail={
                "error": {
                    "message": f"Quota exceeded. Remaining tokens: {remaining}",
                    "type": "insufficient_quota",
                    "code": "insufficient_quota"
                }
            }
        )

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

    if not token_manager.deduct_tokens(api_key, tokens_used):
        raise HTTPException(
            status_code=429,
            detail={
                "error": {
                    "message": "Failed to deduct tokens from quota.",
                    "type": "insufficient_quota",
                    "code": "insufficient_quota"
                }
            }
        )

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


@app.get("/v1/models")
async def list_models(authorization: str = Header(None)):
    """
    Liste les modèles disponibles (format OpenAI).

    Args:
        authorization: Header Authorization au format "Bearer <api_key>".

    Returns:
        Un dict au format OpenAI listant les modèles disponibles.

    Raises:
        HTTPException: 401 si la clé API est manquante ou invalide.
    """
    authenticate(authorization)
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


@app.post("/v1/api-keys")
async def create_api_key(email: str, quota_tokens: int = 10000):
    """
    Crée un nouvel utilisateur et retourne sa clé API.

    Args:
        email: Email de l'utilisateur.
        quota_tokens: Quota de tokens à allouer.

    Returns:
        Un dict contenant l'email, la clé API générée et le quota.
    """
    api_key = token_manager.create_user(email, quota_tokens)
    return {
        "email": email,
        "api_key": api_key,
        "quota_tokens": quota_tokens,
        "message": "Utilisateur créé. Gardez votre api_key secrète."
    }


@app.get("/v1/api-keys/status")
async def get_api_key_status(authorization: str = Header(None)):
    """
    Récupère le statut du compte associé à la clé API fournie.

    Args:
        authorization: Header Authorization au format "Bearer <api_key>".

    Returns:
        Le statut d'utilisation de l'utilisateur (quota, tokens restants, etc.).

    Raises:
        HTTPException: 401 si la clé API est manquante ou invalide.
    """
    user = authenticate(authorization)
    api_key = extract_bearer_token(authorization)
    return token_manager.get_status(api_key)


@app.get("/health")
async def health():
    """
    Vérifie l'état du service.

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
        "endpoints": {
            "POST /v1/completions": "Générer du texte (format OpenAI)",
            "GET /v1/models": "Lister les modèles disponibles",
            "POST /v1/api-keys": "Créer un nouvel utilisateur",
            "GET /v1/api-keys/status": "Vérifier votre quota",
            "GET /health": "Health check"
        },
        "docs": "http://localhost:8002/docs"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
