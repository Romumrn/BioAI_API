import secrets
import json
import time
import uuid
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict
import torch
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Package officiel evo-model (pip install evo-model).
# On renomme la fonction generate() importée en evo_generate pour éviter
# toute collision avec la route FastAPI /v1/completions définie plus bas.
from evo import Evo, generate as evo_generate

# ============ CONFIGURATION ============

MODEL_NAME = "evo-1.5-8k-base"  # nom court attendu par la classe Evo(), pas le repo HF complet
TOKENS_FILE = "tokens_db.json"

# Fenêtre de contexte du modèle (8k, comme son nom l'indique). Au-delà,
# StripedHyena part en "illegal memory access" côté CUDA plutôt que de
# renvoyer une erreur propre — donc on borne nous-mêmes en amont.
MAX_CONTEXT_TOKENS = 8192
MAX_TOKENS_LIMIT = 4096

if torch.cuda.is_available():
    DEVICE = "cuda:0"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

# Chargement via le package officiel evo-model, qui gère lui-même
# StripedHyenaConfig et son propre système de cache (compatible avec
# modeling_hyena.py) — on évite ainsi AutoConfig/AutoModelForCausalLM
# et model.generate() de transformers, qui ne sont pas le chemin
# supporté par les auteurs d'Evo pour la génération multi-tokens.
evo_model = Evo(MODEL_NAME)
model, tokenizer = evo_model.model, evo_model.tokenizer
model.to(DEVICE)
model.eval()


# ============ GESTION DES TOKENS ============

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


# ============ MODELS FASTAPI (format OpenAI) ============

class CompletionRequest(BaseModel):
    """Corps de requête pour l'endpoint /v1/completions, format OpenAI."""
    model: str = MODEL_NAME
    prompt: str
    max_tokens: int = Field(default=100, gt=0, le=MAX_TOKENS_LIMIT)  # équivalent OpenAI de l'ancien max_length, mappé sur n_tokens
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 4                # paramètre natif d'Evo, absent du schéma OpenAI standard
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
    title="EVO-1.5 API",
    description="API compatible OpenAI pour générer des séquences biologiques avec EVO-1.5",
    version="0.2.0"
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
async def create_completion(
    request: CompletionRequest,
    authorization: str = Header(None)
):
    """
    Génère une séquence biologique à partir d'un prompt (format OpenAI).

    Authentifie la requête via le header Authorization (Bearer <api_key>),
    vérifie le quota de tokens restant, appelle le modèle Evo pour générer
    la séquence, puis déduit les tokens consommés.

    Args:
        request: Paramètres de génération au format OpenAI
            (model, prompt, max_tokens, temperature, top_p, top_k, n).
        authorization: Header Authorization au format "Bearer <api_key>".

    Returns:
        CompletionResponse au format OpenAI (id, choices, usage, etc.).

    Raises:
        HTTPException: 401 si la clé API est manquante ou invalide,
            429 si le quota est dépassé ou insuffisant,
            500 en cas d'erreur mémoire GPU ou d'erreur interne.
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

    input_length = len(request.prompt)
    if input_length + request.max_tokens > MAX_CONTEXT_TOKENS:
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "message": f"prompt ({input_length} tokens) + max_tokens ({request.max_tokens}) "
                                f"exceeds the model's context window ({MAX_CONTEXT_TOKENS} tokens).",
                    "type": "invalid_request_error",
                    "code": "context_length_exceeded"
                }
            }
        )

    try:
        # generate() du package evo prend une LISTE de prompts (un par
        # échantillon souhaité) et retourne (sequences, scores) déjà décodés.
        with torch.no_grad():
            output_seqs, output_scores = evo_generate(
                [request.prompt] * request.n,
                model,
                tokenizer,
                n_tokens=request.max_tokens,
                temperature=request.temperature,
                top_k=request.top_k,
                top_p=request.top_p,
                cached_generation=True,
                batched=True,
                prepend_bos=False,
                device=DEVICE,
                verbose=0,
            )

        # output_seqs contient prompt + suite générée pour chaque échantillon:
        # on isole la partie nouvellement générée pour chaque choice retourné.
        choices = []
        total_completion_tokens = 0
        for i, full_seq in enumerate(output_seqs):
            completion_text = full_seq[input_length:]
            total_completion_tokens += len(completion_text)
            choices.append(
                CompletionChoice(
                    text=completion_text,
                    index=i,
                    finish_reason="length"
                )
            )

        tokens_used = input_length + total_completion_tokens

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
                prompt_tokens=input_length,
                completion_tokens=total_completion_tokens,
                total_tokens=tokens_used
            )
        )

    except torch.cuda.OutOfMemoryError:
        raise HTTPException(
            status_code=500,
            detail={
                "error": {
                    "message": "Out of GPU memory. Reduce max_tokens or batch size.",
                    "type": "server_error",
                    "code": "out_of_memory"
                }
            }
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": {
                    "message": str(e),
                    "type": "server_error",
                    "code": "internal_error"
                }
            }
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
                "owned_by": "evo"
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
        Un dict avec le statut, le nom du modèle, le device utilisé
        et un timestamp.
    """
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "device": DEVICE,
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
        "name": "EVO-1.5 API",
        "version": "0.2.0",
        "model": MODEL_NAME,
        "endpoints": {
            "POST /v1/completions": "Générer une séquence (format OpenAI)",
            "GET /v1/models": "Lister les modèles disponibles",
            "POST /v1/api-keys": "Créer un nouvel utilisateur",
            "GET /v1/api-keys/status": "Vérifier votre quota",
            "GET /health": "Health check"
        },
        "docs": "http://localhost:8000/docs"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
