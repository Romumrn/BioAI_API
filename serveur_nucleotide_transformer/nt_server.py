import secrets
import json
import time
import uuid
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Union
import torch
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForMaskedLM

# ============ CONFIGURATION ============

MODEL_NAME = "nucleotide-transformer-v2-100m-multi-species"
MODEL_REPO = "InstaDeepAI/nucleotide-transformer-v2-100m-multi-species"
TOKENS_FILE = "tokens_db.json"

# Fenêtre de contexte du modèle (max_position_embeddings = 2050). Au-delà,
# on préfère un 400 propre à un crash ou un troncage silencieux.
MAX_CONTEXT_TOKENS = 2048

if torch.cuda.is_available():
    DEVICE = "cuda:0"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

# Nucleotide Transformer est un encodeur (masked language model, type ESM),
# pas un modèle génératif comme Evo. On l'expose donc via /v1/embeddings
# (comme le fait l'API OpenAI pour ses modèles d'embeddings) plutôt que
# via /v1/completions, qui n'aurait pas de sens pour ce type d'architecture.
tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO, trust_remote_code=True)
model = AutoModelForMaskedLM.from_pretrained(MODEL_REPO, trust_remote_code=True)
model.to(DEVICE)
model.eval()


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


# ============ MODELS FASTAPI (format OpenAI) ============

class EmbeddingsRequest(BaseModel):
    """Corps de requête pour l'endpoint /v1/embeddings, format OpenAI."""
    model: str = MODEL_NAME
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

# ============ FASTAPI APP ============

app = FastAPI(
    title="Nucleotide Transformer API",
    description="API compatible OpenAI pour extraire des embeddings de séquences ADN avec Nucleotide Transformer",
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

@app.post("/v1/embeddings", response_model=EmbeddingsResponse)
def create_embeddings(
    request: EmbeddingsRequest,
    authorization: str = Header(None)
):
    """
    Calcule les embeddings d'une ou plusieurs séquences ADN (format OpenAI).

    Authentifie la requête via le header Authorization (Bearer <api_key>),
    vérifie le quota de tokens restant, tokenise les séquences, fait passer
    le lot dans le modèle et moyenne les hidden states du dernier layer
    (en ignorant le padding) pour obtenir un vecteur par séquence.

    Args:
        request: Séquence(s) ADN à encoder (un str ou une liste de str).
        authorization: Header Authorization au format "Bearer <api_key>".

    Returns:
        EmbeddingsResponse au format OpenAI (un vecteur par séquence, usage).

    Raises:
        HTTPException: 401 si la clé API est manquante ou invalide,
            429 si le quota est dépassé ou insuffisant,
            500 en cas d'erreur interne.
    """
    user = authenticate(authorization)
    api_key = extract_bearer_token(authorization)

    sequences = [request.input] if isinstance(request.input, str) else request.input

    remaining = user["quota_tokens"] - user["used_tokens"]
    if remaining < 10:
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

    try:
        encoded = tokenizer.batch_encode_plus(
            sequences,
            return_tensors="pt",
            padding=True,
        )

        seq_length = encoded["input_ids"].shape[1]
        if seq_length > MAX_CONTEXT_TOKENS:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": {
                        "message": f"Input sequence tokenizes to {seq_length} tokens, "
                                    f"which exceeds the model's context window ({MAX_CONTEXT_TOKENS} tokens).",
                        "type": "invalid_request_error",
                        "code": "context_length_exceeded"
                    }
                }
            )

        input_ids = encoded["input_ids"].to(DEVICE)
        attention_mask = encoded["attention_mask"].to(DEVICE)
        prompt_tokens = int(attention_mask.sum().item())

        with torch.no_grad():
            outputs = model(
                input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )

        # Moyenne des hidden states du dernier layer sur les tokens
        # non-paddés, pour obtenir un vecteur de taille fixe par séquence.
        last_hidden = outputs.hidden_states[-1]
        mask = attention_mask.unsqueeze(-1)
        pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1)

        data = [
            EmbeddingData(embedding=vec.tolist(), index=i)
            for i, vec in enumerate(pooled)
        ]

        if not token_manager.deduct_tokens(api_key, prompt_tokens):
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

        return EmbeddingsResponse(
            data=data,
            model=request.model,
            usage=EmbeddingsUsage(
                prompt_tokens=prompt_tokens,
                total_tokens=prompt_tokens
            )
        )

    except HTTPException:
        raise
    except torch.cuda.OutOfMemoryError:
        raise HTTPException(
            status_code=500,
            detail={
                "error": {
                    "message": "Out of GPU memory. Reduce batch size.",
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
                "owned_by": "instadeep"
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
        "name": "Nucleotide Transformer API",
        "version": "0.1.0",
        "model": MODEL_NAME,
        "endpoints": {
            "POST /v1/embeddings": "Calculer les embeddings de séquences ADN (format OpenAI)",
            "GET /v1/models": "Lister les modèles disponibles",
            "POST /v1/api-keys": "Créer un nouvel utilisateur",
            "GET /v1/api-keys/status": "Vérifier votre quota",
            "GET /health": "Health check"
        },
        "docs": "http://localhost:8001/docs"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
