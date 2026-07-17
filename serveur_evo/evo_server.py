import os
import sys
import time
import uuid
from pathlib import Path
from datetime import datetime
import torch
from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Le package common/ vit à la racine du repo, un cran au-dessus de ce dossier.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import (  # noqa: E402
    CompletionChoice,
    CompletionResponse,
    CompletionUsage,
    context_length_exceeded,
    require_internal_key,
    server_error,
)

# Package officiel evo-model (pip install evo-model).
# On renomme la fonction generate() importée en evo_generate pour éviter
# toute collision avec la route FastAPI /v1/completions définie plus bas.
from evo import Evo, generate as evo_generate

# ============ CONFIGURATION ============

MODEL_NAME = "evo-1.5-8k-base"  # nom court attendu par la classe Evo(), pas le repo HF complet

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


# ============ MODELS FASTAPI (format OpenAI) ============
# Les schémas de réponse sont partagés (common/schemas.py) ; seule la requête
# est spécifique, ses bornes et ses paramètres dépendant du modèle.

class CompletionRequest(BaseModel):
    """Corps de requête pour l'endpoint /v1/completions, format OpenAI."""
    model: str = MODEL_NAME
    prompt: str
    max_tokens: int = Field(default=100, gt=0, le=MAX_TOKENS_LIMIT)  # équivalent OpenAI de l'ancien max_length, mappé sur n_tokens
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 4                # paramètre natif d'Evo, absent du schéma OpenAI standard
    n: int = 1                    # nombre de complétions à générer

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

@app.post(
    "/v1/completions",
    response_model=CompletionResponse,
    dependencies=[Depends(require_internal_key)],
)
async def create_completion(request: CompletionRequest):
    """
    Génère une séquence biologique à partir d'un prompt (format OpenAI).

    L'authentification de l'utilisateur final et son quota sont gérés par la
    gateway : cet endpoint n'accepte que le secret interne.

    Args:
        request: Paramètres de génération au format OpenAI
            (model, prompt, max_tokens, temperature, top_p, top_k, n).

    Returns:
        CompletionResponse au format OpenAI (id, choices, usage, etc.).

    Raises:
        HTTPException: 401 si le secret interne est absent ou incorrect,
            400 si prompt + max_tokens dépasse la fenêtre de contexte,
            500 en cas d'erreur mémoire GPU ou d'erreur interne.
    """
    input_length = len(request.prompt)
    if input_length + request.max_tokens > MAX_CONTEXT_TOKENS:
        raise context_length_exceeded(
            f"prompt ({input_length} tokens) + max_tokens ({request.max_tokens}) "
            f"exceeds the model's context window ({MAX_CONTEXT_TOKENS} tokens)."
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
        raise server_error(
            "Out of GPU memory. Reduce max_tokens or batch size.",
            code="out_of_memory",
        )
    except Exception as e:
        raise server_error(str(e))


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
                "owned_by": "evo"
            }
        ]
    }


@app.get("/health")
async def health():
    """
    Vérifie l'état du service.

    Ouvert sans authentification : start_all.py et la gateway s'en servent
    pour attendre que le modèle ait fini de charger.

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
        "note": "Serveur interne : les utilisateurs passent par la gateway "
                "(port 8080), qui gère les clés API et les quotas.",
        "endpoints": {
            "POST /v1/completions": "Générer une séquence (format OpenAI, secret interne requis)",
            "GET /v1/models": "Lister les modèles disponibles (secret interne requis)",
            "GET /health": "Health check"
        },
        "docs": "http://localhost:8000/docs"
    }

if __name__ == "__main__":
    import uvicorn
    # Défaut 127.0.0.1 : en bare-metal, ce serveur n'est pas censé être
    # joignable depuis l'extérieur, la gateway est la seule porte d'entrée
    # publique. En conteneur, docker compose passe 0.0.0.0 — indispensable
    # pour que la gateway le joigne depuis son propre namespace réseau — et
    # l'isolation vient de l'absence de publication du port vers l'hôte.
    uvicorn.run(app, host=os.getenv("BIOAI_BIND_HOST", "127.0.0.1"), port=8000)
