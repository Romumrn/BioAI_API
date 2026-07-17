import os
import sys
from pathlib import Path
from datetime import datetime
import torch
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoTokenizer, AutoModelForMaskedLM

# Le package common/ vit à la racine du repo, un cran au-dessus de ce
# dossier, et ce serveur tourne avec son propre venv : on l'ajoute au
# sys.path plutôt que de dépendre d'une installation ou d'un PYTHONPATH.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import (  # noqa: E402
    EmbeddingData,
    EmbeddingsRequest,
    EmbeddingsResponse,
    EmbeddingsUsage,
    context_length_exceeded,
    require_internal_key,
    server_error,
)

# ============ CONFIGURATION ============

MODEL_NAME = "nucleotide-transformer-v2-100m-multi-species"
MODEL_REPO = "InstaDeepAI/nucleotide-transformer-v2-100m-multi-species"

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

@app.post(
    "/v1/embeddings",
    response_model=EmbeddingsResponse,
    dependencies=[Depends(require_internal_key)],
)
def create_embeddings(request: EmbeddingsRequest):
    """
    Calcule les embeddings d'une ou plusieurs séquences ADN (format OpenAI).

    Tokenise les séquences, fait passer le lot dans le modèle et moyenne
    les hidden states du dernier layer (en ignorant le padding) pour obtenir un vecteur par séquence.

    L'authentification de l'utilisateur final et son quota sont gérés par la
    gateway : cet endpoint n'accepte que le secret interne.

    Args:
        request: Séquence(s) ADN à encoder (un str ou une liste de str).

    Returns:
        EmbeddingsResponse au format OpenAI (un vecteur par séquence, usage).

    Raises:
        HTTPException: 401 si le secret interne est absent ou incorrect,
            400 si la séquence dépasse la fenêtre de contexte,
            500 en cas d'erreur interne.
    """
    sequences = [request.input] if isinstance(request.input, str) else request.input

    try:
        encoded = tokenizer.batch_encode_plus(
            sequences,
            return_tensors="pt",
            padding=True,
        )

        seq_length = encoded["input_ids"].shape[1]
        if seq_length > MAX_CONTEXT_TOKENS:
            raise context_length_exceeded(
                f"Input sequence tokenizes to {seq_length} tokens, which exceeds "
                f"the model's context window ({MAX_CONTEXT_TOKENS} tokens)."
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
        raise server_error("Out of GPU memory. Reduce batch size.", code="out_of_memory")
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
                "owned_by": "instadeep"
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
        "name": "Nucleotide Transformer API",
        "version": "0.2.0",
        "model": MODEL_NAME,
        "note": "Serveur interne : les utilisateurs passent par la gateway "
                "(port 8080), qui gère les clés API et les quotas.",
        "endpoints": {
            "POST /v1/embeddings": "Calculer les embeddings de séquences ADN (format OpenAI, secret interne requis)",
            "GET /v1/models": "Lister les modèles disponibles (secret interne requis)",
            "GET /health": "Health check"
        },
        "docs": "http://localhost:8001/docs"
    }

if __name__ == "__main__":
    import uvicorn
    # Défaut 127.0.0.1 : en bare-metal, ce serveur n'est pas censé être
    # joignable depuis l'extérieur, la gateway est la seule porte d'entrée
    # publique. En conteneur, docker compose passe 0.0.0.0 — indispensable
    # pour que la gateway le joigne depuis son propre namespace réseau — et
    # l'isolation vient de l'absence de publication du port vers l'hôte.
    uvicorn.run(app, host=os.getenv("BIOAI_BIND_HOST", "127.0.0.1"), port=8001)
