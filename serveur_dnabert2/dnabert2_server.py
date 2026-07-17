import os
import sys
from pathlib import Path
from datetime import datetime
import torch
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoTokenizer, AutoModel

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

MODEL_NAME = "dnabert2-117m"
MODEL_REPO = "zhihan1996/DNABERT-2-117M"

# Fenêtre de contexte du modèle (max_position_embeddings = 512, un BERT-base
# classique, comme GROVER). Au-delà, on préfère un 400 propre à un crash ou
# un troncage silencieux.
MAX_CONTEXT_TOKENS = 512

if torch.cuda.is_available():
    DEVICE = "cuda:0"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

# DNABERT-2 est un encodeur BERT (masked language model) entraîné sur 135
# génomes multi-espèces avec une tokenisation BPE, pas un modèle génératif.
# Comme Nucleotide Transformer et GROVER, on l'expose donc via /v1/embeddings
# plutôt que via /v1/completions. Contrairement à eux, son code de
# modélisation custom (ALiBi, pas d'AutoModelForMaskedLM enregistré) n'est
# exposé qu'au travers d'AutoModel : on récupère donc le dernier hidden
# state via outputs[0] plutôt que via output_hidden_states=True.
tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
model = AutoModel.from_pretrained(MODEL_REPO, trust_remote_code=True)
model.to(DEVICE)
model.eval()

# DNABERT-2's remote code picks a Triton flash-attention kernel whenever
# `triton` is importable — and recent `transformers` versions require triton
# to be installed just to load the model at all (their dynamic-module loader
# statically scans every import in the remote code, including inside the
# try/except that's supposed to make triton optional). That kernel only runs
# on CUDA; on CPU/MPS it crashes with `assert q.is_cuda`. So outside CUDA we
# force the module back onto its plain PyTorch attention fallback, which the
# remote code already implements for exactly this case.
if DEVICE != "cuda:0":
    sys.modules[type(model).__module__].flash_attn_qkvpacked_func = None


# ============ FASTAPI APP ============

app = FastAPI(
    title="DNABERT-2 API",
    description="API compatible OpenAI pour extraire des embeddings de séquences ADN avec DNABERT-2",
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
    le dernier hidden state (en ignorant le padding) pour obtenir un vecteur par séquence.

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
        encoded = tokenizer(
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
            outputs = model(input_ids, attention_mask=attention_mask)

        # DNABERT-2 (AutoModel, pas AutoModelForMaskedLM) renvoie le dernier
        # hidden state directement en position 0 de la sortie.
        last_hidden = outputs[0]
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
                "owned_by": "zhihan1996"
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
        "name": "DNABERT-2 API",
        "version": "0.2.0",
        "model": MODEL_NAME,
        "note": "Serveur interne : les utilisateurs passent par la gateway "
                "(port 8080), qui gère les clés API et les quotas.",
        "endpoints": {
            "POST /v1/embeddings": "Calculer les embeddings de séquences ADN (format OpenAI, secret interne requis)",
            "GET /v1/models": "Lister les modèles disponibles (secret interne requis)",
            "GET /health": "Health check"
        },
        "docs": "http://localhost:8004/docs"
    }

if __name__ == "__main__":
    import uvicorn
    # Défaut 127.0.0.1 : en bare-metal, ce serveur n'est pas censé être
    # joignable depuis l'extérieur, la gateway est la seule porte d'entrée
    # publique. En conteneur, docker compose passe 0.0.0.0 — indispensable
    # pour que la gateway le joigne depuis son propre namespace réseau — et
    # l'isolation vient de l'absence de publication du port vers l'hôte.
    uvicorn.run(app, host=os.getenv("BIOAI_BIND_HOST", "127.0.0.1"), port=8004)
