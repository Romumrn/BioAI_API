import os
from datetime import datetime
from json import JSONDecodeError
from typing import Optional, Dict
import requests
from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from common import (
    TokenManager,
    api_error,
    authenticate,
    get_internal_key,
    insufficient_quota,
    server_error,
)

# ============ CONFIGURATION ============
#
# Registre des modèles disponibles : nom du modèle -> serveur qui le sert.
# Pour ajouter un modèle, il suffit de démarrer son serveur (voir start_all.py
# ou docker-compose.yml) et de rajouter une entrée ici avec son nom de modèle,
# son port et son type (completions ou embeddings, selon l'endpoint que le
# serveur de modèle expose).


def backend_url(name: str, port: int) -> str:
    """
    URL d'un serveur de modèle, surchargeable par variable d'environnement.

    Le défaut vise localhost : c'est le cas bare-metal (start_all.py), où
    tout tourne sur la même machine. Sous docker compose, chaque serveur est
    un conteneur distinct joignable par son nom de service, et le compose
    injecte BIOAI_EVO_URL=http://evo:8000 & co.

    Args:
        name: Nom court du service (ex: "evo"), qui donne BIOAI_EVO_URL.
        port: Port du serveur en bare-metal.

    Returns:
        L'URL de base du serveur de modèle.
    """
    return os.getenv(f"BIOAI_{name.upper()}_URL", f"http://localhost:{port}")


BACKENDS = {
    "evo-1.5-8k-base": {
        "url": backend_url("evo", 8000),
        "kind": "completions",
        "owned_by": "evo",
    },
    "nucleotide-transformer-v2-100m-multi-species": {
        "url": backend_url("nt", 8001),
        "kind": "embeddings",
        "owned_by": "instadeep",
    },
    "biomistral-7b": {
        "url": backend_url("biomistral", 8002),
        "kind": "completions",
        "owned_by": "biomistral",
    },
    "grover": {
        "url": backend_url("grover", 8003),
        "kind": "embeddings",
        "owned_by": "poetschlab",
    },
    "dnabert2-117m": {
        "url": backend_url("dnabert2", 8004),
        "kind": "embeddings",
        "owned_by": "zhihan1996",
    },
}

# Clés API des utilisateurs finaux. Le défaut vise le fichier à la racine du
# repo (bare-metal) ; sous docker compose, la variable pointe vers un volume
# nommé, sans quoi les utilisateurs disparaîtraient à chaque rebuild d'image.
TOKENS_FILE = os.getenv("BIOAI_TOKENS_FILE", "tokens_db.json")

# La gateway est le seul endroit du projet où vivent les utilisateurs et leur
# quota. Les serveurs de modèles ne connaissent que le secret interne
# (common/internal.py) et ne comptent rien.
token_manager = TokenManager(TOKENS_FILE)

# Quota minimum restant exigé avant d'accepter une requête, par type
# d'endpoint : le coût réel n'est connu qu'après coup, donc on refuse en
# amont les comptes manifestement à sec.
MIN_REMAINING = {"completions": 50, "embeddings": 10}


def resolve_backend(model: Optional[str], expected_kind: str) -> Dict:
    """
    Trouve le sous-serveur associé à un nom de modèle.

    Args:
        model: Nom du modèle demandé (champ "model" de la requête).
        expected_kind: Type d'endpoint attendu ("completions" ou "embeddings").

    Returns:
        L'entrée du registre BACKENDS correspondante.

    Raises:
        HTTPException: 400 si le modèle est inconnu ou ne correspond pas
            au type d'endpoint appelé.
    """
    backend = BACKENDS.get(model)
    if not backend or backend["kind"] != expected_kind:
        available = [name for name, b in BACKENDS.items() if b["kind"] == expected_kind]
        raise api_error(
            400,
            f"Unknown or incompatible model '{model}' for this endpoint. "
            f"Available: {available}",
            "invalid_request_error",
            "model_not_found",
        )
    return backend


def forward(backend: Dict, path: str, payload: dict) -> dict:
    """
    Relaie une requête vers un serveur de modèle, avec le secret interne,
    et renvoie le corps de la réponse en cas de succès.

    Args:
        backend: Entrée du registre BACKENDS ciblée.
        path: Chemin de l'endpoint sur le serveur de modèle (ex: "/v1/completions").
        payload: Corps JSON à transmettre tel quel.

    Returns:
        Le corps JSON de la réponse du serveur de modèle.

    Raises:
        HTTPException: le code renvoyé par le serveur de modèle si celui-ci
            échoue, ou 500 s'il est injoignable.
    """
    try:
        resp = requests.post(
            f"{backend['url']}{path}",
            json=payload,
            headers={"Authorization": f"Bearer {get_internal_key()}"},
            timeout=300,
        )
    except requests.RequestException as e:
        raise server_error(
            f"Backend unreachable ({backend['url']}): {e}",
            code="backend_unavailable",
        )

    if not resp.ok:
        # Le corps d'erreur du serveur de modèle est déjà {"detail": ...} :
        # le relayer tel quel le remettrait dans un second "detail", et
        # l'appelant recevrait {"detail": {"detail": {"error": ...}}}. On
        # déballe donc d'un niveau pour que ces erreurs aient exactement la
        # même forme que celles émises par la gateway elle-même.
        body = resp.json()
        detail = body.get("detail", body) if isinstance(body, dict) else body
        raise HTTPException(status_code=resp.status_code, detail=detail)

    return resp.json()


async def parse_payload(request: Request) -> dict:
    """
    Lit le corps JSON d'une requête et garantit que c'est bien un objet.

    La gateway relaie le corps tel quel au serveur de modèle : elle n'a donc
    pas de schéma Pydantic pour ces endpoints, et rien ne rattraperait un
    corps mal formé — l'appelant recevrait un 500 opaque là où le problème
    est chez lui.

    Args:
        request: Requête brute.

    Returns:
        Le corps JSON désérialisé.

    Raises:
        HTTPException: 400 si le corps n'est pas du JSON valide, ou si c'est
            du JSON valide mais pas un objet (une liste, un nombre...).
    """
    try:
        payload = await request.json()
    except JSONDecodeError as e:
        raise api_error(
            400,
            f"Invalid JSON in request body: {e}",
            "invalid_request_error",
            "invalid_json",
        )

    if not isinstance(payload, dict):
        raise api_error(
            400,
            f"Request body must be a JSON object, got {type(payload).__name__}.",
            "invalid_request_error",
            "invalid_request_body",
        )

    return payload


def handle_request(
    authorization: Optional[str],
    payload: dict,
    kind: str,
) -> dict:
    """
    Chaîne complète d'une requête modèle : authentification de l'utilisateur,
    résolution du backend, contrôle de quota, relais, puis décompte.

    C'est ici, et seulement ici, que le quota d'un utilisateur est vérifié et
    débité : les serveurs de modèles ne comptent rien.

    Args:
        authorization: Header Authorization au format "Bearer <api_key>".
        payload: Corps JSON de la requête, relayé tel quel.
        kind: Type d'endpoint ("completions" ou "embeddings").

    Returns:
        Le corps JSON de la réponse du serveur de modèle.

    Raises:
        HTTPException: 401 si la clé API est invalide, 400 si le modèle est
            inconnu, 429 si le quota est dépassé, 500 en cas d'erreur backend.
    """
    api_key, user = authenticate(token_manager, authorization)
    backend = resolve_backend(payload.get("model"), expected_kind=kind)

    remaining = user["quota_tokens"] - user["used_tokens"]
    if remaining < MIN_REMAINING[kind]:
        raise insufficient_quota(f"Quota exceeded. Remaining tokens: {remaining}")

    data = forward(backend, f"/v1/{kind}", payload)

    tokens_used = data.get("usage", {}).get("total_tokens", 0)
    if not token_manager.deduct_tokens(api_key, tokens_used):
        raise insufficient_quota("Failed to deduct tokens from quota.")

    return data


# ============ FASTAPI APP ============

app = FastAPI(
    title="BioAI Gateway",
    description="Point d'entrée unique : authentification et routage vers les "
                "serveurs de modèles bio-informatique (Evo, Nucleotide Transformer, "
                "BioMistral...)",
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

@app.post("/v1/completions")
async def create_completion(request: Request, authorization: str = Header(None)):
    """
    Génère du texte/une séquence (format OpenAI), routé vers le bon modèle.

    Args:
        request: Requête brute (le corps est relayé tel quel au serveur de modèle).
        authorization: Header Authorization au format "Bearer <api_key>".

    Returns:
        La réponse du serveur de modèle, au format OpenAI.

    Raises:
        HTTPException: 400 si le corps n'est pas un objet JSON valide ou si le
            modèle est inconnu, 401 si la clé API est invalide, 429 si le
            quota est dépassé, 500 en cas d'erreur backend.
    """
    payload = await parse_payload(request)
    data = handle_request(authorization, payload, kind="completions")
    return JSONResponse(content=data)


@app.post("/v1/embeddings")
async def create_embeddings(request: Request, authorization: str = Header(None)):
    """
    Calcule des embeddings (format OpenAI), routé vers le bon modèle.

    Args:
        request: Requête brute (le corps est relayé tel quel au serveur de modèle).
        authorization: Header Authorization au format "Bearer <api_key>".

    Returns:
        La réponse du serveur de modèle, au format OpenAI.

    Raises:
        HTTPException: 400 si le corps n'est pas un objet JSON valide ou si le
            modèle est inconnu, 401 si la clé API est invalide, 429 si le
            quota est dépassé, 500 en cas d'erreur backend.
    """
    payload = await parse_payload(request)
    data = handle_request(authorization, payload, kind="embeddings")
    return JSONResponse(content=data)


@app.get("/v1/models")
async def list_models(authorization: str = Header(None)):
    """
    Liste tous les modèles disponibles, tous serveurs de modèles confondus.

    Args:
        authorization: Header Authorization au format "Bearer <api_key>".

    Returns:
        Un dict au format OpenAI listant les modèles disponibles.

    Raises:
        HTTPException: 401 si la clé API est manquante ou invalide.
    """
    authenticate(token_manager, authorization)
    return {
        "object": "list",
        "data": [
            {
                "id": name,
                "object": "model",
                "created": 0,
                "owned_by": backend["owned_by"]
            }
            for name, backend in BACKENDS.items()
        ]
    }


@app.post("/v1/api-keys")
async def create_api_key(email: str, quota_tokens: int = 10000):
    """
    Crée un nouvel utilisateur et retourne sa clé API pour la gateway.

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
    api_key, _ = authenticate(token_manager, authorization)
    return token_manager.get_status(api_key)


@app.get("/health")
async def health():
    """
    Vérifie l'état de la gateway et de chaque serveur de modèle enregistré.

    Returns:
        Un dict avec le statut global et le détail par serveur de modèle.
    """
    backend_statuses = {}
    for name, backend in BACKENDS.items():
        try:
            r = requests.get(f"{backend['url']}/health", timeout=2)
            backend_statuses[name] = r.json() if r.ok else {"status": "error"}
        except requests.RequestException:
            backend_statuses[name] = {"status": "unreachable"}

    return {
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "backends": backend_statuses
    }


@app.get("/")
async def root():
    """
    Retourne les informations générales de la gateway.

    Returns:
        Un dict décrivant l'API et listant ses endpoints disponibles.
    """
    return {
        "name": "BioAI Gateway",
        "version": "0.1.0",
        "models": list(BACKENDS.keys()),
        "endpoints": {
            "POST /v1/completions": "Générer du texte/une séquence (format OpenAI)",
            "POST /v1/embeddings": "Calculer des embeddings (format OpenAI)",
            "GET /v1/models": "Lister tous les modèles disponibles",
            "POST /v1/api-keys": "Créer un nouvel utilisateur",
            "GET /v1/api-keys/status": "Vérifier votre quota",
            "GET /health": "Health check (gateway + sous-serveurs)"
        },
        "docs": "http://localhost:8080/docs"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
