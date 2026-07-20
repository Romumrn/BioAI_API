import json
import os
import time
import uuid
from datetime import datetime
from json import JSONDecodeError
from typing import List, Optional, Dict
import requests
from fastapi import Depends, FastAPI, HTTPException, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from common import (
    TokenManager,
    api_error,
    authenticate,
    get_internal_key,
    insufficient_quota,
    remaining_tokens,
    require_admin_key,
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
    "med42-8b": {
        "url": backend_url("med42", 8002),
        "kind": "completions",
        "owned_by": "m42-health",
        # Contrairement à Evo (séquences ADN brutes, pas de notion de chat),
        # ce serveur expose un vrai /v1/chat/completions qui relaie
        # l'historique structuré à Ollama. /v1/chat/completions de la gateway
        # le lui transmet donc tel quel plutôt que de l'aplatir en prompt —
        # l'aplatissement casse la tenue de conversation du modèle (voir
        # serveur_med42/med42_server.py:call_ollama_chat).
        "chat_capable": True,
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
    path: Optional[str] = None,
) -> dict:
    """
    Chaîne complète d'une requête modèle : authentification de l'utilisateur,
    résolution du backend, contrôle de quota, relais, puis décompte.

    C'est ici, et seulement ici, que le quota d'un utilisateur est vérifié et
    débité : les serveurs de modèles ne comptent rien.

    Args:
        authorization: Header Authorization au format "Bearer <api_key>".
        payload: Corps JSON de la requête, relayé tel quel.
        kind: Type d'endpoint ("completions" ou "embeddings"), utilisé pour
            résoudre le backend et le seuil de quota minimum.
        path: Chemin à appeler sur le serveur de modèle, si différent de
            f"/v1/{kind}" — cas de /v1/chat/completions sur un backend
            chat_capable (voir create_chat_completion).

    Returns:
        Le corps JSON de la réponse du serveur de modèle.

    Raises:
        HTTPException: 401 si la clé API est invalide, 400 si le modèle est
            inconnu, 429 si le quota est dépassé, 500 en cas d'erreur backend.
    """
    api_key, user = authenticate(token_manager, authorization)
    backend = resolve_backend(payload.get("model"), expected_kind=kind)

    remaining = remaining_tokens(user)
    if remaining is not None and remaining < MIN_REMAINING[kind]:
        raise insufficient_quota(f"Quota exceeded. Remaining tokens: {remaining}")

    data = forward(backend, path or f"/v1/{kind}", payload)

    tokens_used = data.get("usage", {}).get("total_tokens", 0)
    if not token_manager.deduct_tokens(api_key, tokens_used):
        raise insufficient_quota("Failed to deduct tokens from quota.")

    return data


# ============ TRADUCTION CHAT <-> COMPLETIONS ============
#
# OpenGateLLM ne sait appeler qu'un /v1/chat/completions pour tout modèle de
# type text-generation : son client provider n'a aucune entrée pour les
# completions legacy. Nos serveurs de modèles qui n'ont pas de notion de chat
# (Evo : séquences ADN brutes) ne parlent que le format "completions"
# historique — cette traduction est ce qui les rend branchables malgré tout.
# Med42, lui, expose un vrai /v1/chat/completions (voir "chat_capable" dans
# BACKENDS) et n'a pas besoin de cette traduction : l'aplatissement en prompt
# casserait sa tenue de conversation à plusieurs tours.

ROLE_LABELS = {"system": "System", "user": "User", "assistant": "Assistant"}


def messages_to_prompt(messages: List[Dict]) -> str:
    """
    Aplatit une liste de messages de chat en un prompt unique.

    Cette fonction n'est plus appelée que pour Evo (le seul backend qui ne
    soit pas "chat_capable", voir BACKENDS) : Med42 reçoit désormais son
    historique de messages tel quel, non aplati. Le cas d'un seul message
    utilisateur est traité à part, et ce n'est pas une optimisation : evo est
    un modèle d'ADN, à qui l'on envoie une séquence brute. Le préfixer d'un
    "User:" et le suivre d'un "Assistant:" lui ferait continuer une
    conversation en anglais au lieu de la séquence — le modèle répondrait à
    côté au lieu d'échouer franchement.

    Args:
        messages: Liste de messages au format OpenAI ({"role", "content"}).

    Returns:
        Le prompt à envoyer au serveur de modèle.

    Raises:
        HTTPException: 400 si la liste est vide, mal formée, ou si un contenu
            n'est pas du texte (les contenus multimodaux ne sont pas gérés :
            aucun de nos modèles n'est multimodal).
    """
    if not isinstance(messages, list) or not messages:
        raise api_error(
            400,
            "Field 'messages' must be a non-empty list of chat messages.",
            "invalid_request_error",
            "invalid_request_body",
        )

    for message in messages:
        if not isinstance(message, dict) or "content" not in message:
            raise api_error(
                400,
                "Each message must be an object with 'role' and 'content' fields.",
                "invalid_request_error",
                "invalid_request_body",
            )
        if not isinstance(message["content"], str):
            raise api_error(
                400,
                "Message content must be a string. This gateway serves no "
                "multimodal model, so content parts are not supported.",
                "invalid_request_error",
                "invalid_request_body",
            )

    if len(messages) == 1 and messages[0].get("role") == "user":
        return messages[0]["content"]

    transcript = "\n".join(
        f"{ROLE_LABELS.get(m.get('role'), 'User')}: {m['content']}" for m in messages
    )
    return f"{transcript}\n{ROLE_LABELS['assistant']}:"


def chat_to_completion_payload(payload: dict) -> dict:
    """
    Convertit un corps /v1/chat/completions en corps /v1/completions.

    Args:
        payload: Corps de requête au format chat.

    Returns:
        Le corps à relayer au serveur de modèle.

    Raises:
        HTTPException: 400 si les messages sont absents ou mal formés.
    """
    completion_payload = {
        key: value
        for key, value in payload.items()
        # "messages" est remplacé par "prompt" ; "stream" est traité par la
        # gateway et n'a pas de sens pour un serveur qui ne streame pas.
        if key not in ("messages", "stream", "stream_options")
        # Les clients OpenAI envoient volontiers max_tokens: null pour "pas de
        # limite". Nos serveurs typent max_tokens en int borné : un null les
        # ferait répondre 422 là où leur propre défaut convient.
        and value is not None
    }
    completion_payload["prompt"] = messages_to_prompt(payload.get("messages"))
    return completion_payload


def completion_to_chat(data: dict) -> dict:
    """
    Convertit une réponse /v1/completions en réponse /v1/chat/completions.

    Args:
        data: Réponse du serveur de modèle, au format completions.

    Returns:
        La même réponse au format chat.
    """
    return {
        "id": data.get("id", f"chatcmpl-{uuid.uuid4().hex}"),
        "object": "chat.completion",
        "created": data.get("created", int(time.time())),
        "model": data.get("model"),
        "choices": [
            {
                "index": choice.get("index", index),
                "message": {"role": "assistant", "content": choice.get("text", "")},
                "finish_reason": choice.get("finish_reason", "stop"),
            }
            for index, choice in enumerate(data.get("choices", []))
        ],
        "usage": data.get("usage", {}),
    }


def chat_to_sse(chat: dict):
    """
    Rejoue une réponse chat complète sous forme d'événements SSE.

    Aucun de nos serveurs de modèles ne streame : la génération est déjà
    terminée quand cette fonction est appelée. C'est donc du faux streaming —
    tout le texte arrive en un seul chunk, après le temps de génération
    complet. Le but n'est pas la latence mais la compatibilité : le playground
    d'OpenGateLLM demande stream=true par défaut, et refuser le champ y
    rendrait les modèles inutilisables.

    Args:
        chat: Réponse complète au format chat.completion.

    Yields:
        Les lignes SSE, terminées par le sentinel [DONE] attendu des clients
        OpenAI.
    """
    base = {
        "id": chat["id"],
        "object": "chat.completion.chunk",
        "created": chat["created"],
        "model": chat["model"],
    }

    for choice in chat["choices"]:
        content = choice["message"]["content"]
        delta = {
            **base,
            "choices": [
                {
                    "index": choice["index"],
                    "delta": {"role": "assistant", "content": content},
                    "finish_reason": None,
                }
            ],
        }
        yield f"data: {json.dumps(delta)}\n\n"

        stop = {
            **base,
            "choices": [
                {
                    "index": choice["index"],
                    "delta": {},
                    "finish_reason": choice["finish_reason"],
                }
            ],
        }
        yield f"data: {json.dumps(stop)}\n\n"

    yield "data: [DONE]\n\n"


# ============ FASTAPI APP ============

app = FastAPI(
    title="BioAI Gateway",
    description="Point d'entrée unique : authentification et routage vers les "
                "serveurs de modèles bio-informatique (Evo, Nucleotide Transformer, "
                "Med42...)",
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


@app.post("/v1/chat/completions")
async def create_chat_completion(request: Request, authorization: str = Header(None)):
    """
    Génère une réponse de chat (format OpenAI), routée vers le bon modèle.

    C'est le seul format de génération qu'OpenGateLLM sache appeler. Deux
    chemins selon le backend visé :
      - "chat_capable" (Med42) : les messages sont relayés tels quels,
        le serveur de modèle tient lui-même la conversation.
      - sinon (Evo) : traduit vers le format completions que parle le
        serveur, et retraduit la réponse — voir chat_to_completion_payload.

    Args:
        request: Requête brute au format chat (messages, max_tokens, stream...).
        authorization: Header Authorization au format "Bearer <api_key>".

    Returns:
        La réponse au format chat.completion, ou un flux SSE de
        chat.completion.chunk si stream=true.

    Raises:
        HTTPException: 400 si le corps ou les messages sont mal formés ou si le
            modèle est inconnu, 401 si la clé API est invalide, 429 si le
            quota est dépassé, 500 en cas d'erreur backend.
    """
    payload = await parse_payload(request)
    stream = bool(payload.get("stream", False))

    backend = resolve_backend(payload.get("model"), expected_kind="completions")

    if backend.get("chat_capable"):
        forward_payload = {
            key: value
            for key, value in payload.items()
            if key not in ("stream", "stream_options") and value is not None
        }
        chat = handle_request(
            authorization, forward_payload, kind="completions", path="/v1/chat/completions"
        )
    else:
        data = handle_request(
            authorization, chat_to_completion_payload(payload), kind="completions"
        )
        chat = completion_to_chat(data)

    if stream:
        return StreamingResponse(chat_to_sse(chat), media_type="text/event-stream")
    return JSONResponse(content=chat)


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


@app.post("/v1/api-keys", dependencies=[Depends(require_admin_key)])
async def create_api_key(
    email: str,
    quota_tokens: Optional[int] = 10000,
    unlimited: bool = False,
):
    """
    Crée un nouvel utilisateur et retourne sa clé API pour la gateway.

    Réservé à l'administrateur de la gateway (clé BIOAI_ADMIN_KEY) : cet
    endpoint distribue le droit de consommer du GPU, et quota_tokens est un
    paramètre que l'appelant choisit. Ouvert, il annulerait tout l'intérêt
    des quotas.

    Args:
        email: Email de l'utilisateur.
        quota_tokens: Quota de tokens à allouer. Ignoré si unlimited est vrai.
        unlimited: Crée un compte sans plafond. Destiné aux passerelles qui
            appliquent déjà leurs propres quotas en amont (OpenGateLLM), pas
            aux utilisateurs humains.

    Returns:
        Un dict contenant l'email, la clé API générée et le quota (null si
        le compte est illimité).
    """
    quota = None if unlimited else quota_tokens
    api_key = token_manager.create_user(email, quota)
    return {
        "email": email,
        "api_key": api_key,
        "quota_tokens": quota,
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
            "POST /v1/chat/completions": "Idem au format chat (attendu par OpenGateLLM)",
            "POST /v1/embeddings": "Calculer des embeddings (format OpenAI)",
            "GET /v1/models": "Lister tous les modèles disponibles",
            "POST /v1/api-keys": "Créer un nouvel utilisateur (clé admin requise)",
            "GET /v1/api-keys/status": "Vérifier votre quota",
            "GET /health": "Health check (gateway + sous-serveurs)"
        },
        "docs": "http://localhost:8080/docs"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
