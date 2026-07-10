import json
import secrets
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict
import requests
from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# ============ CONFIGURATION ============
#
# Registre des modèles disponibles : nom du modèle -> serveur qui le sert.
# Pour ajouter un modèle, il suffit de démarrer son serveur (voir start_all.py)
# et de rajouter une entrée ici avec son nom de modèle, son port et son type
# (completions ou embeddings, selon l'endpoint que le sous-serveur expose).

BACKENDS = {
    "evo-1.5-8k-base": {
        "url": "http://localhost:8000",
        "kind": "completions",
        "owned_by": "evo",
    },
    "nucleotide-transformer-v2-100m-multi-species": {
        "url": "http://localhost:8001",
        "kind": "embeddings",
        "owned_by": "instadeep",
    },
    "biomistral-7b": {
        "url": "http://localhost:8002",
        "kind": "completions",
        "owned_by": "biomistral",
    },
}

TOKENS_FILE = "tokens_db.json"          # clés API des utilisateurs finaux de la gateway
SERVICE_KEYS_FILE = "service_keys.json"  # clés internes gateway -> sous-serveur


# ============ GESTION DES TOKENS (utilisateurs finaux) ============
# (identique aux serveurs de modèles, c'est la seule instance qui compte
# désormais : c'est ici, et seulement ici, que l'authentification a lieu)

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


# ============ CLÉS INTERNES GATEWAY -> SOUS-SERVEURS ============
#
# Les sous-serveurs (evo_server.py, nt_server.py, biomistral_server.py) ont
# chacun leur propre authentification. Plutôt que de la leur retirer, la
# gateway se crée pour chacun une clé de service à quota quasi illimité,
# qu'elle réutilise pour tous les appels internes. Le quota qui compte pour
# l'utilisateur final est celui de la gateway (token_manager ci-dessus), pas
# celui du sous-serveur.

def _load_service_keys() -> Dict[str, str]:
    """Charge les clés de service déjà générées, si le fichier existe."""
    if Path(SERVICE_KEYS_FILE).exists():
        with open(SERVICE_KEYS_FILE, 'r') as f:
            return json.load(f)
    return {}


def _save_service_keys(keys: Dict[str, str]):
    """Sauvegarde les clés de service sur disque."""
    with open(SERVICE_KEYS_FILE, 'w') as f:
        json.dump(keys, f, indent=2)


_service_keys = _load_service_keys()


def get_service_key(backend_url: str) -> str:
    """
    Récupère (ou crée) la clé de service utilisée par la gateway pour
    s'authentifier auprès d'un sous-serveur donné.

    Args:
        backend_url: URL de base du sous-serveur (ex: "http://localhost:8000").

    Returns:
        La clé API de service pour ce sous-serveur.

    Raises:
        HTTPException: 500 si le sous-serveur est injoignable.
    """
    if backend_url in _service_keys:
        return _service_keys[backend_url]

    try:
        resp = requests.post(
            f"{backend_url}/v1/api-keys",
            params={"email": "gateway@internal", "quota_tokens": 10**12},
            timeout=10,
        )
        resp.raise_for_status()
        api_key = resp.json()["api_key"]
    except requests.RequestException as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": {
                    "message": f"Backend unreachable ({backend_url}): {e}. "
                                "Le sous-serveur a-t-il bien démarré ?",
                    "type": "server_error",
                    "code": "backend_unavailable"
                }
            }
        )

    _service_keys[backend_url] = api_key
    _save_service_keys(_service_keys)
    return api_key


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
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "message": f"Unknown or incompatible model '{model}' for this "
                                f"endpoint. Available: {available}",
                    "type": "invalid_request_error",
                    "code": "model_not_found"
                }
            }
        )
    return backend


def forward(backend: Dict, path: str, payload: dict) -> dict:
    """
    Relaie une requête vers un sous-serveur, avec la clé de service de la
    gateway, et renvoie le corps de la réponse en cas de succès.

    Args:
        backend: Entrée du registre BACKENDS ciblée.
        path: Chemin de l'endpoint sur le sous-serveur (ex: "/v1/completions").
        payload: Corps JSON à transmettre tel quel.

    Returns:
        Le corps JSON de la réponse du sous-serveur.

    Raises:
        HTTPException: le code renvoyé par le sous-serveur si celui-ci échoue,
            ou 500 si le sous-serveur est injoignable.
    """
    service_key = get_service_key(backend["url"])
    try:
        resp = requests.post(
            f"{backend['url']}{path}",
            json=payload,
            headers={"Authorization": f"Bearer {service_key}"},
            timeout=300,
        )
    except requests.RequestException as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": {
                    "message": f"Backend unreachable ({backend['url']}): {e}",
                    "type": "server_error",
                    "code": "backend_unavailable"
                }
            }
        )

    if not resp.ok:
        raise HTTPException(status_code=resp.status_code, detail=resp.json())

    return resp.json()


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

    Authentifie l'utilisateur via la clé de la gateway, détermine le
    sous-serveur cible d'après le champ "model" du corps de la requête,
    relaie la requête telle quelle, puis déduit les tokens consommés du
    quota de l'utilisateur (et non de celui du sous-serveur).

    Args:
        request: Requête brute (le corps est relayé tel quel au sous-serveur).
        authorization: Header Authorization au format "Bearer <api_key>".

    Returns:
        La réponse du sous-serveur, au format OpenAI.

    Raises:
        HTTPException: 401 si la clé API est invalide, 400 si le modèle est
            inconnu, 429 si le quota est dépassé, 500 en cas d'erreur backend.
    """
    user = authenticate(authorization)
    api_key = extract_bearer_token(authorization)
    payload = await request.json()

    backend = resolve_backend(payload.get("model"), expected_kind="completions")

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

    data = forward(backend, "/v1/completions", payload)

    tokens_used = data.get("usage", {}).get("total_tokens", 0)
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

    return JSONResponse(content=data)


@app.post("/v1/embeddings")
async def create_embeddings(request: Request, authorization: str = Header(None)):
    """
    Calcule des embeddings (format OpenAI), routé vers le bon modèle.

    Authentifie l'utilisateur via la clé de la gateway, détermine le
    sous-serveur cible d'après le champ "model" du corps de la requête,
    relaie la requête telle quelle, puis déduit les tokens consommés du
    quota de l'utilisateur (et non de celui du sous-serveur).

    Args:
        request: Requête brute (le corps est relayé tel quel au sous-serveur).
        authorization: Header Authorization au format "Bearer <api_key>".

    Returns:
        La réponse du sous-serveur, au format OpenAI.

    Raises:
        HTTPException: 401 si la clé API est invalide, 400 si le modèle est
            inconnu, 429 si le quota est dépassé, 500 en cas d'erreur backend.
    """
    user = authenticate(authorization)
    api_key = extract_bearer_token(authorization)
    payload = await request.json()

    backend = resolve_backend(payload.get("model"), expected_kind="embeddings")

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

    data = forward(backend, "/v1/embeddings", payload)

    tokens_used = data.get("usage", {}).get("total_tokens", 0)
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

    return JSONResponse(content=data)


@app.get("/v1/models")
async def list_models(authorization: str = Header(None)):
    """
    Liste tous les modèles disponibles, tous sous-serveurs confondus.

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
    user = authenticate(authorization)
    api_key = extract_bearer_token(authorization)
    return token_manager.get_status(api_key)


@app.get("/health")
async def health():
    """
    Vérifie l'état de la gateway et de chaque sous-serveur enregistré.

    Returns:
        Un dict avec le statut global et le détail par sous-serveur.
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
