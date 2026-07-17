"""
Crée un utilisateur sur la gateway et enregistre sa clé dans token.txt.

Remplace les create.py qui vivaient dans chaque dossier de modèle : les clés
API ne sont plus gérées que par la gateway, une seule clé donne donc accès à
tous les modèles.

Depuis que POST /v1/api-keys exige la clé admin, ce script doit la présenter.
Il la cherche là où elle se trouve selon le mode de déploiement : dans .env
sous docker compose, dans .admin_key en bare-metal.

Usage:
    python create_user.py [email] [quota_tokens]
    python create_user.py opengatellm@prabi.fr --unlimited
"""
import os
import sys
from pathlib import Path

import requests

from common import get_admin_key

GATEWAY_URL = os.getenv("BIOAI_GATEWAY_URL", "http://localhost:8080")
ENV_FILE = Path(__file__).resolve().parent / ".env"


def admin_key() -> str:
    """
    Récupère la clé admin de la gateway visée.

    .env est lu en priorité, et ce n'est pas cosmétique : sous docker compose,
    la gateway reçoit sa clé de ce fichier, alors que get_admin_key() créerait
    ici un .admin_key que la gateway conteneurisée n'a jamais vu. On se ferait
    refuser par notre propre gateway.

    Returns:
        La clé admin à présenter à la gateway.
    """
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text().splitlines():
            name, _, value = line.partition("=")
            if name.strip() == "BIOAI_ADMIN_KEY" and value.strip():
                return value.strip()

    return get_admin_key()


args = [a for a in sys.argv[1:] if not a.startswith("--")]
unlimited = "--unlimited" in sys.argv

email = args[0] if args else "test@example.com"
quota = int(args[1]) if len(args) > 1 else 5000

r = requests.post(
    f"{GATEWAY_URL}/v1/api-keys",
    params={"email": email, "quota_tokens": quota, "unlimited": unlimited},
    headers={"Authorization": f"Bearer {admin_key()}"},
)

if r.ok:
    open("token.txt", "w").write(r.json()["api_key"])
    budget = "quota illimité" if unlimited else f"{quota} tokens"
    print(f"✅ Utilisateur {email} créé ({budget}). Clé dans token.txt")
else:
    print("❌", r.status_code, r.text)
