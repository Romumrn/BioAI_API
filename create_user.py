"""
Crée un utilisateur sur la gateway et enregistre sa clé dans token.txt.

Remplace les create.py qui vivaient dans chaque dossier de modèle : les clés
API ne sont plus gérées que par la gateway, une seule clé donne donc accès à
tous les modèles.

Usage:
    python create_user.py [email] [quota_tokens]
"""
import sys

import requests

GATEWAY_URL = "http://localhost:8080"

email = sys.argv[1] if len(sys.argv) > 1 else "test@example.com"
quota = int(sys.argv[2]) if len(sys.argv) > 2 else 5000

r = requests.post(
    f"{GATEWAY_URL}/v1/api-keys",
    params={"email": email, "quota_tokens": quota},
)

if r.ok:
    open("token.txt", "w").write(r.json()["api_key"])
    print(f"✅ Utilisateur {email} créé ({quota} tokens). Clé dans token.txt")
else:
    print("❌", r.text)
