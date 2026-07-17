"""
Test rapide de nucleotide-transformer-v2-100m-multi-species à travers la gateway.

La clé API est celle de la gateway (voir create_user.py à la racine) : elle
donne accès à tous les modèles, c'est le champ "model" ci-dessous qui choisit
lequel répond.

Usage:
    python test.py [sequence]
"""
import sys
from pathlib import Path

import requests

GATEWAY_URL = "http://localhost:8080"
MODEL = "nucleotide-transformer-v2-100m-multi-species"

# token.txt vit à la racine du repo, créé par create_user.py
TOKEN_FILE = Path(__file__).resolve().parent.parent / "token.txt"
try:
    token = TOKEN_FILE.read_text().strip()
except FileNotFoundError:
    print(f"{{TOKEN_FILE}} non trouvé — lancez d'abord: python create_user.py")
    sys.exit(1)

SEQ = sys.argv[1] if len(sys.argv) > 1 else "ATTCCGATTCCGATTCCG"

r = requests.post(
    f"{GATEWAY_URL}/v1/embeddings",
    json={"model": MODEL, "input": SEQ},
    headers={"Authorization": f"Bearer {token}"},
)

if r.ok:
    data = r.json()
    embedding = data["data"][0]["embedding"]

    print(f"\nSéquence : {SEQ}")
    print(f"Embedding: dimension {len(embedding)}, premières valeurs {embedding[:5]}")
    print(f"\nTokens utilisés: {data['usage']['total_tokens']}")
else:
    print("Erreur:", r.text)
