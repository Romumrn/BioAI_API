"""
Test rapide de biomistral-7b à travers la gateway.

La clé API est celle de la gateway (voir create_user.py à la racine) : elle
donne accès à tous les modèles, c'est le champ "model" ci-dessous qui choisit
lequel répond.

Usage:
    python test.py [prompt] [max_tokens] [temperature]
"""
import sys
from pathlib import Path

import requests

GATEWAY_URL = "http://localhost:8080"
MODEL = "biomistral-7b"

# token.txt vit à la racine du repo, créé par create_user.py
TOKEN_FILE = Path(__file__).resolve().parent.parent / "token.txt"
try:
    token = TOKEN_FILE.read_text().strip()
except FileNotFoundError:
    print(f"{{TOKEN_FILE}} non trouvé — lancez d'abord: python create_user.py")
    sys.exit(1)

PROMPT = sys.argv[1] if len(sys.argv) > 1 else "What is the mechanism of action of aspirin?"
MAX_TOKENS = int(sys.argv[2]) if len(sys.argv) > 2 else 200
TEMPERATURE = float(sys.argv[3]) if len(sys.argv) > 3 else 0.7

r = requests.post(
    f"{GATEWAY_URL}/v1/completions",
    json={
        "model": MODEL,
        "prompt": PROMPT,
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
    },
    headers={"Authorization": f"Bearer {token}"},
)

if r.ok:
    data = r.json()
    completion = data["choices"][0]["text"]

    print(f"\nPrompt  : {PROMPT}")
    print(f"Réponse : {completion}")
    print(f"\nTokens utilisés: {data['usage']['total_tokens']} "
          f"(prompt: {data['usage']['prompt_tokens']}, "
          f"generation: {data['usage']['completion_tokens']})")
else:
    print("Erreur:", r.text)
