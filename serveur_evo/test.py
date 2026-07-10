import requests
import sys

# Paramètres avec valeurs par défaut
SEQ = sys.argv[1] if len(sys.argv) > 1 else "ATGC"
LONGUEUR = int(sys.argv[2]) if len(sys.argv) > 2 else 50
TEMPERATURE = float(sys.argv[3]) if len(sys.argv) > 3 else 0.7

# Lire le token
try:
    token = open("token.txt").read().strip()
except FileNotFoundError:
    print("Fichier token.txt non trouvé")
    sys.exit(1)

# Générer
r = requests.post("http://localhost:8000/v1/completions",
                  json={
                      "prompt": SEQ,
                      "max_tokens": LONGUEUR,
                      "temperature": TEMPERATURE
                  },
                  headers={"Authorization": f"Bearer {token}"})

if r.ok:
    data = r.json()
    completion = data["choices"][0]["text"]
    remaining_tokens = None  # pas renvoyé directement dans le format OpenAI standard

    print(f"\nSéquences:")
    print(f"   Input : {SEQ}")
    print(f"   Output: {SEQ}{completion}")
    print(f"\nTokens utilisés: {data['usage']['total_tokens']} "
          f"(prompt: {data['usage']['prompt_tokens']}, "
          f"generation: {data['usage']['completion_tokens']})")
else:
    print("Erreur:", r.text)
