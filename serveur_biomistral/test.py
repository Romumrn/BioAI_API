import requests
import sys

# Paramètres avec valeurs par défaut
PROMPT = sys.argv[1] if len(sys.argv) > 1 else "What is the mechanism of action of aspirin?"
MAX_TOKENS = int(sys.argv[2]) if len(sys.argv) > 2 else 200
TEMPERATURE = float(sys.argv[3]) if len(sys.argv) > 3 else 0.7

# Lire le token
try:
    token = open("token.txt").read().strip()
except FileNotFoundError:
    print("Fichier token.txt non trouvé")
    sys.exit(1)

# Générer
r = requests.post("http://localhost:8002/v1/completions",
                  json={
                      "prompt": PROMPT,
                      "max_tokens": MAX_TOKENS,
                      "temperature": TEMPERATURE
                  },
                  headers={"Authorization": f"Bearer {token}"})

if r.ok:
    data = r.json()
    completion = data["choices"][0]["text"]

    print(f"\nPrompt: {PROMPT}")
    print(f"Réponse: {completion}")
    print(f"\nTokens utilisés: {data['usage']['total_tokens']} "
          f"(prompt: {data['usage']['prompt_tokens']}, "
          f"generation: {data['usage']['completion_tokens']})")
else:
    print("Erreur:", r.text)
