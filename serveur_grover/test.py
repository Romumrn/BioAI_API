import requests
import sys

# Paramètres avec valeurs par défaut
SEQ = sys.argv[1] if len(sys.argv) > 1 else "ATTCCGATTCCGATTCCG"

# Lire le token
try:
    token = open("token.txt").read().strip()
except FileNotFoundError:
    print("Fichier token.txt non trouvé")
    sys.exit(1)

# Calculer l'embedding
r = requests.post("http://localhost:8003/v1/embeddings",
                  json={"input": SEQ},
                  headers={"Authorization": f"Bearer {token}"})

if r.ok:
    data = r.json()
    embedding = data["data"][0]["embedding"]

    print(f"\nSéquence : {SEQ}")
    print(f"Embedding: dimension {len(embedding)}, premières valeurs {embedding[:5]}")
    print(f"\nTokens utilisés: {data['usage']['total_tokens']}")
else:
    print("Erreur:", r.text)
