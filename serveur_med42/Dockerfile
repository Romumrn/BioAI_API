# Med42 — dialogue biomédical, relayé vers Ollama
#
# Aucun modèle chargé ici : ce serveur est un proxy, d'où l'absence de torch
# et une image ~200 Mo au lieu de plusieurs Go.
#
# Build depuis la RACINE du repo (contexte incluant common/) :
#   docker build -f serveur_med42/Dockerfile .
FROM python:3.11-slim

WORKDIR /app

COPY serveur_med42/requirement.txt .
RUN pip install --no-cache-dir -r requirement.txt

COPY common/ ./common/
COPY serveur_med42/med42_server.py .

ENV BIOAI_BIND_HOST=0.0.0.0
EXPOSE 8002

CMD ["python", "med42_server.py"]
