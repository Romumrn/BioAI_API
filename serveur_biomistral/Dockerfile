# BioMistral — texte biomédical, relayé vers Ollama (ou vLLM)
#
# Aucun modèle chargé ici : ce serveur est un proxy, d'où l'absence de torch
# et une image ~200 Mo au lieu de plusieurs Go.
#
# Build depuis la RACINE du repo (contexte incluant common/) :
#   docker build -f serveur_biomistral/Dockerfile .
FROM python:3.11-slim

WORKDIR /app

COPY serveur_biomistral/requirement.txt .
RUN pip install --no-cache-dir -r requirement.txt

COPY common/ ./common/
COPY serveur_biomistral/biomistral_server.py .

ENV BIOAI_BIND_HOST=0.0.0.0
EXPOSE 8002

CMD ["python", "biomistral_server.py"]
