# DNABERT-2 — embeddings ADN multi-espèces (CPU)
#
# Seul serveur en torch CPU : son code custom bascule sur un kernel Triton
# CUDA-only dès que triton est importable, et ce kernel plante hors CUDA.
# dnabert2_server.py le neutralise quand DEVICE != cuda ; installer un torch
# CPU rend ce chemin déterministe et évite d'embarquer ~3 Go de libs CUDA.
#
# Build depuis la RACINE du repo (contexte incluant common/) :
#   docker build -f serveur_dnabert2/Dockerfile .
FROM python:3.13-slim

WORKDIR /app

# Index CPU explicite : le wheel torch par défaut de PyPI embarque CUDA.
# --extra-index-url et pas --index-url : ce dernier REMPLACE PyPI, et
# l'index PyTorch n'héberge pas toutes les dépendances annexes de torch
# (typing-extensions & co) pour toutes les versions de Python — le build
# échouait dessus. Ici, torch vient de l'index PyTorch et le reste de PyPI.
RUN pip install --no-cache-dir torch==2.13.0 \
    --extra-index-url https://download.pytorch.org/whl/cpu

COPY serveur_dnabert2/requirement.txt .
RUN pip install --no-cache-dir -r requirement.txt

COPY common/ ./common/
COPY serveur_dnabert2/dnabert2_server.py .

ENV BIOAI_BIND_HOST=0.0.0.0
EXPOSE 8004

CMD ["python", "dnabert2_server.py"]
