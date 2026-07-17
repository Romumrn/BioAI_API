# GROVER — embeddings ADN (BERT standard, GPU)
#
# Build depuis la RACINE du repo, le contexte devant inclure common/ :
#   docker build -f serveur_grover/Dockerfile .
# (docker compose s'en charge via context: . + dockerfile:)
FROM python:3.10-slim

WORKDIR /app

# Couche torch isolée et STRICTEMENT identique dans evo/nt/grover : Docker la
# calcule une fois et la partage entre les trois images. La déplacer ou la
# modifier dans un seul des trois casse ce partage et coûte ~4 Go.
# --extra-index-url et pas --index-url : ce dernier REMPLACE PyPI, et
# l'index PyTorch n'héberge pas toutes les dépendances annexes de torch
# (typing-extensions & co) pour toutes les versions de Python — le build
# échouait dessus. Ici, torch vient de l'index PyTorch et le reste de PyPI.
RUN pip install --no-cache-dir torch==2.6.0 \
    --extra-index-url https://download.pytorch.org/whl/cu124

# requirement.txt copié seul d'abord : tant qu'il ne change pas, Docker
# réutilise le cache de cette couche même quand le code du serveur bouge.
COPY serveur_grover/requirement.txt .
RUN pip install --no-cache-dir -r requirement.txt

COPY common/ ./common/
COPY serveur_grover/grover_server.py .

# Le compose met 0.0.0.0 : sans ça, la gateway ne joindrait pas le conteneur
# depuis son propre namespace réseau. Le port n'est jamais publié vers l'hôte.
ENV BIOAI_BIND_HOST=0.0.0.0
EXPOSE 8003

CMD ["python", "grover_server.py"]
