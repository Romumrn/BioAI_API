# Nucleotide Transformer — embeddings ADN (encodeur type ESM, GPU)
#
# Build depuis la RACINE du repo (contexte incluant common/) :
#   docker build -f serveur_nucleotide_transformer/Dockerfile .
FROM python:3.10-slim

WORKDIR /app

# Couche torch identique à evo/grover -> partagée par Docker entre les trois
# images. Ne pas la modifier isolément.
# --extra-index-url et pas --index-url : ce dernier REMPLACE PyPI, et
# l'index PyTorch n'héberge pas toutes les dépendances annexes de torch
# (typing-extensions & co) pour toutes les versions de Python — le build
# échouait dessus. Ici, torch vient de l'index PyTorch et le reste de PyPI.
RUN pip install --no-cache-dir torch==2.6.0 \
    --extra-index-url https://download.pytorch.org/whl/cu124

COPY serveur_nucleotide_transformer/requirement.txt .
RUN pip install --no-cache-dir -r requirement.txt

COPY common/ ./common/
COPY serveur_nucleotide_transformer/nt_server.py .

ENV BIOAI_BIND_HOST=0.0.0.0
EXPOSE 8001

CMD ["python", "nt_server.py"]
