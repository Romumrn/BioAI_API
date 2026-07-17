# Evo 1.5 — génération de séquences ADN (StripedHyena, GPU)
#
# Build depuis la RACINE du repo (contexte incluant common/) :
#   docker build -f serveur_evo/Dockerfile .
FROM python:3.10-slim

WORKDIR /app

# StripedHyena passe par des kernels Triton, que Triton compile à la volée à
# la PREMIÈRE GÉNÉRATION — pas au chargement du modèle. Sans chaîne de
# compilation C, l'image démarre, le healthcheck passe, et seule l'inférence
# échoue. C'est donc une dépendance d'exécution ici, pas de build : ne pas
# l'enlever en croyant alléger l'image.
#
# libc6-dev est explicite car Debian ne le déclare qu'en Recommends de gcc :
# avec --no-install-recommends, gcc s'installe sans les fichiers de démarrage
# de la libc (Scrt1.o, crti.o) et tout link échoue.
RUN apt-get update && apt-get install -y --no-install-recommends gcc libc6-dev \
    && rm -rf /var/lib/apt/lists/*

# Couche torch identique à nt/grover -> partagée par Docker entre les trois
# images. Ne pas la modifier isolément.
# --extra-index-url et pas --index-url : ce dernier REMPLACE PyPI, et
# l'index PyTorch n'héberge pas toutes les dépendances annexes de torch
# (typing-extensions & co) pour toutes les versions de Python — le build
# échouait dessus. Ici, torch vient de l'index PyTorch et le reste de PyPI.
RUN pip install --no-cache-dir torch==2.6.0 \
    --extra-index-url https://download.pytorch.org/whl/cu124

# evo-model -> stripedhyena -> flash-attn, qui n'a pas de wheel sur PyPI : pip
# le compile depuis les sources, ce qui exige nvcc (absent d'une image slim) et
# prend des dizaines de minutes. On installe donc le wheel précompilé publié
# par le projet, qui doit correspondre au triplet exact CUDA/torch/Python.
# L'ABI C++ est lue depuis le torch réellement installé plutôt que codée en
# dur : les wheels PyTorch ont changé d'ABI au fil des versions, et se tromper
# donne un `undefined symbol` à l'import, pas une erreur d'installation.
RUN ABI=$(python -c "import torch; print('TRUE' if torch._C._GLIBCXX_USE_CXX11_ABI else 'FALSE')") && \
    pip install --no-cache-dir \
    "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abi${ABI}-cp310-cp310-linux_x86_64.whl"

COPY serveur_evo/requirement.txt .
RUN pip install --no-cache-dir -r requirement.txt

COPY common/ ./common/
COPY serveur_evo/evo_server.py .

ENV BIOAI_BIND_HOST=0.0.0.0
EXPOSE 8000

CMD ["python", "evo_server.py"]
