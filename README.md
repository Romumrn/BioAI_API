# BioAI API

L'idée de départ, c'est simple : aujourd'hui si vous voulez utiliser un modèle d'IA généraliste (GPT, Mistral, Llama...), vous avez plein d'API unifiées pour y accéder facilement — Albert API par exemple, qui fait ça très bien côté souverain français. Mais si vous voulez utiliser un modèle spécialisé en bioinformatique ou en génomique (Evo, Nucleotide Transformer, BioMistral...), il n'y a rien de tel : chaque modèle a sa propre façon de tourner, son propre format d'entrée/sortie, et il faut tout recoder à la main à chaque fois.

BioAI API, c'est une tentative de combler ce trou : une interface unique, façon OpenAI (clé API, quotas, endpoints `/v1/...`), pour venir taper dans différents modèles orientés bio/génomique, quel que soit le fournisseur ou le runtime derrière. On ne cherche pas à concurrencer Albert API ou ce genre de plateformes généralistes — on veut juste faire la même chose, mais sur un créneau qu'elles ne couvrent pas : les petits modèles spécialisés bio-informatique.

**Où on en est : c'est une preuve de concept.** Chaque modèle tourne dans son propre serveur FastAPI, tous construits sur le même patron (clé API, quota de tokens, format de réponse OpenAI). Par-dessus, une [gateway](gateway.py) unique gère désormais l'authentification et le quota à la racine, et route chaque requête vers le bon sous-serveur selon le modèle demandé — c'est elle, et seulement elle, que les utilisateurs finaux doivent appeler. Les sous-serveurs restent joignables individuellement pour du debug, mais gardent leur propre système de clés séparé.

On est ouverts à ajouter à peu près n'importe quel modèle qui a un intérêt en bio/génomique, du moment qu'on peut le faire tourner quelque part. Pour l'instant, on a démarré avec trois modèles choisis pour couvrir des cas assez différents :

- **[Evo](serveur_evo/)** — un modèle génératif (StripedHyena) qui produit des séquences ADN token par token, un peu comme un LLM de texte mais sur de l'ADN. Exposé via `/v1/completions`.
- **[Nucleotide Transformer](serveur_nucleotide_transformer/)** — à l'inverse d'Evo, ce n'est pas un modèle génératif : c'est un encodeur (type BERT) qui sert à produire des embeddings de séquences ADN, utilisables ensuite pour de la classification ou de l'analyse. Exposé via `/v1/embeddings`.
- **[BioMistral](serveur_biomistral/)** — un LLM de texte biomédical (Mistral fine-tuné sur PubMed). C'est celui qu'on a choisi pour tester le passage par un runtime externe : il tourne via Ollama (ou vLLM en alternative), parce qu'on s'est vite rendu compte que tous les modèles ne peuvent pas passer par Ollama (Evo et Nucleotide Transformer, par exemple, ont besoin de leur propre code de chargement et ne rentrent pas dans le format GGUF). BioMistral, lui, s'y prête bien, donc ça nous sert de cas de test pour la brique "proxy vers un runtime externe".

## Structure du repo

```
gateway.py                        # point d'entrée unique : auth + routage (port 8080)
start_all.py                      # lance tout automatiquement (sous-serveurs + gateway)
serveur_evo/                      # Evo — génération de séquences ADN (port 8000)
serveur_nucleotide_transformer/   # Nucleotide Transformer — embeddings ADN (port 8001)
serveur_biomistral/               # BioMistral — texte biomédical, via Ollama/vLLM (port 8002)
```

Dans chaque dossier de modèle :
- `<modele>_server.py` — le serveur FastAPI
- `create.py` — crée une clé API de test (à usage direct/debug, pas celle de la gateway)
- `test.py` — envoie une requête d'exemple
- `requirement.txt` — dépendances Python

## Tout démarrer d'un coup

`start_all.py` scanne les sous-dossiers à la recherche d'un fichier `*_server.py`, lance chacun d'eux, attend qu'ils répondent sur `/health`, puis démarre la gateway :

```bash
pip install -r requirement.txt   # dépendances de la gateway
python start_all.py
```

Ctrl+C arrête tout proprement. Une fois lancé, tout passe par la gateway sur `http://localhost:8080` (docs sur `/docs`) :

```bash
# créer une clé API (côté gateway, c'est la seule qui compte pour un utilisateur final)
curl -X POST "http://localhost:8080/v1/api-keys?email=moi@example.com&quota_tokens=5000"

# l'utiliser
curl -X POST http://localhost:8080/v1/completions \
  -H "Authorization: Bearer sk-..." \
  -H "Content-Type: application/json" \
  -d '{"model": "evo-1.5-8k-base", "prompt": "ATGC", "max_tokens": 50}'
```

Pour BioMistral, il faut en plus avoir Ollama qui tourne avec le modèle pullé, avant de lancer `start_all.py` :

```bash
ollama pull cniongolo/biomistral
ollama serve
```

## Démarrer un serveur de modèle seul (debug)

```bash
cd serveur_evo  # ou l'un des deux autres
pip install -r requirement.txt
python evo_server.py       # ou nt_server.py / biomistral_server.py
python create.py           # génère une clé API locale dans token.txt
python test.py             # envoie une requête de test, en direct, sans passer par la gateway
```

## Et après ?

La gateway route aujourd'hui sur un registre de modèles statique (`BACKENDS` dans `gateway.py`) : ajouter un modèle, c'est démarrer son serveur et rajouter une ligne. Prochaines pistes : un registre dynamique (auto-découverte des sous-serveurs par la gateway elle-même, pas seulement par `start_all.py`), et faire tourner chaque sous-serveur dans son propre conteneur pour isoler les dépendances (Evo, par exemple, a besoin de son propre package `evo-model` et de torch, que les autres modèles n'utilisent pas forcément de la même version).
