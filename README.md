# BioAI API

## Background

There are already several unified APIs for accessing general-purpose AI models (GPT, Mistral, Llama, etc.). In France, Albert API is a good example on the sovereign side: a single interface, one key and quota system, to talk to several models without rewriting the integration for each one.

For models specialized in bioinformatics or genomics (Evo, Nucleotide Transformer, BioMistral...), though, nothing like that exists. Each model has its own input/output format, its own way of running, and you end up hand-rolling the integration every single time.

The idea behind BioAI API is simply to take the same principle as Albert API and apply it to this niche: a single interface, OpenAI-style (API key, quotas, `/v1/...` endpoints), for accessing different bio/genomics-oriented models, whatever the provider or runtime behind them. This isn't an attempt to compete with Albert API or similar general-purpose platforms, just to do the same thing for a niche they don't cover.

## Current state

This is a proof of concept. Each model runs in its own FastAPI server, all built on the same pattern (OpenAI-style request/response format). On top of that, a single [gateway](gateway.py) handles authentication and quotas, and routes each request to the right model server depending on the requested model — it's the only entry point end users should ever call.

### Where authentication lives

Users, API keys and quotas exist in exactly one place: the gateway. The model servers know nothing about users and count nothing. This split is deliberate:

- **The gateway** authenticates the caller, checks the quota, forwards the request, and debits the tokens actually consumed. It's the only service that publishes a port.
- **An admin secret** (`BIOAI_ADMIN_KEY`) guards `POST /v1/api-keys`. Creating a key means handing out the right to burn GPU time, and the caller picks their own `quota_tokens` — so that endpoint can't be open once the gateway is reachable from the internet. It's deliberately *not* the internal secret: the model servers know that one, and they have no business minting user keys.
- **The model servers** are never reachable by an end user. *How* that's enforced depends on how you run them: bare-metal they bind `127.0.0.1`; under Docker they must bind `0.0.0.0` to be reachable by the gateway from its own network namespace, and instead simply don't publish their port (`expose`, not `ports`). Both are driven by `BIOAI_BIND_HOST`, which defaults to `127.0.0.1`.
- **A shared internal secret** is what holds in both cases. Model endpoints reject any call that doesn't carry it — including a call carrying a perfectly valid *user* key, since those are meaningless to them. Bare-metal it's a second barrier behind the localhost bind; under Docker it's the only thing standing between a container on the compose network and the models, which is why `docker compose` refuses to start without `BIOAI_INTERNAL_KEY` set.

Both secrets come from the environment if set (that's what compose injects from `.env` — the internal one into all six containers, the admin one into the gateway alone), falling back to a git-ignored `.internal_key` / `.admin_key` file generated on first run — so a bare-metal server started on its own needs no configuration.

Earlier versions duplicated the whole key/quota system into every model server, which meant each one exposed an unauthenticated `POST /v1/api-keys` on `0.0.0.0` — anyone able to reach the port could mint themselves an unlimited key and bypass the gateway entirely. The shared code now lives in [`common/`](common/), and only the gateway has an authentication surface at all.

We're open to adding pretty much any model useful for bio/genomics, as long as we can get it running somewhere. For now, we've started with five models chosen to cover fairly different cases:

- **[Evo](serveur_evo/)** — a generative model (StripedHyena) that produces DNA sequences token by token, a bit like a text LLM but for DNA. Exposed via `/v1/completions` (and `/v1/chat/completions`, see below).
- **[Nucleotide Transformer](serveur_nucleotide_transformer/)** — unlike Evo, this one isn't generative: it's a BERT-style encoder that produces embeddings from DNA sequences, which can then be reused for classification or analysis. Exposed via `/v1/embeddings`.
- **[GROVER](serveur_grover/)** — another BERT-style encoder, but trained only on the human genome, with a BPE tokenizer built to reflect the "grammar" of human DNA rather than a fixed k-mer size. Same idea as Nucleotide Transformer (embeddings out, downstream task after), exposed the same way via `/v1/embeddings`.
- **[DNABERT-2](serveur_dnabert2/)** — a third BERT-style DNA encoder, but multi-species like Nucleotide Transformer, with its own BPE tokenizer (trained across 135 genomes) instead of GROVER's human-only one. Same `/v1/embeddings` shape as the other two, chosen specifically because it isn't redundant with them: different tokenization strategy, different species coverage, and a different loading path (`AutoModel`, not `AutoModelForMaskedLM`, since its custom modeling code doesn't register a masked-LM head — the embeddings server pulls the last hidden state directly from `outputs[0]` instead of using `output_hidden_states=True`).
- **[BioMistral](serveur_biomistral/)** — a biomedical text LLM (Mistral fine-tuned on PubMed). We picked this one to test running through an external runtime: it runs via Ollama. BioMistral, on the other hand, works fine with it, so it's our test case for the "proxy to an external runtime" building block.

For futher information please visit: **[romumrn.github.io/slides_AI_DNA](https://romumrn.github.io/slides_AI_DNA)**. 

## Repo structure

```
docker-compose.yml                # the whole stack; only the gateway publishes a port
Dockerfile                        # gateway image
.env.example                      # template for the internal + admin secrets
gateway.py                        # single entry point: auth + quotas + routing (127.0.0.1:8080, behind Caddy)
common/                           # code shared by the gateway and the model servers
start_all.py                      # bare-metal launcher (model servers + gateway)
create_user.py                    # creates an API key on the gateway, saves it to token.txt
serveur_evo/                      # Evo — DNA sequence generation (port 8000, internal)
serveur_nucleotide_transformer/   # Nucleotide Transformer — DNA embeddings (port 8001, internal)
serveur_biomistral/               # BioMistral — biomedical text, via Ollama/vLLM (port 8002, internal)
serveur_grover/                   # GROVER — DNA embeddings (port 8003, internal)
serveur_dnabert2/                 # DNABERT-2 — DNA embeddings, multi-species (port 8004, internal)
```

Inside `common/`:
- `tokens.py` — users and quotas (`TokenManager`). Gateway only.
- `internal.py` — the shared gateway → model-server secret. Both sides.
- `schemas.py` — OpenAI-format request/response models.
- `errors.py` — OpenAI-format HTTP errors.

It depends on nothing but the stdlib, FastAPI and Pydantic, so it imports cleanly from the isolated venvs some model servers need (`.venv_nt`, `.venv_dnabert2`).

Inside each model folder:
- `<model>_server.py` — the FastAPI server
- `Dockerfile` — its image (built from the repo root, so it can `COPY common/`)
- `test.py` — sends a sample request through the gateway
- `requirement.txt` — pinned Python dependencies

## Starting everything at once (Docker — the normal way)

The five models don't agree on much: Evo wants `transformers` 5.2 and `evo-model`, Nucleotide Transformer needs `transformers` <5 for its remote code, DNABERT-2 runs on Python 3.13 with a **CPU** build of torch, and BioMistral needs no torch at all. Each server gets its own image, so none of this leaks into your shell environment.

```bash
# one-time: generate the two secrets (internal + admin)
cp .env.example .env
python3 -c 'import secrets;print(f"BIOAI_INTERNAL_KEY={secrets.token_hex(32)}\nBIOAI_ADMIN_KEY={secrets.token_hex(32)}")' > .env

docker compose up -d
docker compose ps          # 6 services; only the gateway publishes a port
curl localhost:8080/health # gateway + every model server
```

```bash
# create an API key (gateway-side — the only kind of key there is).
# create_user.py finds the admin key in .env (or .admin_key) by itself.
python create_user.py me@example.com 5000

# use it
curl -X POST http://localhost:8080/v1/completions \
  -H "Authorization: Bearer $(cat token.txt)" \
  -H "Content-Type: application/json" \
  -d '{"model": "evo-1.5-8k-base", "prompt": "ATGC", "max_tokens": 50}'
```

The gateway publishes on `127.0.0.1:8080`, not `0.0.0.0`. It's meant to sit behind a reverse proxy that terminates TLS (see *Exposing the gateway* below); binding it wide would let anyone bypass that proxy, and Docker writes its own iptables rules, so a host firewall won't necessarily save you.

Two things the compose file expects from the host:

- **Ollama**, for BioMistral — that server is a proxy, it loads nothing itself. The container reaches the host's Ollama through `host.docker.internal`, so `ollama serve` must be up with the model pulled (`ollama pull cniongolo/biomistral`). Ollama's default bind is `127.0.0.1`, which no container can reach, so it has to listen wider:

  ```bash
  OLLAMA_HOST=0.0.0.0 ollama serve
  ```

  Be aware that this exposes Ollama's API, unauthenticated, on every interface of the host. If the machine is reachable from an untrusted network, bind it to the Docker bridge only (`OLLAMA_HOST=172.17.0.1`), which containers can still reach while the outside world can't.
- **A Hugging Face cache** at `/data_local/hf_cache` (override with `HF_CACHE_DIR` in `.env`), bind-mounted into the model containers. Without it every image would re-download multi-gigabyte weights. It's mounted read-write on purpose: `trust_remote_code` writes the downloaded modelling code into the cache, and `huggingface_hub` takes locks there.

Evo, Nucleotide Transformer and GROVER are given the GPU; DNABERT-2 deliberately runs on CPU (see `serveur_dnabert2/Dockerfile`). Evo's first start is slow — several GB of weights — hence its 10-minute healthcheck `start_period`.

## Starting everything at once (bare-metal, for iterating without a rebuild)

`start_all.py` scans the subfolders for a `*_server.py` file, starts each one, waits for it to respond on `/health`, then starts the gateway on top. For each sub-server, it looks for a local `.venv*` folder (e.g. `.venv_nt`, `.venv_dnabert2`) and runs the server with that interpreter if one exists, **falling back to whatever Python ran `start_all.py` otherwise**.

That fallback is the catch: `serveur_evo/` and `serveur_grover/` have no local venv, so they inherit your active environment, and they die on `import torch` if it hasn't got one. Run it from an environment that has the dependencies:

```bash
conda activate evo         # an env with torch + transformers + evo-model
python start_all.py
```

If a server does fail, `start_all.py` now says so within seconds instead of waiting 180s for a process that's already dead.

Evo, Nucleotide Transformer, GROVER and DNABERT-2 all need `torch` + `transformers` (see their `requirement.txt`, now pinned to the versions each one is actually known to work with), and ideally a GPU if you want anything faster than a crawl. Evo's weights alone are a multi-gigabyte download from Hugging Face on first run, so make sure you've got the disk space and time before kicking it off.

The version conflicts are the reason the Docker path exists. Nucleotide Transformer and DNABERT-2 both ship custom modeling code that needs `trust_remote_code=True` and only works with an older `transformers` release (`<5`), so bare-metal they need their own virtualenv if your main environment is already on a newer version (`.venv_nt` and `.venv_dnabert2` in this repo). GROVER, being a plain BERT architecture, doesn't have this problem. DNABERT-2 has one more quirk: its remote code only loads if `triton` is installed (recent `transformers` versions statically require every package the remote code imports, even ones behind a `try/except ImportError`), but that same `triton` package makes the code pick a CUDA-only fused-attention kernel that crashes on CPU/MPS. `dnabert2_server.py` works around this by force-disabling that kernel outside CUDA, right after the model loads — and its image installs a CPU build of torch so that path is the deterministic one.

## Starting a single model server (debug)

A model server no longer accepts user keys, so testing one still goes through the gateway — run both, then point `test.py` at the model you want via its `model` field (it already does):

```bash
cd serveur_evo  # or one of the other folders
pip install -r requirement.txt
python evo_server.py       # or nt_server.py / biomistral_server.py / grover_server.py
# in another shell, from the repo root:
python gateway.py
python create_user.py      # creates a key on the gateway, saved to token.txt
cd serveur_evo && python test.py
```

To hit a model server directly (bypassing the gateway), pass the internal secret rather than a user key:

```bash
curl -X POST http://localhost:8000/v1/completions \
  -H "Authorization: Bearer $(cat .internal_key)" \
  -H "Content-Type: application/json" \
  -d '{"model": "evo-1.5-8k-base", "prompt": "ATGC", "max_tokens": 50}'
```

Nothing is debited from any quota that way — quotas only exist on the gateway.

## Exposing the gateway

The gateway listens on `127.0.0.1:8080`. Caddy terminates TLS and is the only path in from the outside. On `prabi-cloud149.univ-lyon1.fr` it already served a Streamlit app on `/`, so the gateway got a path prefix rather than a new hostname (which would have meant a new DNS record):

```caddyfile
prabi-cloud149.univ-lyon1.fr {
   handle_path /bioai/* {      # strips the prefix: /bioai/v1/models -> /v1/models
      reverse_proxy localhost:8080
   }
   handle {
      reverse_proxy localhost:8501
   }
}
```

So the public base URL is **`https://prabi-cloud149.univ-lyon1.fr/bioai/`** — trailing slash included, and that slash is not decorative. See the OpenGateLLM section below.

## Plugging into OpenGateLLM

[OpenGateLLM](https://github.com/etalab-ia/OpenGateLLM) is an LLM gateway that can put this API behind its own routing, quotas and accounting. It talks to us as a plain OpenAI-compatible provider.

**Why `/v1/chat/completions` exists.** OpenGateLLM's provider client has exactly one entry for generation: `/v1/chat/completions`. The legacy `/v1/completions` format our model servers speak isn't in its endpoint table at all. So the gateway grows a `/v1/chat/completions` that flattens `messages` into a `prompt`, forwards to `/v1/completions`, and converts the answer back. Without it, only the embeddings models would be pluggable.

The flattening treats a lone user message as a raw prompt, deliberately: Evo is a **DNA** model, and wrapping a sequence in `User:` / `Assistant:` would make it continue an English conversation instead of the sequence — answering wrong rather than failing loudly. Multi-turn conversations only make sense for BioMistral, and get a labelled transcript.

`stream: true` is honoured, but it's emulated: no model server here streams, so the whole answer arrives in one SSE chunk after full generation. It exists because OpenGateLLM's playground asks for streaming by default.

**Set-up.** Create a service account with no quota of its own — OpenGateLLM already meters its own users, and double-metering would just mean the whole thing dies at an unpredictable moment:

```bash
python create_user.py opengatellm@prabi.univ-lyon1.fr --unlimited
```

Then, one router (model) per model, each with one provider of type `openai`:

```yaml
models:
  - name: nucleotide-transformer-v2-100m-multi-species
    type: text-embeddings-inference     # text-generation for evo / biomistral
    providers:
      - type: openai
        url: https://prabi-cloud149.univ-lyon1.fr/bioai/
        key: sk-...                     # the --unlimited key above
        model_name: nucleotide-transformer-v2-100m-multi-species
```

| Model | OpenGateLLM router `type` |
|---|---|
| `nucleotide-transformer-v2-100m-multi-species`, `grover`, `dnabert2-117m` | `text-embeddings-inference` |
| `evo-1.5-8k-base`, `biomistral-7b` | `text-generation` |

Three things worth knowing, each of which cost an afternoon to find out:

- **The trailing slash on `url` is mandatory.** OpenGateLLM builds request URLs with `urljoin()`, and `urljoin("https://host/bioai", "v1/models")` returns `https://host/v1/models` — the prefix is dropped and the request lands on Streamlit. With the slash it resolves to `/bioai/v1/models`, as intended.
- **`model_name` must match an `id` in our `GET /v1/models` exactly.** On provider creation OpenGateLLM calls that endpoint and asserts exactly one model matches, otherwise it refuses the provider as unreachable.
- **Registering an embeddings provider consumes quota.** OpenGateLLM probes `POST /v1/embeddings` with `input: "hello world"` to discover the vector size (512 for Nucleotide Transformer, 768 for GROVER and DNABERT-2). Yes, it sends English prose to a DNA model — it only measures the vector's length, so it doesn't matter.

## What's next?

Adding more models and have a OICD to manage Users