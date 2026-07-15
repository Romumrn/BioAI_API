# BioAI API

Here's the pitch: if you want to use a general-purpose AI model today (GPT, Mistral, Llama...), you've got plenty of unified APIs to reach for — Albert API, for instance, does a great job on the French sovereign side. But if you want to use a model specialized in bioinformatics or genomics (Evo, Nucleotide Transformer, BioMistral...), there's nothing like that. Each model has its own way of running, its own input/output format, and you end up hand-rolling the integration every single time.

BioAI API is an attempt to fill that gap: a single interface, OpenAI-style (API key, quotas, `/v1/...` endpoints), for tapping into different bio/genomics-oriented models, whatever the provider or runtime behind them. We're not trying to compete with Albert API or similar general-purpose platforms — we just want to do the same thing, but for a niche they don't cover: small, specialized bioinformatics models.

**Where things stand: this is a proof of concept.** Each model runs in its own FastAPI server, all built on the same pattern (API key, token quota, OpenAI-style response format). On top of that, a single [gateway](gateway.py) now handles authentication and quotas at the root, and routes each request to the right sub-server depending on the requested model — it's the only thing end users should ever call directly. The sub-servers are still reachable individually for debugging, but they keep their own separate key system.

We're open to adding pretty much any model that's useful for bio/genomics, as long as we can get it running somewhere. For now, we've started with four models chosen to cover fairly different cases:

- **[Evo](serveur_evo/)** — a generative model (StripedHyena) that produces DNA sequences token by token, kind of like a text LLM but for DNA. Exposed via `/v1/completions`.
- **[Nucleotide Transformer](serveur_nucleotide_transformer/)** — unlike Evo, this isn't generative: it's a BERT-style encoder that produces embeddings from DNA sequences, which you can then use for classification or analysis. Exposed via `/v1/embeddings`.
- **[GROVER](serveur_grover/)** — another BERT-style encoder, but trained only on the human genome, with a BPE tokenization built to reflect the "grammar" of human DNA rather than a fixed k-mer size. Same idea as Nucleotide Transformer (embeddings in, downstream task out), exposed the same way via `/v1/embeddings`.
- **[BioMistral](serveur_biomistral/)** — a biomedical text LLM (Mistral fine-tuned on PubMed). We picked this one to test running through an external runtime: it runs via Ollama (or vLLM as an alternative), because we quickly realized not every model can go through Ollama (Evo, Nucleotide Transformer and GROVER, for example, need their own loading code and don't fit the GGUF format). BioMistral, on the other hand, works well with it, so it's our test case for the "proxy to an external runtime" building block.

If any of this — tokenization, embeddings, masked vs. causal language modeling, why DNA even fits into an LLM-shaped box — sounds fuzzy, we put together a slide deck that walks through the concepts and through these exact models: **[romumrn.github.io/slides_AI_DNA](https://romumrn.github.io/slides_AI_DNA)**. Worth a skim before diving into the code.

## Repo structure

```
gateway.py                        # single entry point: auth + routing (port 8080)
start_all.py                      # starts everything automatically (sub-servers + gateway)
serveur_evo/                      # Evo — DNA sequence generation (port 8000)
serveur_nucleotide_transformer/   # Nucleotide Transformer — DNA embeddings (port 8001)
serveur_biomistral/               # BioMistral — biomedical text, via Ollama/vLLM (port 8002)
serveur_grover/                   # GROVER — DNA embeddings (port 8003)
```

Inside each model folder:
- `<model>_server.py` — the FastAPI server
- `create.py` — creates a test API key (for direct/debug use, not the gateway's)
- `test.py` — sends a sample request
- `requirement.txt` — Python dependencies

## Start everything at once

`start_all.py` scans the subfolders for a `*_server.py` file, starts each one, waits for it to respond on `/health`, then starts the gateway on top:

```bash
pip install -r requirement.txt   # gateway dependencies
python start_all.py
```

Ctrl+C shuts everything down cleanly. Once it's up, everything goes through the gateway at `http://localhost:8080` (docs at `/docs`):

```bash
# create an API key (gateway-side — this is the only one that matters to an end user)
curl -X POST "http://localhost:8080/v1/api-keys?email=me@example.com&quota_tokens=5000"

# use it
curl -X POST http://localhost:8080/v1/completions \
  -H "Authorization: Bearer sk-..." \
  -H "Content-Type: application/json" \
  -d '{"model": "evo-1.5-8k-base", "prompt": "ATGC", "max_tokens": 50}'
```

For BioMistral, you'll also need Ollama running with the model pulled, before starting `start_all.py`:

```bash
ollama pull cniongolo/biomistral
ollama serve
```

Evo, Nucleotide Transformer and GROVER all need `torch` + `transformers` (see their `requirement.txt`), plus a GPU if you want anything faster than a crawl — and Evo's weights alone are a multi-gigabyte download from Hugging Face on first run, so make sure you've got disk space and time before kicking it off. One extra wrinkle we hit in practice: Nucleotide Transformer ships custom modeling code that needs `trust_remote_code=True` and only works with an older `transformers` release, so it may need its own virtualenv if your main env is already on a newer version (GROVER, being a plain BERT architecture, doesn't have this problem).

## Starting a single model server (debug)

```bash
cd serveur_evo  # or one of the other three
pip install -r requirement.txt
python evo_server.py       # or nt_server.py / biomistral_server.py / grover_server.py
python create.py           # generates a local API key in token.txt
python test.py             # sends a test request, directly, bypassing the gateway
```

## What's next?

The gateway currently routes based on a static model registry (`BACKENDS` in `gateway.py`): adding a model means starting its server and adding a line. Next steps we're eyeing: a dynamic registry (auto-discovery of sub-servers by the gateway itself, not just by `start_all.py`), and running each sub-server in its own container to isolate dependencies (Evo, for instance, needs its own `evo-model` package and a torch version that the other models may not share).
