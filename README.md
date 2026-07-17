# BioAI API

## Background

There are already several unified APIs for accessing general-purpose AI models (GPT, Mistral, Llama, etc.). In France, Albert API is a good example on the sovereign side: a single interface, one key and quota system, to talk to several models without rewriting the integration for each one.

For models specialized in bioinformatics or genomics (Evo, Nucleotide Transformer, BioMistral...), though, nothing like that exists. Each model has its own input/output format, its own way of running, and you end up hand-rolling the integration every single time.

The idea behind BioAI API is simply to take the same principle as Albert API and apply it to this niche: a single interface, OpenAI-style (API key, quotas, `/v1/...` endpoints), for accessing different bio/genomics-oriented models, whatever the provider or runtime behind them. This isn't an attempt to compete with Albert API or similar general-purpose platforms, just to do the same thing for a niche they don't cover.

## Current state

This is a proof of concept. Each model runs in its own FastAPI server, all built on the same pattern (API key, token quota, OpenAI-style response format). On top of that, a single [gateway](gateway.py) handles authentication and quotas, and routes each request to the right sub-server depending on the requested model — it's the only entry point end users should ever call. The sub-servers are still reachable individually for debugging, but they keep their own separate key system.

We're open to adding pretty much any model useful for bio/genomics, as long as we can get it running somewhere. For now, we've started with five models chosen to cover fairly different cases:

- **[Evo](serveur_evo/)** — a generative model (StripedHyena) that produces DNA sequences token by token, a bit like a text LLM but for DNA. Exposed via `/v1/completions`.
- **[Nucleotide Transformer](serveur_nucleotide_transformer/)** — unlike Evo, this one isn't generative: it's a BERT-style encoder that produces embeddings from DNA sequences, which can then be reused for classification or analysis. Exposed via `/v1/embeddings`.
- **[GROVER](serveur_grover/)** — another BERT-style encoder, but trained only on the human genome, with a BPE tokenizer built to reflect the "grammar" of human DNA rather than a fixed k-mer size. Same idea as Nucleotide Transformer (embeddings out, downstream task after), exposed the same way via `/v1/embeddings`.
- **[DNABERT-2](serveur_dnabert2/)** — a third BERT-style DNA encoder, but multi-species like Nucleotide Transformer, with its own BPE tokenizer (trained across 135 genomes) instead of GROVER's human-only one. Same `/v1/embeddings` shape as the other two, chosen specifically because it isn't redundant with them: different tokenization strategy, different species coverage, and a different loading path (`AutoModel`, not `AutoModelForMaskedLM`, since its custom modeling code doesn't register a masked-LM head — the embeddings server pulls the last hidden state directly from `outputs[0]` instead of using `output_hidden_states=True`).
- **[BioMistral](serveur_biomistral/)** — a biomedical text LLM (Mistral fine-tuned on PubMed). We picked this one to test running through an external runtime: it runs via Ollama. BioMistral, on the other hand, works fine with it, so it's our test case for the "proxy to an external runtime" building block.

For futher information please visit: **[romumrn.github.io/slides_AI_DNA](https://romumrn.github.io/slides_AI_DNA)**. 

## Repo structure

```
gateway.py                        # single entry point: auth + routing (port 8080)
start_all.py                      # starts everything automatically (sub-servers + gateway)
serveur_evo/                      # Evo — DNA sequence generation (port 8000)
serveur_nucleotide_transformer/   # Nucleotide Transformer — DNA embeddings (port 8001)
serveur_biomistral/               # BioMistral — biomedical text, via Ollama/vLLM (port 8002)
serveur_grover/                   # GROVER — DNA embeddings (port 8003)
serveur_dnabert2/                 # DNABERT-2 — DNA embeddings, multi-species (port 8004)
```

Inside each model folder:
- `<model>_server.py` — the FastAPI server
- `create.py` — creates a test API key (for direct/debug use, not the gateway's)
- `test.py` — sends a sample request
- `requirement.txt` — Python dependencies

## Starting everything at once

`start_all.py` scans the subfolders for a `*_server.py` file, starts each one, waits for it to respond on `/health`, then starts the gateway on top. For each sub-server, it looks for a local `.venv*` folder (e.g. `.venv_nt`, `.venv_dnabert2`) and runs the server with that interpreter if one exists, falling back to whatever Python ran `start_all.py` otherwise. That way a sub-server pinned to an older `transformers` release doesn't need the whole project to be on the same version:

```bash
pip install -r requirement.txt   # gateway dependencies
python start_all.py
``` 

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

Evo, Nucleotide Transformer, GROVER and DNABERT-2 all need `torch` + `transformers` (see their `requirement.txt`), and ideally a GPU if you want anything faster than a crawl. Evo's weights alone are a multi-gigabyte download from Hugging Face on first run, so make sure you've got the disk space and time before kicking it off.

One thing we ran into in practice: Nucleotide Transformer and DNABERT-2 both ship custom modeling code that needs `trust_remote_code=True` and only works with an older `transformers` release (`<5`), so they may need their own virtualenv if your main environment is already on a newer version. GROVER, being a plain BERT architecture, doesn't have this problem (confirmed with `.venv_nt` for Nucleotide Transformer and `.venv_dnabert2` for DNABERT-2 in this repo). DNABERT-2 has one more quirk: its remote code only loads if `triton` is installed (recent `transformers` versions statically require every package the remote code imports, even ones behind a `try/except ImportError`), but that same `triton` package makes the code pick a CUDA-only fused-attention kernel that crashes on CPU/MPS. `dnabert2_server.py` works around this by force-disabling that kernel outside CUDA, right after the model loads.

## Starting a single model server (debug)

```bash
cd serveur_evo  # or one of the other folders
pip install -r requirement.txt
python evo_server.py       # or nt_server.py / biomistral_server.py / grover_server.py
python create.py           # generates a local API key in token.txt
python test.py             # sends a test request, directly, bypassing the gateway
```

## What's next?

Adding more models and have a OICD to manage Users