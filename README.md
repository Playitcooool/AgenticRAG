# AgenticRAG

AgenticRAG is a compact Python implementation of an agentic retrieval-augmented generation workflow inspired by Google's agentic RAG pattern. The architecture is domain-agnostic: it extracts information needs from the request text, retrieves evidence for each need, checks whether the retrieved context is sufficient, and iterates when gaps remain.

The included demo uses a clinical-style request for medications, diet instructions, and allergy or adverse-event history, but those areas are not hard-coded into the agents.

The pipeline covers five phases:

1. **Orchestration**: a root agent parses the request, a planner decomposes it into generic information needs, and a query rewriter creates focused search fanouts.
2. **Search**: a RAG agent retrieves relevant document snippets across all fanout queries.
3. **Sufficient Context**: a quality-control agent reviews retrieved snippets, an intermediate draft, and explicit missing-piece analysis.
4. **Iteration**: missing context feedback triggers narrower follow-up searches.
5. **Synthesis**: once context is sufficient, a synthesis agent writes the final grounded summary.

## Quick Start

```bash
uv venv
uv run agentic-rag-demo
```

## FAISS vs turbovec Experiment

Install the optional benchmark backends:

```bash
uv sync --extra bench
```

Run the full benchmark on the existing datasets:

```bash
uv run agentic-rag-benchmark --datasets medical hotpotqa --backends faiss-flat faiss-pq turbovec
```

Write full results to JSON:

```bash
uv run agentic-rag-benchmark \
  --datasets medical hotpotqa musique 2wikimultihop novel \
  --backends faiss-flat faiss-pq turbovec \
  --output results/faiss_vs_turbovec.json
```

The experiment keeps the agentic RAG architecture fixed and swaps only the retrieval data backbone. It reports:

- sufficient-context rate after the iterative agent loop
- evidence recall when benchmark questions provide evidence strings
- average retrieval/synthesis loop rounds
- average latency per question

`turbovec` is wired through its Python `TurboQuantIndex(dim=..., bit_width=...)` API. FAISS is wired through both `IndexFlatIP` (`faiss-flat`, an exact-search ceiling) and `IndexPQ` (`faiss-pq`, a compressed-index baseline). The benchmark also accepts `--backends lexical exact-numpy` for debugging when native vector packages are unavailable.

Use a local OpenAI-compatible LLM for task decomposition, sufficient-context judging, and final synthesis:

```bash
uv run agentic-rag-benchmark \
  --datasets medical \
  --backends lexical
```

Model settings live in `config.yaml`:

```yaml
llm:
  base_url: "http://localhost:1234"
  model: "unsloth:gemma-4-E4B-it-UD-MLX-4bit"
  api_key: "no_need"
  timeout: 120
  temperature: 0.0
embeddings:
  provider: "local"
  model_path: "models/embeddinggemma-300m"
  batch_size: 32
```

The vector benchmark uses a local EmbeddingGemma model loaded from `embeddings.model_path`; it does not call an embedding API. Put a local SentenceTransformers-compatible copy of `google/embeddinggemma-300m` at `models/embeddinggemma-300m`, or override the path. If you need the old deterministic vectors for debugging only, pass `--embedding-provider hashing`.

Chunk embeddings are cached on disk per dataset in `.cache/embeddings/<dataset>/`. The cache key includes the embedding provider/model settings and corpus chunk hashes, so FAISS and turbovec reuse the same dataset chunk vectors instead of re-embedding text for every backend run.

You can split embedding and RAG phases:

```bash
EMBEDDING_ONLY=1 DATASETS=medical ./scripts/run_faiss_turbovec_llm_experiment.sh
RAG_ONLY=1 DATASETS=medical ./scripts/run_faiss_turbovec_llm_experiment.sh
```

`RAG_ONLY=1` requires the dataset cache to already exist and will not compute missing corpus embeddings.

For small datasets, `faiss-pq` with the default `--bit-width 4` may be skipped because FAISS recommends at least 624 training vectors. Use `--bit-width 2` or omit `faiss-pq` for smaller corpora.

Use another config file or override individual values when needed:

```bash
uv run agentic-rag-benchmark --config config.yaml --llm-model another-model-id
```

To start the FAISS/turbovec LLM experiment yourself from a terminal:

```bash
./scripts/run_faiss_turbovec_llm_experiment.sh
```

Useful smoke-test override:

```bash
LIMIT=25 RUN_NAME=faiss_vs_turbovec_llm_limit25 ./scripts/run_faiss_turbovec_llm_experiment.sh
```

If your local EmbeddingGemma files live elsewhere:

```bash
EMBEDDING_MODEL_PATH=/path/to/embeddinggemma-300m ./scripts/run_faiss_turbovec_llm_experiment.sh
```

The script streams output to your terminal and also writes:

- `results/<run-name>.log`
- `results/<run-name>.json`

## Repository Layout

```text
src/agentic_rag/   Core implementation
tests/             Unit tests for retrieval and orchestration
datasets/          Existing benchmark datasets
```

## Status

This implementation is intentionally lightweight and dependency-free. It uses deterministic lexical retrieval and rule-based agents so the control flow is easy to inspect and test locally. Domain behavior should come from prompts, data, or optional task extractors layered on top, not from fixed assumptions inside the core pipeline.
