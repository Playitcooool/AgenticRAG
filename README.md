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

## Repository Layout

```text
src/agentic_rag/   Core implementation
tests/             Unit tests for retrieval and orchestration
datasets/          Existing benchmark datasets
```

## Status

This implementation is intentionally lightweight and dependency-free. It uses deterministic lexical retrieval and rule-based agents so the control flow is easy to inspect and test locally. Domain behavior should come from prompts, data, or optional task extractors layered on top, not from fixed assumptions inside the core pipeline.
