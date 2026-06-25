"""Benchmark FAISS and turbovec inside the agentic RAG architecture."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from agentic_rag.config import load_config
from agentic_rag.embedding_cache import load_or_compute_embeddings
from agentic_rag.embeddings import HashingEmbedder, LocalSentenceTransformerEmbedder
from agentic_rag.llm import LLMClientError, OpenAICompatibleClient, AnthropicMessagesClient
from agentic_rag.llm_agents import build_llm_agents
from agentic_rag.pipeline import AgenticRAGPipeline
from agentic_rag.retriever import LexicalRetriever
from agentic_rag.types import ContextStatus, DocumentChunk
from agentic_rag.vector_backends import BackendUnavailable, build_vector_index
from agentic_rag.vector_retriever import VectorRetriever


@dataclass
class BenchmarkRow:
    dataset: str
    backend: str
    questions: int
    sufficient_rate: float
    evidence_recall: float | None
    avg_rounds: float
    avg_latency_ms: float
    status: str = "ok"
    error: str = ""


@dataclass
class PreparedEmbeddings:
    embedder: object
    dim: int
    record_vectors: list[list[float]]


def main() -> None:
    args = _parse_args()
    config = load_config(args.config)
    llm_client = None if args.embedding_only else _build_llm_client(args, config)
    rows = []
    if not args.embedding_only:
        _print_header()
    for dataset in args.datasets:
        dataset_path = args.data_dir / dataset
        records = _load_records(dataset_path / "chunks.json", source=dataset)
        questions = _load_questions(dataset_path / "questions.json")
        if args.limit is not None:
            questions = questions[: args.limit]
        vector_backends = [backend for backend in args.backends if _normalize_backend(backend) != "lexical"]
        prepared_embeddings: PreparedEmbeddings | None = None
        embedding_error: Exception | None = None
        if vector_backends:
            print(f"# embedding phase dataset={dataset} records={len(records)}", flush=True)
            try:
                prepared_embeddings = _prepare_embeddings(dataset, records, args, config)
            except (BackendUnavailable, ImportError, ValueError, LLMClientError, FileNotFoundError) as exc:
                embedding_error = exc
                print(f"# embedding phase failed dataset={dataset}: {exc}", flush=True)
        if args.embedding_only:
            continue

        for backend in args.backends:
            print(f"# rag phase dataset={dataset} backend={backend} questions={len(questions)}", flush=True)
            row = _run_backend(
                dataset,
                backend,
                records,
                questions,
                args,
                llm_client,
                prepared_embeddings=prepared_embeddings,
                embedding_error=embedding_error,
            )
            rows.append(row)
            _print_row(row)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps([asdict(row) for row in rows], indent=2), encoding="utf-8")


def _run_backend(
    dataset: str,
    backend: str,
    records: list[DocumentChunk],
    questions: list[dict],
    args: argparse.Namespace,
    llm_client: OpenAICompatibleClient | None,
    prepared_embeddings: PreparedEmbeddings | None,
    embedding_error: Exception | None,
) -> BenchmarkRow:
    try:
        retriever = _build_retriever(backend, records, args=args, prepared_embeddings=prepared_embeddings, embedding_error=embedding_error)
    except (BackendUnavailable, ImportError, ValueError, LLMClientError, FileNotFoundError) as exc:
        return BenchmarkRow(
            dataset=dataset,
            backend=backend,
            questions=0,
            sufficient_rate=0.0,
            evidence_recall=None,
            avg_rounds=0.0,
            avg_latency_ms=0.0,
            status="skipped",
            error=str(exc),
        )

    sufficient_count = 0
    evidence_scores = []
    rounds = []
    latencies = []
    llm_agents = build_llm_agents(llm_client) if llm_client else {}
    pipeline = AgenticRAGPipeline(retriever, max_rounds=args.max_rounds, top_k=args.top_k, **llm_agents)

    try:
        for item in questions:
            started = time.perf_counter()
            trace = pipeline.run(item["question"])
            latencies.append((time.perf_counter() - started) * 1000)
            rounds.append(len(trace.queries_by_round))
            if trace.assessments[-1].status == ContextStatus.SUFFICIENT:
                sufficient_count += 1
            evidence = _normalize_evidence(item.get("evidence"))
            if evidence:
                retrieved_text = "\n".join(result.record.text.lower() for result in _all_retrieved(trace))
                hits = sum(1 for claim in evidence if claim.lower() in retrieved_text)
                evidence_scores.append(hits / len(evidence))
    except Exception as exc:
        return BenchmarkRow(
            dataset=dataset,
            backend=backend,
            questions=len(latencies),
            sufficient_rate=sufficient_count / len(latencies) if latencies else 0.0,
            evidence_recall=statistics.mean(evidence_scores) if evidence_scores else None,
            avg_rounds=statistics.mean(rounds) if rounds else 0.0,
            avg_latency_ms=statistics.mean(latencies) if latencies else 0.0,
            status="failed",
            error=str(exc),
        )

    return BenchmarkRow(
        dataset=dataset,
        backend=backend,
        questions=len(questions),
        sufficient_rate=sufficient_count / len(questions) if questions else 0.0,
        evidence_recall=statistics.mean(evidence_scores) if evidence_scores else None,
        avg_rounds=statistics.mean(rounds) if rounds else 0.0,
        avg_latency_ms=statistics.mean(latencies) if latencies else 0.0,
    )


def _build_retriever(
    backend: str,
    records: list[DocumentChunk],
    args: argparse.Namespace,
    prepared_embeddings: PreparedEmbeddings | None,
    embedding_error: Exception | None,
):
    normalized = _normalize_backend(backend)
    if normalized == "lexical":
        return LexicalRetriever(records)
    if embedding_error is not None:
        raise embedding_error
    if prepared_embeddings is None:
        raise ValueError("Vector backend requested without prepared embeddings.")
    index = build_vector_index(normalized, dim=prepared_embeddings.dim, bit_width=args.bit_width)
    return VectorRetriever(
        records=records,
        embedder=prepared_embeddings.embedder,
        index=index,
        record_vectors=prepared_embeddings.record_vectors,
    )


def _prepare_embeddings(dataset: str, records: list[DocumentChunk], args: argparse.Namespace, config: dict) -> PreparedEmbeddings:
    embedder = _build_embedder(args, config)
    dim = _embedding_dim(embedder)
    dataset_cache_dir = args.embedding_cache_dir / _safe_dataset_cache_name(dataset)
    record_vectors = load_or_compute_embeddings(
        records=records,
        embedder=embedder,
        embedder_config=_embedder_cache_config(args, config),
        cache_dir=dataset_cache_dir,
        require_cached=args.rag_only,
    )
    return PreparedEmbeddings(embedder=embedder, dim=dim, record_vectors=record_vectors)


def _build_embedder(args: argparse.Namespace, config: dict):
    embedding_config = config.get("embeddings", {}) if isinstance(config.get("embeddings", {}), dict) else {}
    provider = args.embedding_provider or embedding_config.get("provider", "local")
    if provider == "hashing":
        return HashingEmbedder(dim=args.dim)
    if provider != "local":
        raise ValueError(f"unknown embedding provider: {provider}")

    model_path = args.embedding_model_path or embedding_config.get("model_path")
    batch_size = args.embedding_batch_size or int(embedding_config.get("batch_size", 32))
    if not model_path:
        raise ValueError("Local embeddings require embeddings.model_path in config.")
    return LocalSentenceTransformerEmbedder(model_path=model_path, batch_size=batch_size)


def _normalize_backend(backend: str) -> str:
    return backend.lower().replace("_", "-")


def _safe_dataset_cache_name(dataset: str) -> str:
    return dataset.replace("/", "__").replace("\\", "__")


def _embedding_dim(embedder) -> int:
    if isinstance(embedder, HashingEmbedder):
        return embedder.dim
    probe = embedder.encode_one("dimension probe")
    if not probe:
        raise ValueError("embedding model returned an empty vector")
    return len(probe)


def _embedder_cache_config(args: argparse.Namespace, config: dict) -> dict:
    embedding_config = config.get("embeddings", {}) if isinstance(config.get("embeddings", {}), dict) else {}
    provider = args.embedding_provider or embedding_config.get("provider", "local")
    if provider == "hashing":
        return {
            "provider": "hashing",
            "dim": args.dim,
        }
    return {
        "provider": "local",
        "model_path": str(args.embedding_model_path or embedding_config.get("model_path")),
        "batch_size": args.embedding_batch_size or int(embedding_config.get("batch_size", 32)),
    }


def _build_llm_client(args: argparse.Namespace, config: dict) -> OpenAICompatibleClient | AnthropicMessagesClient | None:
    llm_config = config.get("llm", {}) if isinstance(config.get("llm", {}), dict) else {}
    base_url = args.llm_base_url or llm_config.get("base_url")
    model = args.llm_model or llm_config.get("model")
    api_key = args.llm_api_key or llm_config.get("api_key") or "no_need"
    timeout = args.llm_timeout if args.llm_timeout is not None else float(llm_config.get("timeout", 60.0))
    temperature = (
        args.llm_temperature if args.llm_temperature is not None else float(llm_config.get("temperature", 0.0))
    )
    provider = args.llm_provider or llm_config.get("provider", "openai")

    if not base_url:
        return None

    if provider == "anthropic":
        client = AnthropicMessagesClient(
            base_url=base_url,
            model=model,
            api_key=api_key,
            timeout=timeout,
            temperature=temperature,
        )
    else:
        client = OpenAICompatibleClient(
            base_url=base_url,
            model=model,
            api_key=api_key,
            timeout=timeout,
            temperature=temperature,
        )
    try:
        resolved = client.resolve_model()
    except LLMClientError as exc:
        raise SystemExit(f"Could not initialize LLM at {base_url}: {exc}") from exc
    print(f"# using LLM: {base_url} model={resolved} provider={provider}")
    return client


def _load_records(path: Path, source: str) -> list[DocumentChunk]:
    raw_chunks = json.loads(path.read_text(encoding="utf-8"))
    records = []
    for raw_chunk in raw_chunks:
        record_id, text = raw_chunk.split(":", 1)
        records.append(DocumentChunk(record_id=record_id.strip(), source=source, text=text.strip()))
    return records


def _load_questions(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def _normalize_evidence(raw_evidence) -> list[str]:
    if isinstance(raw_evidence, str):
        return [raw_evidence] if raw_evidence.strip() else []
    if isinstance(raw_evidence, list):
        return [str(item) for item in raw_evidence if str(item).strip()]
    return []


def _all_retrieved(trace) -> list:
    seen = {}
    for round_results in trace.retrieved_by_round:
        for result in round_results:
            seen[result.record.record_id] = result
    return list(seen.values())


def _print_table(rows: list[BenchmarkRow]) -> None:
    _print_header()
    for row in rows:
        _print_row(row)


def _print_header() -> None:
    headers = [
        "dataset",
        "backend",
        "status",
        "questions",
        "sufficient",
        "evidence",
        "rounds",
        "latency_ms",
    ]
    print("\t".join(headers), flush=True)


def _print_row(row: BenchmarkRow) -> None:
    evidence = "n/a" if row.evidence_recall is None else f"{row.evidence_recall:.3f}"
    print(
        "\t".join(
            [
                row.dataset,
                row.backend,
                row.status,
                str(row.questions),
                f"{row.sufficient_rate:.3f}",
                evidence,
                f"{row.avg_rounds:.2f}",
                f"{row.avg_latency_ms:.2f}",
            ]
        ),
        flush=True,
    )
    if row.error:
        print(f"# {row.backend} skipped: {row.error}", flush=True)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--data-dir", type=Path, default=Path("datasets"))
    parser.add_argument("--datasets", nargs="+", default=["medical", "hotpotqa", "musique", "2wikimultihop", "novel"])
    parser.add_argument("--backends", nargs="+", default=["faiss-flat", "faiss-pq", "turbovec"])
    parser.add_argument("--limit", type=int, default=None, help="Questions per dataset. Defaults to all questions.")
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--dim", type=int, default=384)
    parser.add_argument("--bit-width", type=int, default=4, choices=[2, 4])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--llm-base-url", default=None, help="Override config llm.base_url.")
    parser.add_argument("--llm-model", default=None, help="Override config llm.model.")
    parser.add_argument("--llm-api-key", default=None, help="Override config llm.api_key.")
    parser.add_argument("--llm-timeout", type=float, default=None, help="Override config llm.timeout.")
    parser.add_argument("--llm-temperature", type=float, default=None, help="Override config llm.temperature.")
    parser.add_argument("--llm-provider", default=None, choices=["openai", "anthropic"], help="LLM API provider: openai (default) or anthropic.")
    parser.add_argument("--embedding-provider", default=None, help="Override config embeddings.provider: local or hashing.")
    parser.add_argument("--embedding-model-path", default=None, help="Override config embeddings.model_path.")
    parser.add_argument("--embedding-batch-size", type=int, default=None, help="Override config embeddings.batch_size.")
    parser.add_argument("--embedding-cache-dir", type=Path, default=Path(".cache/embeddings"))
    parser.add_argument("--embedding-only", action="store_true", help="Only compute/load dataset chunk embedding caches; skip RAG.")
    parser.add_argument("--rag-only", action="store_true", help="Require existing chunk embedding caches; never compute missing corpus embeddings.")
    return parser.parse_args(argv)


if __name__ == "__main__":
    main()
