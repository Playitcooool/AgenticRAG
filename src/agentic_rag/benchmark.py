"""Benchmark FAISS and turbovec inside the agentic RAG architecture."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from agentic_rag.config import load_config
from agentic_rag.embeddings import HashingEmbedder, LocalSentenceTransformerEmbedder
from agentic_rag.llm import LLMClientError, OpenAICompatibleClient
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


def main() -> None:
    args = _parse_args()
    config = load_config(args.config)
    llm_client = _build_llm_client(args, config)
    rows = []
    _print_header()
    for dataset in args.datasets:
        dataset_path = args.data_dir / dataset
        records = _load_records(dataset_path / "chunks.json", source=dataset)
        questions = _load_questions(dataset_path / "questions.json")[: args.limit]
        for backend in args.backends:
            print(f"# starting dataset={dataset} backend={backend} questions={len(questions)}", flush=True)
            row = _run_backend(dataset, backend, records, questions, args, config, llm_client)
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
    config: dict,
    llm_client: OpenAICompatibleClient | None,
) -> BenchmarkRow:
    try:
        retriever = _build_retriever(backend, records, args=args, config=config)
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

    return BenchmarkRow(
        dataset=dataset,
        backend=backend,
        questions=len(questions),
        sufficient_rate=sufficient_count / len(questions) if questions else 0.0,
        evidence_recall=statistics.mean(evidence_scores) if evidence_scores else None,
        avg_rounds=statistics.mean(rounds) if rounds else 0.0,
        avg_latency_ms=statistics.mean(latencies) if latencies else 0.0,
    )


def _build_retriever(backend: str, records: list[DocumentChunk], args: argparse.Namespace, config: dict):
    normalized = backend.lower().replace("_", "-")
    if normalized == "lexical":
        return LexicalRetriever(records)
    embedder = _build_embedder(args, config)
    dim = _embedding_dim(embedder)
    index = build_vector_index(normalized, dim=dim, bit_width=args.bit_width)
    return VectorRetriever(records, embedder, index)


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


def _embedding_dim(embedder) -> int:
    if isinstance(embedder, HashingEmbedder):
        return embedder.dim
    probe = embedder.encode_one("dimension probe")
    if not probe:
        raise ValueError("embedding model returned an empty vector")
    return len(probe)


def _build_llm_client(args: argparse.Namespace, config: dict) -> OpenAICompatibleClient | None:
    llm_config = config.get("llm", {}) if isinstance(config.get("llm", {}), dict) else {}
    base_url = args.llm_base_url or llm_config.get("base_url")
    model = args.llm_model or llm_config.get("model")
    api_key = args.llm_api_key or llm_config.get("api_key") or "no_need"
    timeout = args.llm_timeout if args.llm_timeout is not None else float(llm_config.get("timeout", 60.0))
    temperature = (
        args.llm_temperature if args.llm_temperature is not None else float(llm_config.get("temperature", 0.0))
    )

    if not base_url:
        return None
    client = OpenAICompatibleClient(
        base_url=base_url,
        model=model,
        api_key=api_key,
        timeout=timeout,
        temperature=temperature,
    )
    try:
        model = client.resolve_model()
    except LLMClientError as exc:
        raise SystemExit(f"Could not initialize local LLM at {base_url}: {exc}") from exc
    print(f"# using local LLM: {base_url} model={model}")
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--data-dir", type=Path, default=Path("datasets"))
    parser.add_argument("--datasets", nargs="+", default=["medical", "hotpotqa", "musique", "2wikimultihop", "novel"])
    parser.add_argument("--backends", nargs="+", default=["faiss-flat", "faiss-pq", "turbovec"])
    parser.add_argument("--limit", type=int, default=50, help="Questions per dataset.")
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
    parser.add_argument("--embedding-provider", default=None, help="Override config embeddings.provider: local or hashing.")
    parser.add_argument("--embedding-model-path", default=None, help="Override config embeddings.model_path.")
    parser.add_argument("--embedding-batch-size", type=int, default=None, help="Override config embeddings.batch_size.")
    return parser.parse_args()


if __name__ == "__main__":
    main()
