"""Benchmark FAISS and turbovec inside the agentic RAG architecture."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from agentic_rag.embeddings import HashingEmbedder
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
    rows = []
    for dataset in args.datasets:
        dataset_path = args.data_dir / dataset
        records = _load_records(dataset_path / "chunks.json", source=dataset)
        questions = _load_questions(dataset_path / "questions.json")[: args.limit]
        for backend in args.backends:
            rows.append(_run_backend(dataset, backend, records, questions, args))

    _print_table(rows)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps([asdict(row) for row in rows], indent=2), encoding="utf-8")


def _run_backend(
    dataset: str,
    backend: str,
    records: list[DocumentChunk],
    questions: list[dict],
    args: argparse.Namespace,
) -> BenchmarkRow:
    try:
        retriever = _build_retriever(backend, records, dim=args.dim, bit_width=args.bit_width)
    except (BackendUnavailable, ImportError, ValueError) as exc:
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
    pipeline = AgenticRAGPipeline(retriever, max_rounds=args.max_rounds, top_k=args.top_k)

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


def _build_retriever(backend: str, records: list[DocumentChunk], dim: int, bit_width: int):
    normalized = backend.lower().replace("_", "-")
    if normalized == "lexical":
        return LexicalRetriever(records)
    embedder = HashingEmbedder(dim=dim)
    index = build_vector_index(normalized, dim=dim, bit_width=bit_width)
    return VectorRetriever(records, embedder, index)


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
    print("\t".join(headers))
    for row in rows:
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
            )
        )
        if row.error:
            print(f"# {row.backend} skipped: {row.error}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("datasets"))
    parser.add_argument("--datasets", nargs="+", default=["medical", "hotpotqa", "musique", "2wikimultihop", "novel"])
    parser.add_argument("--backends", nargs="+", default=["faiss-flat", "faiss-pq", "turbovec"])
    parser.add_argument("--limit", type=int, default=50, help="Questions per dataset.")
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--dim", type=int, default=384)
    parser.add_argument("--bit-width", type=int, default=4, choices=[2, 4])
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    main()
