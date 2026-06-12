"""Shared types for the agentic RAG pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class ContextStatus(str, Enum):
    """Decision made by the sufficient-context agent."""

    SUFFICIENT = "sufficient"
    INSUFFICIENT = "insufficient"


@dataclass(frozen=True)
class DocumentChunk:
    """A source document chunk."""

    record_id: str
    source: str
    text: str


@dataclass(frozen=True)
class SearchResult:
    """A retrieved snippet with score and provenance."""

    query: str
    record: DocumentChunk
    score: float


@dataclass(frozen=True)
class RetrievalTask:
    """An information need extracted from the user request."""

    name: str
    description: str
    required_terms: tuple[str, ...]
    search_hints: tuple[str, ...]


@dataclass
class ContextAssessment:
    """Quality-control output from the sufficient-context agent."""

    status: ContextStatus
    finding: str
    gap: str
    feedback: list[str] = field(default_factory=list)
    covered_tasks: list[str] = field(default_factory=list)
    missing_tasks: list[str] = field(default_factory=list)


@dataclass
class PipelineTrace:
    """Inspectable trace for the full five-phase workflow."""

    request: str
    tasks: list[RetrievalTask]
    queries_by_round: list[list[str]]
    retrieved_by_round: list[list[SearchResult]]
    drafts: list[str]
    assessments: list[ContextAssessment]
    final_answer: str
