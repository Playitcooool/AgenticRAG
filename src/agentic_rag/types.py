"""Shared domain types for the agentic RAG pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class ContextStatus(str, Enum):
    """Decision made by the sufficient-context agent."""

    SUFFICIENT = "sufficient"
    INSUFFICIENT = "insufficient"


@dataclass(frozen=True)
class PatientRecord:
    """A source document chunk from a patient record."""

    record_id: str
    source: str
    text: str


@dataclass(frozen=True)
class SearchResult:
    """A retrieved snippet with score and provenance."""

    query: str
    record: PatientRecord
    score: float


@dataclass(frozen=True)
class ClinicalTask:
    """A clinical information need extracted from the doctor request."""

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
    tasks: list[ClinicalTask]
    queries_by_round: list[list[str]]
    retrieved_by_round: list[list[SearchResult]]
    drafts: list[str]
    assessments: list[ContextAssessment]
    final_answer: str
