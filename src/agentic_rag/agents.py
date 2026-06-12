"""Agents that implement orchestration, context checking, and synthesis."""

from __future__ import annotations

from collections import defaultdict
import re

from agentic_rag.types import ContextAssessment, ContextStatus, RetrievalTask, SearchResult


WORD_RE = re.compile(r"[a-zA-Z0-9]+")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "patient",
    "please",
    "request",
    "show",
    "summarize",
    "summary",
    "tell",
    "the",
    "this",
    "to",
    "with",
}


class RootAgent:
    """Parse a request into generic information needs."""

    def parse(self, request: str) -> list[RetrievalTask]:
        clauses = _extract_task_clauses(request)
        return [_task_from_clause(index, clause) for index, clause in enumerate(clauses, start=1)]


class PlannerAgent:
    """Declare the work areas to inspect."""

    def plan(self, tasks: list[RetrievalTask]) -> list[RetrievalTask]:
        return tasks


class QueryRewriter:
    """Rewrite broad tasks into retriever-friendly queries."""

    def initial_queries(self, tasks: list[RetrievalTask]) -> list[str]:
        return [task.search_hints[0] for task in tasks]

    def follow_up_queries(self, assessment: ContextAssessment, tasks: list[RetrievalTask]) -> list[str]:
        task_by_name = {task.name: task for task in tasks}
        queries = []
        for missing_task in assessment.missing_tasks:
            task = task_by_name[missing_task]
            queries.extend(task.search_hints)
            queries.extend(assessment.feedback)
        return list(dict.fromkeys(queries))


class RAGAgent:
    """Fan out searches through the retriever."""

    def __init__(self, retriever):
        self.retriever = retriever

    def search(self, queries: list[str], top_k: int = 4) -> list[SearchResult]:
        return self.retriever.search(queries, top_k=top_k)


class DraftAgent:
    """Create an intermediate rough answer for quality control."""

    def draft(self, tasks: list[RetrievalTask], snippets: list[SearchResult]) -> str:
        grouped = _group_snippets_by_task(tasks, snippets)
        lines = []
        for task in tasks:
            task_snippets = grouped.get(task.name, [])
            if not task_snippets:
                lines.append(f"{task.description}: not found in retrieved snippets.")
                continue
            evidence = "; ".join(_shorten(result.record.text) for result in task_snippets[:2])
            lines.append(f"{task.description}: {evidence}")
        return "\n".join(lines)


class SufficientContextAgent:
    """Inspect snippets, draft, and missing pieces before synthesis."""

    def assess(self, request: str, tasks: list[RetrievalTask], snippets: list[SearchResult], draft: str) -> ContextAssessment:
        del request, draft
        grouped = _group_snippets_by_task(tasks, snippets)
        covered = [task.name for task in tasks if grouped.get(task.name)]
        missing = [task.name for task in tasks if not grouped.get(task.name)]

        finding_parts = [
            f"{task.name}: found {len(grouped[task.name])} supporting snippet(s)"
            for task in tasks
            if grouped.get(task.name)
        ]
        gap_parts = [f"{task.name}: no source snippet matched the required concepts" for task in tasks if task.name in missing]

        if not missing:
            return ContextAssessment(
                status=ContextStatus.SUFFICIENT,
                finding="; ".join(finding_parts),
                gap="No missing information need detected.",
                covered_tasks=covered,
                missing_tasks=[],
            )

        feedback = []
        for task in tasks:
            if task.name in missing:
                feedback.extend(task.search_hints)
                feedback.extend(task.required_terms)

        return ContextAssessment(
            status=ContextStatus.INSUFFICIENT,
            finding="; ".join(finding_parts) if finding_parts else "No required information need was grounded.",
            gap="; ".join(gap_parts),
            feedback=list(dict.fromkeys(feedback)),
            covered_tasks=covered,
            missing_tasks=missing,
        )


class SynthesisAgent:
    """Write the final answer once context is sufficient."""

    def synthesize(self, tasks: list[RetrievalTask], snippets: list[SearchResult], assessment: ContextAssessment) -> str:
        grouped = _group_snippets_by_task(tasks, snippets)
        lines = ["Grounded summary:"]
        for task in tasks:
            evidence = grouped.get(task.name, [])
            if not evidence:
                lines.append(f"- {task.description}: not found in the available records.")
                continue
            best = evidence[0]
            lines.append(
                f"- {task.description}: {_shorten(best.record.text, limit=260)} "
                f"[source: {best.record.source}#{best.record.record_id}]"
            )
        lines.append(f"Sufficient-context finding: {assessment.finding}")
        return "\n".join(lines)


def _group_snippets_by_task(tasks: list[RetrievalTask], snippets: list[SearchResult]) -> dict[str, list[SearchResult]]:
    grouped: dict[str, list[SearchResult]] = defaultdict(list)
    for result in snippets:
        text = result.record.text.lower()
        query = result.query.lower()
        for task in tasks:
            if any(term in text or term in query for term in task.required_terms):
                grouped[task.name].append(result)
    return grouped


def _extract_task_clauses(request: str) -> list[str]:
    text = request.lower()
    text = re.sub(r"\b(summarize|show|tell me|list|find|retrieve|answer|explain)\b", " ", text)
    text = re.sub(r"\b(this|the|a|an)?\s*patient'?s?\b", " ", text)
    text = re.sub(r"[?.!]", " ", text)
    parts = [part.strip(" :;,-") for part in re.split(r",|\band\b", text)]
    clauses = [part for part in parts if _content_terms(part)]
    return clauses or [request.strip()]


def _task_from_clause(index: int, clause: str) -> RetrievalTask:
    description = " ".join(clause.split())
    alternatives = [part.strip() for part in re.split(r"\bor\b|/", description) if _content_terms(part)]
    primary = alternatives[0] if alternatives else description
    required_terms = tuple(dict.fromkeys(term for phrase in [description, *alternatives] for term in _content_terms(phrase)))
    search_hints = tuple(dict.fromkeys([primary, description, *alternatives, *required_terms]))
    return RetrievalTask(
        name=f"task_{index}",
        description=description,
        required_terms=required_terms or tuple(_content_terms(description)),
        search_hints=search_hints or (description,),
    )


def _content_terms(text: str) -> list[str]:
    return [term for term in WORD_RE.findall(text.lower()) if term not in STOPWORDS and len(term) > 1]


def _shorten(text: str, limit: int = 180) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3].rstrip() + "..."
