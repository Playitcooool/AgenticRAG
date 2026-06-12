"""Agents that implement orchestration, context checking, and synthesis."""

from __future__ import annotations

from collections import defaultdict

from agentic_rag.types import ClinicalTask, ContextAssessment, ContextStatus, SearchResult


class RootAgent:
    """Parse a doctor's request and delegate clinical work areas."""

    def parse(self, request: str) -> list[ClinicalTask]:
        request_lower = request.lower()
        tasks = []
        if any(term in request_lower for term in ("med", "drug", "pharmacy")):
            tasks.append(
                ClinicalTask(
                    name="pharmacy",
                    description="Current and discharge medications",
                    required_terms=("medication", "medications", "drug", "pharmacy", "lisinopril", "metformin"),
                    search_hints=("discharge medications", "pharmacy reconciliation", "medication list"),
                )
            )
        if any(term in request_lower for term in ("diet", "nutrition", "sodium", "food")):
            tasks.append(
                ClinicalTask(
                    name="nutrition",
                    description="Diet and nutrition instructions",
                    required_terms=("diet", "nutrition", "sodium", "fluid", "meal"),
                    search_hints=("nutrition notes", "diet instructions", "low sodium diet"),
                )
            )
        if any(term in request_lower for term in ("allerg", "reaction", "rash", "adverse")):
            tasks.append(
                ClinicalTask(
                    name="allergies",
                    description="Allergies, reactions, or adverse events",
                    required_terms=("allergy", "allergies", "rash", "reaction", "adverse"),
                    search_hints=("allergy history", "adverse reaction", "rash during stay"),
                )
            )
        if not tasks:
            tasks.append(
                ClinicalTask(
                    name="general",
                    description="General clinical context",
                    required_terms=tuple(request_lower.split()),
                    search_hints=(request,),
                )
            )
        return tasks


class PlannerAgent:
    """Declare the work areas to inspect."""

    def plan(self, tasks: list[ClinicalTask]) -> list[ClinicalTask]:
        return tasks


class QueryRewriter:
    """Rewrite broad clinical tasks into retriever-friendly queries."""

    def initial_queries(self, tasks: list[ClinicalTask]) -> list[str]:
        return [task.search_hints[0] for task in tasks]

    def follow_up_queries(self, assessment: ContextAssessment, tasks: list[ClinicalTask]) -> list[str]:
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

    def draft(self, tasks: list[ClinicalTask], snippets: list[SearchResult]) -> str:
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

    def assess(self, request: str, tasks: list[ClinicalTask], snippets: list[SearchResult], draft: str) -> ContextAssessment:
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
                gap="No missing clinical area detected.",
                covered_tasks=covered,
                missing_tasks=[],
            )

        feedback = []
        for task in tasks:
            if task.name in missing:
                feedback.extend(task.search_hints)
                if task.name == "allergies":
                    feedback.extend(("rash", "adverse event", "allergic reaction", "new reaction during stay"))

        return ContextAssessment(
            status=ContextStatus.INSUFFICIENT,
            finding="; ".join(finding_parts) if finding_parts else "No required clinical area was grounded.",
            gap="; ".join(gap_parts),
            feedback=list(dict.fromkeys(feedback)),
            covered_tasks=covered,
            missing_tasks=missing,
        )


class SynthesisAgent:
    """Write the final answer once context is sufficient."""

    def synthesize(self, tasks: list[ClinicalTask], snippets: list[SearchResult], assessment: ContextAssessment) -> str:
        grouped = _group_snippets_by_task(tasks, snippets)
        lines = ["Clinical summary for the doctor:"]
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


def _group_snippets_by_task(tasks: list[ClinicalTask], snippets: list[SearchResult]) -> dict[str, list[SearchResult]]:
    grouped: dict[str, list[SearchResult]] = defaultdict(list)
    for result in snippets:
        text = result.record.text.lower()
        query = result.query.lower()
        for task in tasks:
            if any(term in text or term in query for term in task.required_terms):
                grouped[task.name].append(result)
    return grouped


def _shorten(text: str, limit: int = 180) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3].rstrip() + "..."
