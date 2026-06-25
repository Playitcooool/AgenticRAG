"""LLM-backed agents for the agentic RAG pipeline."""

from __future__ import annotations

import json

from agentic_rag.agents import RootAgent, SufficientContextAgent, SynthesisAgent, _content_terms
from agentic_rag.llm import LLMClient, LLMClientError
from agentic_rag.types import ContextAssessment, ContextStatus, RetrievalTask, SearchResult


class LLMRootAgent(RootAgent):
    """Parse information needs with a local OpenAI-compatible model."""

    def __init__(self, client: LLMClient):
        self.client = client
        self.fallback = RootAgent()

    def parse(self, request: str) -> list[RetrievalTask]:
        system = (
            "You split user questions into retrieval tasks for an agentic RAG system. "
            "Return strict JSON only."
        )
        user = f"""
Request:
{request}

Return JSON with this shape:
{{
  "tasks": [
    {{
      "description": "short information need",
      "required_terms": ["keywords that should appear in supporting evidence"],
      "search_hints": ["search query", "alternate search query"]
    }}
  ]
}}
Rules:
- Keep 1 to 5 tasks.
- Use concise search queries.
- required_terms should be lowercase lexical terms or short phrases.
"""
        try:
            payload = self.client.chat_json(system, user)
            tasks = payload.get("tasks", [])
            parsed = [_task_from_payload(index, item) for index, item in enumerate(tasks, start=1)]
            return parsed or self.fallback.parse(request)
        except (LLMClientError, TypeError, ValueError):
            return self.fallback.parse(request)


class LLMSufficientContextAgent(SufficientContextAgent):
    """Let the model judge whether retrieved snippets answer every task."""

    def __init__(self, client: LLMClient):
        self.client = client
        self.fallback = SufficientContextAgent()

    def assess(self, request: str, tasks: list[RetrievalTask], snippets: list[SearchResult], draft: str) -> ContextAssessment:
        system = (
            "You are the sufficient-context quality-control agent in an agentic RAG system. "
            "Judge only whether the snippets contain enough source evidence. Return strict JSON only."
        )
        user = json.dumps(
            {
                "request": request,
                "tasks": [
                    {
                        "name": task.name,
                        "description": task.description,
                        "required_terms": list(task.required_terms),
                    }
                    for task in tasks
                ],
                "draft": draft,
                "snippets": [
                    {
                        "record_id": result.record.record_id,
                        "source": result.record.source,
                        "query": result.query,
                        "text": result.record.text[:1200],
                    }
                    for result in snippets[:12]
                ],
                "required_output": {
                    "status": "sufficient or insufficient",
                    "finding": "what is grounded",
                    "gap": "what is missing",
                    "covered_tasks": ["task names"],
                    "missing_tasks": ["task names"],
                    "feedback": ["follow-up search queries for missing tasks"],
                },
            },
            indent=2,
        )
        try:
            payload = self.client.chat_json(system, user)
            missing = [name for name in payload.get("missing_tasks", []) if isinstance(name, str)]
            covered = [name for name in payload.get("covered_tasks", []) if isinstance(name, str)]
            status = ContextStatus.SUFFICIENT if str(payload.get("status", "")).lower() == "sufficient" else ContextStatus.INSUFFICIENT
            if missing:
                status = ContextStatus.INSUFFICIENT
            return ContextAssessment(
                status=status,
                finding=str(payload.get("finding", "")),
                gap=str(payload.get("gap", "")),
                feedback=[str(item) for item in payload.get("feedback", [])],
                covered_tasks=covered,
                missing_tasks=missing,
            )
        except (LLMClientError, TypeError, ValueError):
            return self.fallback.assess(request, tasks, snippets, draft)


class LLMSynthesisAgent(SynthesisAgent):
    """Generate the final grounded answer with a local model."""

    def __init__(self, client: LLMClient):
        self.client = client
        self.fallback = SynthesisAgent()

    def synthesize(self, tasks: list[RetrievalTask], snippets: list[SearchResult], assessment: ContextAssessment) -> str:
        system = (
            "You write concise grounded answers for an agentic RAG system. "
            "Use only the provided snippets and cite source#record_id inline."
        )
        user = json.dumps(
            {
                "tasks": [
                    {
                        "name": task.name,
                        "description": task.description,
                    }
                    for task in tasks
                ],
                "assessment": {
                    "status": assessment.status.value,
                    "finding": assessment.finding,
                    "gap": assessment.gap,
                },
                "snippets": [
                    {
                        "record_id": result.record.record_id,
                        "source": result.record.source,
                        "text": result.record.text[:1200],
                    }
                    for result in snippets[:12]
                ],
            },
            indent=2,
        )
        try:
            return self.client.chat(system, user)
        except LLMClientError:
            return self.fallback.synthesize(tasks, snippets, assessment)


def build_llm_agents(client: LLMClient) -> dict[str, object]:
    return {
        "root_agent": LLMRootAgent(client),
        "sufficient_context_agent": LLMSufficientContextAgent(client),
        "synthesis_agent": LLMSynthesisAgent(client),
    }


def _task_from_payload(index: int, item: dict) -> RetrievalTask:
    description = str(item.get("description", "")).strip()
    if not description:
        raise ValueError("task description is required")
    raw_terms = item.get("required_terms") or _content_terms(description)
    raw_hints = item.get("search_hints") or [description]
    required_terms = tuple(dict.fromkeys(str(term).lower().strip() for term in raw_terms if str(term).strip()))
    search_hints = tuple(dict.fromkeys(str(hint).strip() for hint in raw_hints if str(hint).strip()))
    return RetrievalTask(
        name=f"task_{index}",
        description=description,
        required_terms=required_terms or tuple(_content_terms(description)),
        search_hints=search_hints or (description,),
    )
