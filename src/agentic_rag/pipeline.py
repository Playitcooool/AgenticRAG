"""End-to-end five-phase agentic RAG pipeline."""

from __future__ import annotations

from agentic_rag.agents import (
    DraftAgent,
    PlannerAgent,
    QueryRewriter,
    RAGAgent,
    RootAgent,
    SufficientContextAgent,
    SynthesisAgent,
)
from agentic_rag.types import ContextStatus, PipelineTrace, SearchResult


class AgenticRAGPipeline:
    """Coordinate orchestration, retrieval, context checking, iteration, and synthesis."""

    def __init__(self, retriever, max_rounds: int = 3, top_k: int = 4):
        self.root_agent = RootAgent()
        self.planner_agent = PlannerAgent()
        self.query_rewriter = QueryRewriter()
        self.rag_agent = RAGAgent(retriever)
        self.draft_agent = DraftAgent()
        self.sufficient_context_agent = SufficientContextAgent()
        self.synthesis_agent = SynthesisAgent()
        self.max_rounds = max_rounds
        self.top_k = top_k

    def run(self, request: str) -> PipelineTrace:
        tasks = self.planner_agent.plan(self.root_agent.parse(request))
        queries = self.query_rewriter.initial_queries(tasks)
        all_results: list[SearchResult] = []
        queries_by_round = []
        retrieved_by_round = []
        drafts = []
        assessments = []

        for _round in range(self.max_rounds):
            queries_by_round.append(queries)
            round_results = self.rag_agent.search(queries, top_k=self.top_k)
            retrieved_by_round.append(round_results)
            all_results = _merge_results(all_results, round_results)

            draft = self.draft_agent.draft(tasks, all_results)
            drafts.append(draft)

            assessment = self.sufficient_context_agent.assess(request, tasks, all_results, draft)
            assessments.append(assessment)
            if assessment.status == ContextStatus.SUFFICIENT:
                break

            queries = self.query_rewriter.follow_up_queries(assessment, tasks)
            if not queries:
                break

        final_answer = self.synthesis_agent.synthesize(tasks, all_results, assessments[-1])
        return PipelineTrace(
            request=request,
            tasks=tasks,
            queries_by_round=queries_by_round,
            retrieved_by_round=retrieved_by_round,
            drafts=drafts,
            assessments=assessments,
            final_answer=final_answer,
        )


def _merge_results(existing: list[SearchResult], new_results: list[SearchResult]) -> list[SearchResult]:
    by_id = {result.record.record_id: result for result in existing}
    for result in new_results:
        current = by_id.get(result.record.record_id)
        if current is None or result.score > current.score:
            by_id[result.record.record_id] = result
    return sorted(by_id.values(), key=lambda result: result.score, reverse=True)
