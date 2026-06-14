from __future__ import annotations

import unittest
from pathlib import Path

from agentic_rag.agents import SufficientContextAgent
from agentic_rag.benchmark import _parse_args
from agentic_rag.demo import build_demo_retriever
from agentic_rag.pipeline import AgenticRAGPipeline
from agentic_rag.retriever import LexicalRetriever
from agentic_rag.types import ContextStatus, DocumentChunk, RetrievalTask


class LexicalRetrieverTest(unittest.TestCase):
    def test_searches_all_query_fanouts_and_deduplicates_records(self) -> None:
        retriever = LexicalRetriever(
            [
                DocumentChunk("1", "Pharmacy", "Discharge medications include metformin."),
                DocumentChunk("2", "Nutrition", "Low sodium diet recommended."),
                DocumentChunk("3", "Other", "Follow up in clinic."),
            ]
        )

        results = retriever.search(["discharge medications", "low sodium diet"], top_k=2)

        self.assertEqual({"1", "2"}, {result.record.record_id for result in results})


class SufficientContextAgentTest(unittest.TestCase):
    def test_flags_missing_required_information_need(self) -> None:
        tasks = [
            RetrievalTask("task_1", "release date", ("release", "date"), ("release date",)),
            RetrievalTask("task_2", "runtime", ("runtime",), ("runtime",)),
        ]
        retriever = LexicalRetriever(
            [DocumentChunk("1", "Product Brief", "The product release date is July 12.")]
        )
        snippets = retriever.search(["release date"], top_k=1)

        assessment = SufficientContextAgent().assess("release date and runtime", tasks, snippets, "draft")

        self.assertEqual(ContextStatus.INSUFFICIENT, assessment.status)
        self.assertEqual(["task_1"], assessment.covered_tasks)
        self.assertEqual(["task_2"], assessment.missing_tasks)
        self.assertIn("runtime", assessment.feedback)


class AgenticRAGPipelineTest(unittest.TestCase):
    def test_iterates_after_sufficient_context_feedback(self) -> None:
        request = "Summarize discharge medications, diet instructions, and allergies or adverse reactions."
        trace = AgenticRAGPipeline(build_demo_retriever(), max_rounds=3, top_k=1).run(request)

        self.assertEqual(2, len(trace.queries_by_round))
        self.assertEqual(ContextStatus.INSUFFICIENT, trace.assessments[0].status)
        self.assertEqual(["task_3"], trace.assessments[0].missing_tasks)
        self.assertIn("adverse reactions", trace.queries_by_round[1])
        self.assertEqual(ContextStatus.SUFFICIENT, trace.assessments[-1].status)
        self.assertIn("lisinopril", trace.final_answer)
        self.assertIn("low sodium", trace.final_answer)
        self.assertIn("rash", trace.final_answer)

    def test_root_agent_extracts_tasks_without_domain_presets(self) -> None:
        trace = AgenticRAGPipeline(build_demo_retriever(), max_rounds=1, top_k=1).run(
            "Summarize discharge medications, diet instructions, and allergies or adverse reactions."
        )

        self.assertEqual(
            ["discharge medications", "diet instructions", "allergies or adverse reactions"],
            [task.description for task in trace.tasks],
        )
        self.assertEqual(["task_1", "task_2", "task_3"], [task.name for task in trace.tasks])


class BenchmarkCliTest(unittest.TestCase):
    def test_limit_defaults_to_all_questions(self) -> None:
        args = _parse_args([])

        self.assertIsNone(args.limit)
        self.assertEqual(Path("config.yaml"), args.config)


if __name__ == "__main__":
    unittest.main()
