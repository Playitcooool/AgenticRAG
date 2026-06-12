from __future__ import annotations

import unittest

from agentic_rag.agents import SufficientContextAgent
from agentic_rag.demo import build_demo_retriever
from agentic_rag.pipeline import AgenticRAGPipeline
from agentic_rag.retriever import LexicalRetriever
from agentic_rag.types import ClinicalTask, ContextStatus, PatientRecord


class LexicalRetrieverTest(unittest.TestCase):
    def test_searches_all_query_fanouts_and_deduplicates_records(self) -> None:
        retriever = LexicalRetriever(
            [
                PatientRecord("1", "Pharmacy", "Discharge medications include metformin."),
                PatientRecord("2", "Nutrition", "Low sodium diet recommended."),
                PatientRecord("3", "Other", "Follow up in clinic."),
            ]
        )

        results = retriever.search(["discharge medications", "low sodium diet"], top_k=2)

        self.assertEqual({"1", "2"}, {result.record.record_id for result in results})


class SufficientContextAgentTest(unittest.TestCase):
    def test_flags_missing_required_clinical_area(self) -> None:
        tasks = [
            ClinicalTask("pharmacy", "Meds", ("medications",), ("discharge medications",)),
            ClinicalTask("allergies", "Allergies", ("allergy", "rash"), ("allergy history", "rash")),
        ]
        retriever = LexicalRetriever(
            [PatientRecord("1", "Pharmacy", "Discharge medications include lisinopril.")]
        )
        snippets = retriever.search(["discharge medications"], top_k=1)

        assessment = SufficientContextAgent().assess("meds and allergies", tasks, snippets, "draft")

        self.assertEqual(ContextStatus.INSUFFICIENT, assessment.status)
        self.assertEqual(["pharmacy"], assessment.covered_tasks)
        self.assertEqual(["allergies"], assessment.missing_tasks)
        self.assertIn("rash", assessment.feedback)


class AgenticRAGPipelineTest(unittest.TestCase):
    def test_iterates_after_sufficient_context_feedback(self) -> None:
        request = "Summarize discharge medications, diet instructions, and allergies or adverse reactions."
        trace = AgenticRAGPipeline(build_demo_retriever(), max_rounds=3, top_k=1).run(request)

        self.assertEqual(2, len(trace.queries_by_round))
        self.assertEqual(ContextStatus.INSUFFICIENT, trace.assessments[0].status)
        self.assertEqual(["allergies"], trace.assessments[0].missing_tasks)
        self.assertIn("rash", trace.queries_by_round[1])
        self.assertEqual(ContextStatus.SUFFICIENT, trace.assessments[-1].status)
        self.assertIn("lisinopril", trace.final_answer)
        self.assertIn("low sodium", trace.final_answer)
        self.assertIn("rash", trace.final_answer)


if __name__ == "__main__":
    unittest.main()
