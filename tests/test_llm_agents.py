from __future__ import annotations

import unittest

from agentic_rag.llm import OpenAICompatibleClient
from agentic_rag.llm_agents import LLMRootAgent, LLMSufficientContextAgent
from agentic_rag.types import ContextStatus, DocumentChunk, RetrievalTask, SearchResult


class FakeLLMClient:
    def __init__(self, payload):
        self.payload = payload

    def chat_json(self, system: str, user: str):
        return self.payload


class LLMRootAgentTest(unittest.TestCase):
    def test_default_client_api_key_matches_local_server(self) -> None:
        self.assertEqual("no_need", OpenAICompatibleClient().api_key)

    def test_parses_tasks_from_json_response(self) -> None:
        agent = LLMRootAgent(
            FakeLLMClient(
                {
                    "tasks": [
                        {
                            "description": "adverse reactions",
                            "required_terms": ["rash", "reaction"],
                            "search_hints": ["rash during stay", "adverse reaction"],
                        }
                    ]
                }
            )
        )

        tasks = agent.parse("Summarize adverse reactions.")

        self.assertEqual("adverse reactions", tasks[0].description)
        self.assertEqual(("rash", "reaction"), tasks[0].required_terms)
        self.assertEqual(("rash during stay", "adverse reaction"), tasks[0].search_hints)


class LLMSufficientContextAgentTest(unittest.TestCase):
    def test_uses_json_context_decision(self) -> None:
        agent = LLMSufficientContextAgent(
            FakeLLMClient(
                {
                    "status": "insufficient",
                    "finding": "medications are grounded",
                    "gap": "allergy evidence is missing",
                    "covered_tasks": ["task_1"],
                    "missing_tasks": ["task_2"],
                    "feedback": ["rash", "allergic reaction"],
                }
            )
        )
        tasks = [
            RetrievalTask("task_1", "medications", ("medications",), ("medications",)),
            RetrievalTask("task_2", "allergies", ("allergy",), ("allergies",)),
        ]
        snippets = [SearchResult("medications", DocumentChunk("1", "Pharmacy", "Discharge medications include metformin."), 1.0)]

        assessment = agent.assess("meds and allergies", tasks, snippets, "draft")

        self.assertEqual(ContextStatus.INSUFFICIENT, assessment.status)
        self.assertEqual(["task_1"], assessment.covered_tasks)
        self.assertEqual(["task_2"], assessment.missing_tasks)
        self.assertEqual(["rash", "allergic reaction"], assessment.feedback)


if __name__ == "__main__":
    unittest.main()
