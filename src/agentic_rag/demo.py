"""Runnable demonstration of the clinical agentic RAG loop."""

from __future__ import annotations

from agentic_rag.pipeline import AgenticRAGPipeline
from agentic_rag.retriever import LexicalRetriever
from agentic_rag.types import PatientRecord


def build_demo_retriever() -> LexicalRetriever:
    records = [
        PatientRecord(
            record_id="pharmacy-001",
            source="Pharmacy Reconciliation",
            text="Discharge medications: lisinopril 10 mg daily, metformin 500 mg twice daily, and atorvastatin 20 mg nightly.",
        ),
        PatientRecord(
            record_id="nutrition-001",
            source="Nutrition Notes",
            text="Nutrition team recommended a low sodium diet with carbohydrate-controlled meals and daily weight monitoring.",
        ),
        PatientRecord(
            record_id="clinical-001",
            source="Progress Note",
            text="Patient reported an itchy rash after the first dose of amoxicillin. Antibiotic was stopped and adverse reaction was documented.",
        ),
        PatientRecord(
            record_id="clinical-002",
            source="Discharge Summary",
            text="The patient improved with diuresis and diabetes education. Follow up with primary care in one week.",
        ),
    ]
    return LexicalRetriever(records)


def main() -> None:
    request = "Summarize this patient's discharge medications, diet instructions, and allergies or adverse reactions."
    trace = AgenticRAGPipeline(build_demo_retriever(), max_rounds=3, top_k=1).run(request)

    print("PHASE TRACE")
    for index, queries in enumerate(trace.queries_by_round, start=1):
        print(f"\nRound {index} queries:")
        for query in queries:
            print(f"  - {query}")
        assessment = trace.assessments[index - 1]
        print(f"Assessment: {assessment.status.value}")
        print(f"Finding: {assessment.finding}")
        print(f"Gap: {assessment.gap}")

    print("\nFINAL ANSWER")
    print(trace.final_answer)


if __name__ == "__main__":
    main()
