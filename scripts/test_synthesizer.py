"""
Test Report Synthesizer Script

This script verifies the ReportSynthesizer module in isolation.
It validates:
1. Normal synthesis of research notes into Markdown format with inline citations.
2. Handling of source URL to citation index mapping.
3. Fallback behavior when LLM generation fails or when notes are empty.

Usage:
    python3 scripts/test_synthesizer.py
"""

import sys
import os
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime

# Add project root to python path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from app.agent.schemas import Note
from app.agent.synthesizer import ReportSynthesizer
from app.llm.client import BaseLLM, LLMFactory
from app.llm.schemas import Message, LLMResponse, TokenUsage
from app.llm.exceptions import AuthenticationError

# Set up logging to stdout
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("test_synthesizer")


class MockSuccessLLM(BaseLLM):
    """Mock LLM returning a valid structured Markdown report for testing."""
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> LLMResponse:
        report_text = (
            "# Research Report: Applications of Machine Learning in Healthcare\n\n"
            "## Executive Summary\n"
            "Machine learning is transforming modern medicine through diagnostic precision and efficiency [1][2].\n\n"
            "## Key Findings\n"
            "- ML algorithms detect diabetic retinopathy with 94% accuracy [1].\n"
            "- NLP analyzes EHRs to predict patient readmission rates [2].\n\n"
            "## Sources & References\n"
            "[1] https://example.com/medical-ai-retinopathy\n"
            "[2] https://example.com/ehr-nlp-study\n"
        )
        return LLMResponse(
            content=report_text,
            role="assistant",
            provider="mock",
            model="mock-v1",
            latency_ms=10.0,
            usage=TokenUsage(prompt_tokens=100, completion_tokens=100, total_tokens=200)
        )

    def chat(
        self,
        messages: List[Message],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> LLMResponse:
        return self.generate(prompt="mock", system_prompt=None, temperature=temperature, max_tokens=max_tokens, tools=tools)


class MockFailingLLM(BaseLLM):
    """Mock LLM that raises an exception to test synthesizer fallback resilience."""
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> LLMResponse:
        raise RuntimeError("Simulated LLM API failure for fallback testing.")

    def chat(
        self,
        messages: List[Message],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> LLMResponse:
        raise RuntimeError("Simulated LLM API failure for fallback testing.")


def test_successful_synthesis():
    logger.info("=== Test 1: Successful Report Synthesis ===")
    
    query = "Applications of Machine Learning in Healthcare"
    now_utc = datetime.now()
    notes = [
        Note(
            note="ML algorithms can detect early stage diabetic retinopathy from retinal scans with over 94% accuracy.",
            source_url="https://example.com/medical-ai-retinopathy",
            timestamp=now_utc
        ),
        Note(
            note="Natural Language Processing is used to analyze unstructured Electronic Health Records (EHRs) to predict patient readmissions.",
            source_url="https://example.com/ehr-nlp-study",
            timestamp=now_utc
        ),
        Note(
            note="Deep learning models accelerate drug discovery by predicting molecular binding affinity.",
            source_url="https://example.com/medical-ai-retinopathy",  # Duplicate URL to test index deduplication
            timestamp=now_utc
        )
    ]
    visited_urls = [
        "https://example.com/medical-ai-retinopathy",
        "https://example.com/ehr-nlp-study",
        "https://example.com/additional-source"
    ]
    raw_summary = "Machine learning significantly improves diagnosis speed, EHR analysis, and pharmaceutical discovery."

    # Use live client if configured, otherwise MockSuccessLLM
    try:
        llm = LLMFactory.create()
        logger.info("Using configured live LLM client.")
    except (AuthenticationError, Exception):
        logger.info("API key not set; using MockSuccessLLM for offline verification.")
        llm = MockSuccessLLM()

    synthesizer = ReportSynthesizer(llm)

    report = synthesizer.synthesize(
        query=query,
        notes=notes,
        visited_urls=visited_urls,
        raw_summary=raw_summary
    )

    logger.info("Generated Markdown Report:\n" + "=" * 50 + "\n" + report + "\n" + "=" * 50)

    # Assertions
    assert report.startswith("# "), "Report should start with H1 header (#)"
    assert "Executive Summary" in report, "Report must contain Executive Summary section"
    assert "Sources & References" in report, "Report must contain Sources & References section"
    assert "https://example.com/medical-ai-retinopathy" in report, "Report must cite retinal scan URL"
    assert "https://example.com/ehr-nlp-study" in report, "Report must cite EHR study URL"
    logger.info("✅ Test 1 Passed: Successful Report Synthesis verified.\n")


def test_fallback_synthesis():
    logger.info("=== Test 2: Fallback Report Synthesis on LLM Error ===")

    query = "Quantum Computing Advancements"
    now_utc = datetime.now()
    notes = [
        Note(
            note="Superconducting qubits achieved quantum supremacy in specific matrix calculations.",
            source_url="https://quantum-journal.org/supremacy",
            timestamp=now_utc
        )
    ]
    visited_urls = ["https://quantum-journal.org/supremacy"]

    # Use mock failing LLM
    failing_llm = MockFailingLLM()
    synthesizer = ReportSynthesizer(failing_llm)

    report = synthesizer.synthesize(
        query=query,
        notes=notes,
        visited_urls=visited_urls,
        raw_summary="Quantum computing reached significant milestones."
    )

    logger.info("Generated Fallback Report:\n" + "=" * 50 + "\n" + report + "\n" + "=" * 50)

    assert "# Research Report: Quantum Computing Advancements" in report
    assert "Executive Summary" in report
    assert "https://quantum-journal.org/supremacy" in report
    logger.info("✅ Test 2 Passed: Fallback synthesis executed cleanly.\n")


def test_empty_notes_synthesis():
    logger.info("=== Test 3: Empty Notes Handling ===")

    query = "Unknown Topic With No Findings"
    llm = MockSuccessLLM()
    synthesizer = ReportSynthesizer(llm)

    report = synthesizer.synthesize(
        query=query,
        notes=[],
        visited_urls=[],
        raw_summary=None
    )

    logger.info("Empty Notes Report:\n" + "=" * 50 + "\n" + report + "\n" + "=" * 50)

    assert "no factual notes or webpage data could be gathered" in report
    logger.info("✅ Test 3 Passed: Empty notes handled gracefully.\n")



if __name__ == "__main__":
    logger.info("Starting Report Synthesizer Verification...")
    test_successful_synthesis()
    test_fallback_synthesis()
    test_empty_notes_synthesis()
    logger.info("All Report Synthesizer tests completed successfully!")
