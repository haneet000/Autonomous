"""
Unit Tests for Report Synthesizer Module

This module tests ReportSynthesizer citation indexing, Markdown structure, and fallback resilience.
"""

from datetime import datetime
from app.agent.schemas import Note
from app.agent.synthesizer import ReportSynthesizer
from tests.conftest import MockLLM


class MockFailingLLM(MockLLM):
    """Mock LLM that raises an exception during generate call."""
    def generate(self, *args, **kwargs):
        raise RuntimeError("Simulated API failure")


def test_synthesize_report_success():
    """Verify successful report synthesis with citation mapping."""
    now = datetime.now()
    notes = [
        Note(
            note="Transformer architectures enable parallel training across large scale text datasets.",
            source_url="https://arxiv.org/abs/1706.03762",
            timestamp=now
        ),
        Note(
            note="Attention mechanisms allow modeling dependencies regardless of sequence distance.",
            source_url="https://arxiv.org/abs/1706.03762",
            timestamp=now
        )
    ]
    visited_urls = ["https://arxiv.org/abs/1706.03762"]

    llm = MockLLM()
    synthesizer = ReportSynthesizer(llm)

    report = synthesizer.synthesize(
        query="Attention Is All You Need Paper",
        notes=notes,
        visited_urls=visited_urls,
        raw_summary="Transformer paper summary."
    )

    assert "# Research Report:" in report
    assert "Executive Summary" in report
    assert "Sources & References" in report


def test_synthesize_fallback_on_error():
    """Verify fallback synthesis when LLM fails."""
    now = datetime.now()
    notes = [
        Note(
            note="Superconducting qubits require dilution refrigerators at millikelvin temperatures.",
            source_url="https://example.com/quantum-cryo",
            timestamp=now
        )
    ]
    visited_urls = ["https://example.com/quantum-cryo"]

    failing_llm = MockFailingLLM()
    synthesizer = ReportSynthesizer(failing_llm)

    report = synthesizer.synthesize(
        query="Quantum Cryogenics",
        notes=notes,
        visited_urls=visited_urls,
        raw_summary="Cryogenic requirements for quantum chips."
    )

    assert "# Research Report: Quantum Cryogenics" in report
    assert "Executive Summary" in report
    assert "https://example.com/quantum-cryo" in report
    assert "- Superconducting qubits require dilution refrigerators at millikelvin temperatures." in report


def test_synthesize_empty_notes():
    """Verify empty fallback when no notes or raw summary exist."""
    llm = MockLLM()
    synthesizer = ReportSynthesizer(llm)

    report = synthesizer.synthesize(
        query="Non Existent Topic",
        notes=[],
        visited_urls=[],
        raw_summary=None
    )

    assert "no factual notes or webpage data could be gathered" in report
