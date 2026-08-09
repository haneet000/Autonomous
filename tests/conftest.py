"""
Shared Pytest Fixtures

This module provides common fixtures for the test suite, including temporary
SQLite database lifecycle management, mock LLM instances, and FastAPI TestClient.

Architectural Rationale:
- Test Isolation: Each test operates on an isolated temporary database.
- Deterministic Execution: Mock LLM implementations allow testing the ReAct agent,
  report synthesizer, and API endpoints without calling external cloud APIs.
"""

import sys
import os
import pytest
from typing import List, Optional, Dict, Any
from datetime import datetime
from fastapi.testclient import TestClient

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.config import settings
from app.memory.database import init_db
from app.llm.client import BaseLLM
from app.llm.schemas import Message, LLMResponse, ToolCall, TokenUsage
from app.main import app


class MockLLM(BaseLLM):
    """
    Mock BaseLLM provider for testing.
    """

    def __init__(self, responses: Optional[List[LLMResponse]] = None) -> None:
        self.responses = responses or []
        self.call_count = 0

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> LLMResponse:
        self.call_count += 1
        if self.responses:
            return self.responses.pop(0)
        
        return LLMResponse(
            content="# Research Report: Test\n\n## Executive Summary\nTest summary.\n\n## Key Findings\n- Fact 1 [1]\n\n## Sources & References\n[1] https://example.com/test",
            role="assistant",
            provider="mock",
            model="mock-v1",
            latency_ms=5.0,
            usage=TokenUsage(prompt_tokens=50, completion_tokens=50, total_tokens=100)
        )

    def chat(
        self,
        messages: List[Message],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> LLMResponse:
        self.call_count += 1
        if self.responses:
            return self.responses.pop(0)

        # Default chat response calls finish tool
        return LLMResponse(
            content="Finished research.",
            role="assistant",
            provider="mock",
            model="mock-v1",
            tool_calls=[
                ToolCall(
                    id="call_mock_1",
                    name="finish",
                    arguments='{"final_summary": "Completed research summary."}'
                )
            ],
            latency_ms=10.0,
            usage=TokenUsage(prompt_tokens=100, completion_tokens=20, total_tokens=120)
        )


@pytest.fixture
def tmp_db_path(tmp_path, monkeypatch):
    """
    Fixture providing a temporary SQLite database path for isolated database tests.
    """
    db_file = tmp_path / "test_suite_agent.db"
    monkeypatch.setattr(settings, "database_path", str(db_file))
    init_db()
    return str(db_file)


@pytest.fixture
def mock_llm():
    """
    Fixture providing a MockLLM instance.
    """
    return MockLLM()


@pytest.fixture
def api_client(tmp_db_path):
    """
    Fixture providing a FastAPI TestClient instance initialized with a temporary database.
    """
    with TestClient(app) as client:
        yield client
