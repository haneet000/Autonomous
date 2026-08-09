"""
Unit & Integration Tests for ReAct Agent Core Loop

This module tests the ReAct loop execution, tool invocation, duplicate URL fetch guardrails,
and step iteration limit handling.
"""

from app.agent.core import ReActAgent
from app.llm.schemas import LLMResponse, ToolCall, TokenUsage
from tests.conftest import MockLLM


def test_agent_run_successful_finish(tmp_db_path):
    """Verify complete agent run loop executing save_note and finish tools."""
    # Step 1 response: call save_note
    resp1 = LLMResponse(
        content="I should save this key finding.",
        role="assistant",
        provider="mock",
        model="mock-v1",
        tool_calls=[
            ToolCall(
                id="call_1",
                name="save_note",
                arguments='{"note": "Microservices enhance scalability.", "source_url": "https://example.com/microservices"}'
            )
        ],
        latency_ms=10.0,
        usage=TokenUsage(prompt_tokens=50, completion_tokens=20, total_tokens=70)
    )

    # Step 2 response: call finish
    resp2 = LLMResponse(
        content="I have gathered enough facts to conclude.",
        role="assistant",
        provider="mock",
        model="mock-v1",
        tool_calls=[
            ToolCall(
                id="call_2",
                name="finish",
                arguments='{"final_summary": "Microservices architectures decouple deployment units for scale."}'
            )
        ],
        latency_ms=15.0,
        usage=TokenUsage(prompt_tokens=80, completion_tokens=30, total_tokens=110)
    )

    # Synthesis response
    resp_synth = LLMResponse(
        content="# Research Report: Microservices\n\n## Executive Summary\nMicroservices decouple units.\n\n## Key Findings\n- Microservices enhance scalability. [1]\n\n## Sources & References\n[1] https://example.com/microservices",
        role="assistant",
        provider="mock",
        model="mock-v1",
        latency_ms=12.0,
        usage=TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
    )

    mock_llm = MockLLM(responses=[resp1, resp2, resp_synth])
    agent = ReActAgent(llm=mock_llm)

    result = agent.run(query="Microservices Architecture Benefits", max_iterations=5)

    assert result.success is True
    assert len(result.notes) == 1
    assert result.notes[0].note == "Microservices enhance scalability."
    assert result.notes[0].source_url == "https://example.com/microservices"
    assert len(result.steps) == 2
    assert "# Research Report: Microservices" in result.final_summary


def test_agent_duplicate_fetch_prevented(tmp_db_path):
    """Verify duplicate fetch tool calls return warning observation without re-fetching."""
    agent = ReActAgent(llm=MockLLM())
    agent.visited_urls.add("https://example.com/already-fetched")

    obs = agent._execute_tool("fetch_page", {"url": "https://example.com/already-fetched"})
    assert "already fetched this url" in obs.lower()


def test_agent_iteration_limit_cap(tmp_db_path):
    """Verify agent stops cleanly when max_iterations is reached without calling finish."""
    # Endless search calls
    search_resp = LLMResponse(
        content="Searching for more info.",
        role="assistant",
        provider="mock",
        model="mock-v1",
        tool_calls=[
            ToolCall(
                id="call_loop",
                name="search",
                arguments='{"query": "endless query"}'
            )
        ],
        latency_ms=5.0,
        usage=TokenUsage(prompt_tokens=30, completion_tokens=10, total_tokens=40)
    )

    # 3 search responses for 2 max iterations (plus synthesis)
    mock_llm = MockLLM(responses=[search_resp, search_resp, search_resp])
    agent = ReActAgent(llm=mock_llm)

    result = agent.run(query="Infinite Search Query", max_iterations=2)

    assert len(result.steps) == 2
    assert result.success is True
    assert agent.finished is True
