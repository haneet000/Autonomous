"""
Unit Tests for SQLite Memory & Repository Layer

This module tests database initialization, foreign key constraints, and CRUD operations.
"""

from app.memory.database import get_db_connection
from app.memory import repository


def test_db_tables_exist(tmp_db_path):
    """Verify that all required database tables are created."""
    with get_db_connection() as conn:
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table';"
        ).fetchall()
        table_names = [r["name"] for r in tables]

    assert "research_jobs" in table_names
    assert "agent_steps" in table_names
    assert "research_notes" in table_names
    assert "visited_urls" in table_names


def test_repository_full_lifecycle(tmp_db_path):
    """Verify complete CRUD lifecycle: create, steps, notes, visited URLs, update, load."""
    job_id = "test-job-lifecycle-1"
    query = "Autonomous AI Agent Architecture"

    # 1. Create Job
    repository.create_job(job_id, query)

    # 2. Insert Step
    repository.insert_step(
        job_id=job_id,
        step_num=1,
        thought="Need to search for clean architecture principles.",
        tool_name="search",
        tool_args={"query": "clean architecture python"},
        observation="Found top 5 articles.",
        latency_ms=150.0
    )

    # 3. Insert Note
    repository.insert_note(
        job_id=job_id,
        note="Decouple memory repository from agent execution loop.",
        source_url="https://example.com/clean-python"
    )

    # 4. Insert Visited URL
    repository.insert_visited_url(job_id, "https://example.com/clean-python")

    # 5. Complete Job
    summary = "# Research Report: Autonomous AI Agent Architecture\n\n## Key Findings\n- Decouple memory."
    repository.update_job(
        job_id=job_id,
        success=True,
        final_summary=summary,
        total_tokens=250,
        total_latency_ms=450.0
    )

    # 6. Load Result
    result = repository.load_job_result(job_id)
    assert result is not None
    assert result.query == query
    assert result.success is True
    assert result.final_summary == summary
    assert len(result.steps) == 1
    assert result.steps[0].tool_name == "search"
    assert len(result.notes) == 1
    assert result.notes[0].note == "Decouple memory repository from agent execution loop."
    assert len(result.visited_urls) == 1
    assert result.visited_urls[0] == "https://example.com/clean-python"


def test_list_recent_jobs(tmp_db_path):
    """Verify listing recent jobs with limit and pagination."""
    for i in range(5):
        job_id = f"job-list-{i}"
        repository.create_job(job_id, f"Query {i}")
        repository.update_job(job_id, success=True, final_summary="Done", total_tokens=100, total_latency_ms=50.0)

    jobs = repository.list_recent_jobs(limit=3, offset=0)
    assert len(jobs) == 3
    assert "job_id" in jobs[0]
    assert "query" in jobs[0]
