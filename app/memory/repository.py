"""
Memory Repository Layer

This module provides data access functions (CRUD operations) for the SQLite database.
It abstracts all raw SQL queries away from the agent execution layer.

Architectural Decisions:
- Decoupling: Separation of concerns between DB queries and core ReAct agent logic.
- Type mapping: Translates SQLite database records to strongly typed Pydantic models
  like Note, AgentStep, and AgentRunResult.
- Serialization: Saves dictionary structures to SQLite as text via json.dumps and
  reconstructs them via json.loads.
"""

import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from app.memory.database import get_db_connection
from app.agent.schemas import Note as NoteSchema, AgentStep as StepSchema, AgentRunResult

logger = logging.getLogger("research-agent.memory.repository")


def create_job(job_id: str, query: str) -> None:
    """
    Creates a new research job entry.
    """
    logger.debug(f"Creating job record in DB: job_id={job_id}, query='{query}'")
    query_sql = """
    INSERT INTO research_jobs (job_id, query)
    VALUES (?, ?);
    """
    with get_db_connection() as conn:
        conn.execute(query_sql, (job_id, query))


def update_job(
    job_id: str,
    success: bool,
    final_summary: str,
    total_tokens: int,
    total_latency_ms: float
) -> None:
    """
    Updates the final results of a research job.
    """
    logger.debug(f"Updating job record in DB: job_id={job_id}, success={success}")
    query_sql = """
    UPDATE research_jobs
    SET success = ?, final_summary = ?, total_tokens = ?, total_latency_ms = ?
    WHERE job_id = ?;
    """
    with get_db_connection() as conn:
        conn.execute(
            query_sql,
            (1 if success else 0, final_summary, total_tokens, total_latency_ms, job_id)
        )


def insert_step(
    job_id: str,
    step_num: int,
    thought: str,
    tool_name: str,
    tool_args: Dict[str, Any],
    observation: str,
    latency_ms: float
) -> None:
    """
    Inserts a single ReAct step into the agent_steps table.
    """
    logger.debug(f"Saving step {step_num} for job_id {job_id} in DB.")
    query_sql = """
    INSERT INTO agent_steps (job_id, step_num, thought, tool_name, tool_args, observation, latency_ms)
    VALUES (?, ?, ?, ?, ?, ?, ?);
    """
    args_json = json.dumps(tool_args)
    with get_db_connection() as conn:
        conn.execute(
            query_sql,
            (job_id, step_num, thought, tool_name, args_json, observation, latency_ms)
        )


def insert_note(job_id: str, note: str, source_url: str) -> None:
    """
    Inserts a research finding/note into the research_notes table.
    """
    logger.debug(f"Saving note for job_id {job_id} in DB.")
    query_sql = """
    INSERT INTO research_notes (job_id, note, source_url)
    VALUES (?, ?, ?);
    """
    with get_db_connection() as conn:
        conn.execute(query_sql, (job_id, note, source_url))


def insert_visited_url(job_id: str, url: str) -> None:
    """
    Inserts a visited/fetched URL. IGNOREs duplicates silently.
    """
    logger.debug(f"Saving visited URL for job_id {job_id} in DB: {url}")
    query_sql = """
    INSERT OR IGNORE INTO visited_urls (job_id, url)
    VALUES (?, ?);
    """
    with get_db_connection() as conn:
        conn.execute(query_sql, (job_id, url))


def load_job_result(job_id: str) -> Optional[AgentRunResult]:
    """
    Retrieves the complete result of a research run, including all steps, notes, and visited URLs.

    Args:
        job_id: The unique research job identifier.

    Returns:
        An AgentRunResult schema object, or None if the job is not found.
    """
    logger.info(f"Loading job results from DB for job_id: {job_id}")
    
    job_sql = "SELECT * FROM research_jobs WHERE job_id = ?;"
    steps_sql = "SELECT * FROM agent_steps WHERE job_id = ? ORDER BY step_num ASC;"
    notes_sql = "SELECT * FROM research_notes WHERE job_id = ? ORDER BY created_at ASC;"
    urls_sql = "SELECT url FROM visited_urls WHERE job_id = ? ORDER BY created_at ASC;"

    with get_db_connection() as conn:
        # 1. Fetch Job Metadata
        job_row = conn.execute(job_sql, (job_id,)).fetchone()
        if not job_row:
            logger.warning(f"Job ID {job_id} not found in database.")
            return None

        # 2. Fetch Steps
        step_rows = conn.execute(steps_sql, (job_id,)).fetchall()
        steps: List[StepSchema] = []
        for r in step_rows:
            try:
                args_dict = json.loads(r["tool_args"])
            except Exception:
                args_dict = {"raw": r["tool_args"]}
            
            # Convert timestamp to datetime object if present
            timestamp = datetime.utcnow()
            if "created_at" in r.keys() and r["created_at"]:
                try:
                    timestamp = datetime.strptime(r["created_at"], "%Y-%m-%d %H:%M:%S")
                except ValueError:
                    pass

            steps.append(
                StepSchema(
                    step_num=r["step_num"],
                    thought=r["thought"] or "",
                    tool_name=r["tool_name"],
                    tool_args=args_dict,
                    observation=r["observation"],
                    latency_ms=r["latency_ms"],
                    timestamp=timestamp
                )
            )

        # 3. Fetch Notes
        note_rows = conn.execute(notes_sql, (job_id,)).fetchall()
        notes: List[NoteSchema] = []
        for r in note_rows:
            timestamp = datetime.utcnow()
            if "created_at" in r.keys() and r["created_at"]:
                try:
                    timestamp = datetime.strptime(r["created_at"], "%Y-%m-%d %H:%M:%S")
                except ValueError:
                    pass

            notes.append(
                NoteSchema(
                    note=r["note"],
                    source_url=r["source_url"],
                    timestamp=timestamp
                )
            )

        # 4. Fetch Visited URLs
        url_rows = conn.execute(urls_sql, (job_id,)).fetchall()
        visited_urls = [r["url"] for r in url_rows]

    # Map to typed AgentRunResult Pydantic model
    return AgentRunResult(
        query=job_row["query"],
        success=bool(job_row["success"]),
        final_summary=job_row["final_summary"] or "",
        steps=steps,
        notes=notes,
        visited_urls=visited_urls,
        total_tokens_used=job_row["total_tokens"],
        total_latency_ms=job_row["total_latency_ms"]
    )


def list_recent_jobs(limit: int = 20, offset: int = 0) -> List[Dict[str, Any]]:
    """
    Retrieves a list of recent research jobs ordered by creation date descending.

    Args:
        limit: Max number of jobs to retrieve.
        offset: Offset for pagination.

    Returns:
        List of dictionaries containing job summary metadata.
    """
    logger.debug(f"Listing recent jobs (limit={limit}, offset={offset}) from DB.")
    query_sql = """
    SELECT job_id, query, success, total_tokens, total_latency_ms, created_at
    FROM research_jobs
    ORDER BY created_at DESC
    LIMIT ? OFFSET ?;
    """
    with get_db_connection() as conn:
        rows = conn.execute(query_sql, (limit, offset)).fetchall()
        jobs = []
        for r in rows:
            jobs.append({
                "job_id": r["job_id"],
                "query": r["query"],
                "success": bool(r["success"]),
                "total_tokens": r["total_tokens"],
                "total_latency_ms": r["total_latency_ms"],
                "created_at": r["created_at"]
            })
        return jobs

