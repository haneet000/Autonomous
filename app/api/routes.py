"""
FastAPI REST API Routes

This module implements the API routers and endpoint handlers for:
- System Health Check
- Research Job Submission (Async Background Execution)
- Job Status & Detail Polling
- Markdown Report Export
- Job History Listing
"""

import uuid
import logging
from typing import Optional
from fastapi import APIRouter, BackgroundTasks, HTTPException, Query, status
from fastapi.responses import PlainTextResponse

from app.config import settings
from app.api.schemas import (
    ResearchJobRequest,
    ResearchJobResponse,
    ResearchJobSummary,
    JobListResponse,
    HealthCheckResponse,
)
from app.agent.schemas import AgentRunResult
from app.api.services import execute_research_job
from app.memory import repository
from app.memory.database import get_db_connection

logger = logging.getLogger("research-agent.api.routes")

router = APIRouter(prefix="/api/v1", tags=["Research Agent API"])


@router.get(
    "/health",
    response_model=HealthCheckResponse,
    summary="Server & Database Health Check",
    description="Returns current server status, database connection state, and configured provider."
)
def health_check() -> HealthCheckResponse:
    """
    Health check endpoint for monitoring uptime and database connectivity.
    """
    db_status = "connected"
    try:
        with get_db_connection() as conn:
            conn.execute("SELECT 1;")
    except Exception as e:
        logger.error(f"Health check database ping failed: {e}")
        db_status = "disconnected"

    return HealthCheckResponse(
        status="ok",
        database=db_status,
        default_provider=settings.llm_provider
    )


@router.post(
    "/research",
    response_model=ResearchJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit Research Query (Async)",
    description="Submits a research topic for asynchronous execution. Returns 202 Accepted immediately with a job_id."
)
def submit_research_job(
    payload: ResearchJobRequest,
    background_tasks: BackgroundTasks
) -> ResearchJobResponse:
    """
    Submit a research query. Begins execution in an async background worker.
    """
    job_id = str(uuid.uuid4())
    logger.info(f"Received API research job request (job_id: '{job_id}', query: '{payload.query}')")

    # 1. Initialize record in database
    repository.create_job(job_id=job_id, query=payload.query)

    # 2. Add job to FastAPI background execution queue
    background_tasks.add_task(
        execute_research_job,
        job_id=job_id,
        query=payload.query,
        provider=payload.provider,
        max_iterations=payload.max_iterations
    )

    return ResearchJobResponse(
        job_id=job_id,
        query=payload.query,
        status="running",
        message="Research job submitted successfully and executing in the background."
    )


@router.get(
    "/research/{job_id}",
    response_model=AgentRunResult,
    summary="Get Research Job Results & Steps",
    description="Retrieves the current execution trace, steps, notes, visited URLs, and final report for a job_id."
)
def get_research_job(job_id: str) -> AgentRunResult:
    """
    Poll research job status and complete results by job_id.
    """
    result = repository.load_job_result(job_id=job_id)
    if not result:
        logger.warning(f"API requested non-existent job_id: '{job_id}'")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Research job with ID '{job_id}' was not found."
        )
    return result


@router.get(
    "/research/{job_id}/report",
    response_class=PlainTextResponse,
    summary="Export Synthesized Markdown Report",
    description="Returns the raw synthesized Markdown research report as plain text/markdown."
)
def get_research_report(job_id: str) -> PlainTextResponse:
    """
    Retrieve the raw Markdown research report document.
    """
    result = repository.load_job_result(job_id=job_id)
    if not result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Research job with ID '{job_id}' was not found."
        )
    return PlainTextResponse(
        content=result.final_summary,
        media_type="text/markdown"
    )


@router.get(
    "/jobs",
    response_model=JobListResponse,
    summary="List Recent Research Jobs",
    description="Returns a paginated summary list of recent research jobs."
)
def list_jobs(
    limit: int = Query(20, ge=1, le=100, description="Max jobs to return."),
    offset: int = Query(0, ge=0, description="Pagination offset.")
) -> JobListResponse:
    """
    List recent research jobs.
    """
    raw_jobs = repository.list_recent_jobs(limit=limit, offset=offset)
    summaries = [
        ResearchJobSummary(
            job_id=j["job_id"],
            query=j["query"],
            success=j["success"],
            total_tokens=j["total_tokens"],
            total_latency_ms=j["total_latency_ms"],
            created_at=str(j["created_at"])
        )
        for j in raw_jobs
    ]
    return JobListResponse(total=len(summaries), jobs=summaries)
