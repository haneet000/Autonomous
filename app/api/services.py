"""
API Background Services

This module handles asynchronous background processing of research jobs.

Architectural Decisions:
- Non-blocking execution: Research jobs run in FastAPI BackgroundTasks, returning 202 Accepted
  immediately so client HTTP calls never hang or time out.
- Exception Shielding: Catches unhandled runtime errors during agent execution and updates
  the job status in SQLite repository to 'failed', preventing background thread silent drops.
"""

import logging
from typing import Optional
from app.llm.client import LLMFactory
from app.agent.core import ReActAgent
from app.memory import repository

logger = logging.getLogger("research-agent.api.services")


def execute_research_job(
    job_id: str,
    query: str,
    provider: Optional[str] = None,
    max_iterations: Optional[int] = None
) -> None:
    """
    Background worker target executed asynchronously by FastAPI BackgroundTasks.

    Args:
        job_id: Pre-generated unique research job identifier.
        query: User research question.
        provider: Optional LLM provider override.
        max_iterations: Optional loop iteration cap override.
    """
    logger.info(f"Starting background research execution for job_id '{job_id}' (Query: '{query}')")
    
    try:
        # Create provider LLM client
        llm = LLMFactory.create(provider=provider)
        
        # Instantiate ReAct Agent and run execution loop
        agent = ReActAgent(llm=llm)
        result = agent.run(query=query, job_id=job_id, max_iterations=max_iterations)
        
        logger.info(f"Background research job '{job_id}' completed. Success: {result.success}")

    except Exception as e:
        logger.error(f"Fatal error in background research job '{job_id}': {e}", exc_info=True)
        # Record failure state in database repository
        repository.update_job(
            job_id=job_id,
            success=False,
            final_summary=f"Execution failed due to server error: {str(e)}",
            total_tokens=0,
            total_latency_ms=0.0
        )
