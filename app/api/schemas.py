"""
API Request and Response Schemas

This module defines Pydantic models for REST API requests, responses, health checks,
and pagination structures.

Why Pydantic:
- Enforces payload validation at HTTP boundaries.
- Auto-generates OpenAPI / Swagger schemas and documentation.
- Guarantees strongly-typed JSON responses for API clients.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime
from app.agent.schemas import AgentStep, Note, AgentRunResult


class ResearchJobRequest(BaseModel):
    """
    Payload for submitting a new research query.
    """
    query: str = Field(
        ...,
        min_length=3,
        max_length=500,
        description="The research question or topic to investigate."
    )
    provider: Optional[str] = Field(
        None,
        description="Optional override for the LLM provider ('groq', 'gemini'). Defaults to server env config."
    )
    max_iterations: Optional[int] = Field(
        None,
        ge=1,
        le=50,
        description="Optional override for maximum research loop iterations."
    )


class ResearchJobResponse(BaseModel):
    """
    Response returned immediately when a research job is accepted.
    """
    job_id: str = Field(..., description="Unique job identifier.")
    query: str = Field(..., description="Original research query.")
    status: str = Field(..., description="Status of the job ('running', 'completed', 'failed').")
    message: str = Field(..., description="Human-readable status summary.")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Job creation timestamp.")


class ResearchJobSummary(BaseModel):
    """
    Summary metadata for job listing endpoint.
    """
    job_id: str = Field(..., description="Unique job identifier.")
    query: str = Field(..., description="Research query.")
    success: bool = Field(..., description="Whether execution finished successfully.")
    total_tokens: int = Field(0, description="Total tokens consumed.")
    total_latency_ms: float = Field(0.0, description="Total LLM latency in ms.")
    created_at: str = Field(..., description="Creation ISO string.")


class JobListResponse(BaseModel):
    """
    Response schema for job history listing.
    """
    total: int = Field(..., description="Total count of jobs returned in this batch.")
    jobs: List[ResearchJobSummary] = Field(default_factory=list, description="List of research job summaries.")


class HealthCheckResponse(BaseModel):
    """
    Response model for server health check monitoring.
    """
    status: str = Field("ok", description="Server health status.")
    database: str = Field("connected", description="Database connection state.")
    default_provider: str = Field(..., description="Configured default LLM provider.")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Current server time.")
