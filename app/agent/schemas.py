"""
ReAct Agent Schemas

This module defines the Pydantic models representing the data structures
used in the ReAct execution loop, including steps, in-memory notes, and
the final execution results.

Why Pydantic:
- Ensures strict typing and validation at the module boundaries.
- Standardizes output shapes which the FastAPI and UI layers can reliably consume.
- Facilitates auto-generation of OpenAPI documentation in later phases.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime


class Note(BaseModel):
    """
    Model representing a key piece of information extracted during research.
    """
    note: str = Field(..., description="The content of the research note/finding.")
    source_url: str = Field(..., description="The source URL from which the finding was extracted.")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="When the note was captured.")


class AgentStep(BaseModel):
    """
    Model representing a single step in the ReAct (Thought-Action-Observation) loop.
    """
    step_num: int = Field(..., description="The sequence number of the step.")
    thought: str = Field(..., description="The agent's reasoning/thought process for this step.")
    tool_name: str = Field(..., description="The name of the tool called (or 'finish').")
    tool_args: Dict[str, Any] = Field(default_factory=dict, description="The arguments passed to the tool.")
    observation: str = Field(..., description="The observation/result returned by the tool.")
    latency_ms: float = Field(..., description="The execution latency of the LLM call for this step in milliseconds.")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="When this step was recorded.")


class AgentRunResult(BaseModel):
    """
    Model representing the final result of a research agent execution run.
    """
    query: str = Field(..., description="The original research query.")
    success: bool = Field(..., description="Whether the research completed successfully.")
    final_summary: str = Field(..., description="The final summarized response compiled by the agent.")
    steps: List[AgentStep] = Field(default_factory=list, description="The list of ReAct steps taken during execution.")
    notes: List[Note] = Field(default_factory=list, description="All individual research notes/findings saved during the run.")
    visited_urls: List[str] = Field(default_factory=list, description="List of unique URLs visited/fetched during the run.")
    total_tokens_used: int = Field(0, description="Total tokens consumed during the run.")
    total_latency_ms: float = Field(0.0, description="Total latency accumulated during LLM steps in milliseconds.")
