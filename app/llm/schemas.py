"""
LLM Schemas (Pydantic Models)

This module defines the shared data contracts used across the entire LLM layer.
Every message sent to an LLM and every response received is represented as a
typed Pydantic model rather than a raw dictionary.

Why this file exists:
    1. Type safety — Pydantic validates field types at construction time,
       catching bugs before they propagate.
    2. Serialization — models can be dumped to dict/JSON when building
       provider-specific API payloads.
    3. Documentation — the schema IS the documentation; any developer
       can read these models to understand the data flow.

Interview Talking Points:
    - Discuss how Pydantic models serve as runtime-validated data contracts
      that replace the fragile dict-based approach.
    - Explain why Role is an Enum: it constrains the set of valid values
      at the type level, preventing typos like "assitant" from silently
      propagating.
"""

from enum import Enum
from typing import Optional, List
from pydantic import BaseModel, Field


class Role(str, Enum):
    """
    Enumeration of valid message roles in a chat conversation.

    - SYSTEM:    Sets the agent's persona / instructions.
    - USER:      The human (or application) input.
    - ASSISTANT: The LLM's response.
    - TOOL:      The result of a tool execution, fed back to the LLM.
    """
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ToolCall(BaseModel):
    """
    Represents a single tool invocation requested by the LLM.

    Attributes:
        id:        Provider-assigned identifier for this tool call
                   (used to correlate tool results back to the request).
        name:      The name of the tool/function the LLM wants to invoke.
        arguments: A JSON-encoded string of the arguments the LLM wants
                   to pass to the tool. We keep it as a string because
                   the argument schema varies per tool and will be parsed
                   by the tool executor.
    """
    id: str = Field(description="Unique identifier for this tool call")
    name: str = Field(description="Name of the tool/function to invoke")
    arguments: str = Field(description="JSON-encoded arguments for the tool")


class Message(BaseModel):
    """
    Represents a single message in a chat conversation.

    This is the universal message format used across all providers.
    Provider-specific clients are responsible for converting Message
    objects into their SDK's native format.

    Attributes:
        role:         Who sent this message (system, user, assistant, tool).
        content:      The text body of the message. May be None when the
                      assistant response is purely a tool call.
        tool_calls:   List of ToolCall objects if the assistant is requesting
                      tool executions.
        tool_call_id: When role=TOOL, this links the result back to the
                      ToolCall that triggered it.
    """
    role: Role
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    tool_call_id: Optional[str] = None


class TokenUsage(BaseModel):
    """
    Token consumption metrics returned by the provider.

    Not all providers return this data consistently; fields are Optional
    so we can gracefully handle missing values.

    Attributes:
        prompt_tokens:     Tokens consumed by the input messages.
        completion_tokens: Tokens generated in the response.
        total_tokens:      Sum of prompt + completion tokens.
    """
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None


class LLMResponse(BaseModel):
    """
    Unified response object returned by every BaseLLM implementation.

    This is the single return type that the agent loop consumes.
    It abstracts away all provider-specific response structures.

    Attributes:
        content:     The text response from the LLM (None if tool_calls only).
        tool_calls:  List of tool invocations requested by the LLM.
        usage:       Token consumption metrics (None if provider doesn't report).
        model:       The model identifier that generated this response.
        provider:    The provider name (e.g. "groq", "gemini").
        latency_ms:  Round-trip latency in milliseconds.
    """
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    usage: Optional[TokenUsage] = None
    model: str = ""
    provider: str = ""
    latency_ms: float = 0.0
