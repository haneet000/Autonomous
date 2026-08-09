"""
LLM Custom Exceptions

This module defines a unified exception hierarchy for the LLM layer.
Every provider-specific error (e.g. groq.RateLimitError, google API errors)
is caught inside the client wrappers and re-raised as one of these exceptions.

Why this file exists:
    The rest of the application should never need to know which LLM provider
    is active. By mapping all provider errors into this hierarchy, the agent
    loop can catch `RateLimitError` regardless of whether Groq or Gemini
    raised it. This is the Dependency Inversion Principle (SOLID "D") in action.

Interview Talking Points:
    - Explain how wrapping vendor exceptions behind a custom hierarchy
      enables the Open/Closed Principle: adding a new provider means adding
      a new client class, not modifying existing error-handling code.
    - Discuss the difference between retryable vs. non-retryable errors
      and why this distinction matters for production reliability.
"""


class LLMError(Exception):
    """
    Base exception for all LLM-related errors.
    Every custom exception in this module inherits from LLMError,
    allowing callers to catch all LLM errors with a single except clause.
    """

    def __init__(self, message: str, provider: str = "unknown") -> None:
        self.provider = provider
        super().__init__(f"[{provider}] {message}")


class AuthenticationError(LLMError):
    """
    Raised when the API key is missing, invalid, or revoked (HTTP 401/403).
    This is a non-retryable error — retrying with the same key will always fail.
    """
    pass


class RateLimitError(LLMError):
    """
    Raised when the provider returns HTTP 429 (Too Many Requests).
    This is a retryable error — the retry decorator should back off and try again.
    """
    pass


class ProviderUnavailableError(LLMError):
    """
    Raised on transient server-side failures (HTTP 500, 502, 503, 504)
    or network-level connection timeouts.
    This is a retryable error.
    """
    pass


class InvalidRequestError(LLMError):
    """
    Raised when the request payload is malformed (HTTP 400/422).
    This is a non-retryable error — the caller must fix the input.
    """
    pass


class ToolCallError(LLMError):
    """
    Raised when the LLM response contains a tool call that cannot be
    parsed into our ToolCall schema. This typically indicates a prompt
    engineering issue rather than a provider failure.
    """
    pass
