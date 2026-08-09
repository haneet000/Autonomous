"""
Unit Tests for LLM Abstraction Layer & Retry Logic

This module tests BaseLLM implementations, LLMFactory, and exponential backoff retry logic.
"""

import pytest
from unittest.mock import MagicMock
from app.llm.client import with_retry, LLMFactory, BaseLLM
from app.llm.exceptions import (
    RateLimitError,
    AuthenticationError,
    InvalidRequestError,
)


def test_retry_decorator_recovers_transient_error():
    """Verify with_retry retries on transient RateLimitError and eventually succeeds."""
    attempts = 0

    @with_retry(max_retries=2)
    def flappy_function():
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RateLimitError("Rate limit exceeded")
        return "success"

    result = flappy_function()
    assert result == "success"
    assert attempts == 2


def test_retry_decorator_does_not_retry_auth_error():
    """Verify with_retry immediately fails on non-retryable AuthenticationError."""
    attempts = 0

    @with_retry(max_retries=3)
    def auth_failing_function():
        nonlocal attempts
        attempts += 1
        raise AuthenticationError("Invalid API key")

    with pytest.raises(AuthenticationError):
        auth_failing_function()

    assert attempts == 1


def test_llm_factory_unknown_provider():
    """Verify LLMFactory raises ValueError for unregistered provider."""
    with pytest.raises(ValueError, match="Unknown LLM provider"):
        LLMFactory.create(provider="unknown_vendor")
