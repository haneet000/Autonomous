"""
Unit Tests for Application Configuration

This module tests setting defaults and environment variable overrides.
"""

from app.config import settings, Settings


def test_default_settings():
    """Verify default setting values."""
    assert settings.llm_provider in ["groq", "gemini"]
    assert isinstance(settings.max_iterations, int)
    assert settings.max_iterations > 0
    assert isinstance(settings.request_timeout, int)
    assert settings.port == 8000
    assert settings.host == "0.0.0.0"


def test_custom_settings_instantiation():
    """Verify Settings model with custom overrides."""
    custom = Settings(
        llm_provider="gemini",
        max_iterations=15,
        database_path="custom_test.db",
        port=9000
    )
    assert custom.llm_provider == "gemini"
    assert custom.max_iterations == 15
    assert custom.database_path == "custom_test.db"
    assert custom.port == 9000
