"""
LLM Package

This package provides a provider-agnostic LLM abstraction layer.
All application components should import from this package rather than
directly using Groq, Gemini, or any other provider SDK.

Public API:
    - LLMFactory.create() — returns a configured BaseLLM implementation
    - generate() / chat() — standard methods on all BaseLLM subclasses
"""
