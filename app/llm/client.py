"""
LLM Client Layer — Provider-Agnostic Interface

This module implements the Strategy Pattern for LLM access:

    BaseLLM (abstract interface)
        ├── GroqClient   (primary — Llama 3.3 via Groq Cloud)
        └── GeminiClient (fallback — Gemini Flash via Google AI)

    LLMFactory.create()  →  returns the configured BaseLLM implementation

Why this architecture exists:
    The agent loop calls `llm.generate(...)` or `llm.chat(...)`. It never
    knows (or cares) which provider is behind the interface. Swapping from
    Groq to OpenAI tomorrow requires only a new client class and a factory
    entry — zero changes in the agent code. This is the Open/Closed Principle
    (SOLID "O") and Strategy Pattern in action.

Why exponential backoff with jitter:
    Transient errors (429, 5xx) are common with hosted LLM APIs. Fixed-delay
    retries cause "thundering herd" problems when multiple workers retry at the
    same instant. Jitter randomises the wait window, spreading load.

Interview Talking Points:
    - Explain the Strategy Pattern and how it enables runtime provider switching.
    - Discuss why the retry decorator is a cross-cutting concern separated from
      business logic (Single Responsibility Principle).
    - Explain the difference between `generate()` (single prompt) and `chat()`
      (multi-turn conversation with message history).
"""

import json
import time
import random
import logging
import functools
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Callable, Type

from app.config import settings
from app.llm.schemas import (
    Message, Role, ToolCall, TokenUsage, LLMResponse,
)
from app.llm.exceptions import (
    LLMError,
    AuthenticationError,
    RateLimitError,
    ProviderUnavailableError,
    InvalidRequestError,
    ToolCallError,
)

logger = logging.getLogger("research-agent.llm")


# ─────────────────────────────────────────────────────────────────────────────
# Retry Decorator
# ─────────────────────────────────────────────────────────────────────────────

def with_retry(
    max_retries: Optional[int] = None,
    retryable_exceptions: tuple = (RateLimitError, ProviderUnavailableError),
) -> Callable:
    """
    Decorator factory that wraps an LLM call with exponential backoff + jitter.

    Only retries on transient, retryable exceptions. Non-retryable errors
    (AuthenticationError, InvalidRequestError) propagate immediately.

    Args:
        max_retries:          Override for settings.max_retries.
        retryable_exceptions: Tuple of exception classes that should trigger retries.

    Returns:
        A decorator that wraps the target function.
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            retries = max_retries if max_retries is not None else settings.max_retries
            last_exception: Optional[Exception] = None

            for attempt in range(retries + 1):
                try:
                    return func(*args, **kwargs)
                except retryable_exceptions as exc:
                    last_exception = exc
                    if attempt < retries:
                        # Exponential backoff: 1s, 2s, 4s ... with ±25% jitter
                        base_delay = 2 ** attempt
                        jitter = base_delay * 0.25 * (random.random() * 2 - 1)
                        wait = max(0.1, base_delay + jitter)
                        logger.warning(
                            f"Retry {attempt + 1}/{retries} after {type(exc).__name__}: "
                            f"{exc}. Waiting {wait:.2f}s..."
                        )
                        time.sleep(wait)
                    else:
                        logger.error(
                            f"All {retries} retries exhausted. Last error: {exc}"
                        )
                        raise
            # Should be unreachable, but satisfies the type checker
            raise last_exception  # type: ignore[misc]
        return wrapper
    return decorator


# ─────────────────────────────────────────────────────────────────────────────
# Abstract Base — BaseLLM
# ─────────────────────────────────────────────────────────────────────────────

class BaseLLM(ABC):
    """
    Abstract interface that all LLM provider clients must implement.

    Methods:
        generate():  Single-turn text generation from a prompt string.
        chat():      Multi-turn conversation from a list of Message objects.

    Subclasses are responsible for:
        - Translating our Message/ToolCall schemas into provider-native formats.
        - Mapping provider-specific exceptions into our custom hierarchy.
        - Extracting token usage when available.
    """

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> LLMResponse:
        """
        Single-turn generation.

        Args:
            prompt:        The user prompt text.
            system_prompt: Optional system instructions.
            temperature:   Sampling temperature override.
            max_tokens:    Max output tokens override.
            tools:         Tool/function definitions for function calling.

        Returns:
            A unified LLMResponse.
        """
        ...

    @abstractmethod
    def chat(
        self,
        messages: List[Message],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> LLMResponse:
        """
        Multi-turn chat completion.

        Args:
            messages:    Ordered list of Message objects (system, user, assistant, tool).
            temperature: Sampling temperature override.
            max_tokens:  Max output tokens override.
            tools:       Tool/function definitions for function calling.

        Returns:
            A unified LLMResponse.
        """
        ...


# ─────────────────────────────────────────────────────────────────────────────
# Groq Client
# ─────────────────────────────────────────────────────────────────────────────

class GroqClient(BaseLLM):
    """
    LLM client implementation backed by the Groq Cloud API.

    Uses the official `groq` Python SDK which exposes an OpenAI-compatible
    chat completions interface. Groq runs Llama, Mixtral, and Gemma models
    on custom LPU hardware, offering very low latency inference.

    Architectural Notes:
        - The Groq SDK mirrors the OpenAI SDK structure, so messages are
          passed as a list of dicts with 'role' and 'content' keys.
        - Tool calls follow the OpenAI function-calling format.
    """

    PROVIDER_NAME = "groq"

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None) -> None:
        """
        Initialise the Groq client.

        Args:
            api_key: Groq API key. Falls back to settings.groq_api_key.
            model:   Model identifier. Falls back to settings.default_model.
        """
        self._api_key = api_key or settings.groq_api_key
        self._model = model or settings.default_model

        if not self._api_key:
            raise AuthenticationError(
                "Groq API key is not set. Set GROQ_API_KEY in your .env file.",
                provider=self.PROVIDER_NAME,
            )

        # Lazy import to avoid hard dependency at module level
        import groq as groq_sdk
        self._client = groq_sdk.Groq(api_key=self._api_key)
        logger.info(f"GroqClient initialised with model '{self._model}'")

    # ── Public methods ───────────────────────────────────────────────────

    @with_retry()
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> LLMResponse:
        """Single-turn generation via Groq."""
        messages: List[Message] = []
        if system_prompt:
            messages.append(Message(role=Role.SYSTEM, content=system_prompt))
        messages.append(Message(role=Role.USER, content=prompt))
        return self.chat(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
        )

    @with_retry()
    def chat(
        self,
        messages: List[Message],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> LLMResponse:
        """Multi-turn chat completion via Groq."""
        temp = temperature if temperature is not None else settings.temperature
        tokens = max_tokens if max_tokens is not None else settings.max_tokens

        # Convert our Message models → Groq-native dicts
        raw_messages = self._build_messages(messages)

        # Build kwargs for the API call
        kwargs: Dict[str, Any] = {
            "model": self._model,
            "messages": raw_messages,
            "temperature": temp,
            "max_tokens": tokens,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        logger.info(
            f"Groq request: model={self._model}, messages={len(messages)}, "
            f"temp={temp}, max_tokens={tokens}, tools={'yes' if tools else 'no'}"
        )

        start = time.perf_counter()
        try:
            response = self._client.chat.completions.create(**kwargs)
        except Exception as exc:
            # Fallback recovery for Groq tool_use_failed errors
            recovered_calls = self._recover_failed_generation(exc)
            if recovered_calls:
                latency_ms = (time.perf_counter() - start) * 1000
                logger.info(f"Successfully recovered {len(recovered_calls)} tool call(s) from Groq failed_generation error.")
                return LLMResponse(
                    content=None,
                    tool_calls=recovered_calls,
                    usage=None,
                    model=self._model,
                    provider=self.PROVIDER_NAME,
                    latency_ms=round(latency_ms, 2),
                )
            self._handle_groq_error(exc)

        latency_ms = (time.perf_counter() - start) * 1000

        return self._parse_response(response, latency_ms)

    # ── Private helpers ──────────────────────────────────────────────────

    @staticmethod
    def _build_messages(messages: List[Message]) -> List[Dict[str, Any]]:
        """Convert our Message schema list to Groq-compatible dicts."""
        raw: List[Dict[str, Any]] = []
        for msg in messages:
            entry: Dict[str, Any] = {
                "role": msg.role.value,
                "content": msg.content or "",
            }
            if msg.tool_calls:
                entry["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": tc.arguments,
                        },
                    }
                    for tc in msg.tool_calls
                ]
            if msg.tool_call_id:
                entry["tool_call_id"] = msg.tool_call_id
            raw.append(entry)
        return raw

    def _parse_response(self, response: Any, latency_ms: float) -> LLMResponse:
        """Extract our unified LLMResponse from a Groq SDK response object."""
        choice = response.choices[0]
        assistant_msg = choice.message

        # Extract text content
        content = assistant_msg.content

        # Extract tool calls (if any)
        tool_calls: Optional[List[ToolCall]] = None
        if assistant_msg.tool_calls:
            tool_calls = []
            for tc in assistant_msg.tool_calls:
                try:
                    tool_calls.append(
                        ToolCall(
                            id=tc.id,
                            name=tc.function.name,
                            arguments=tc.function.arguments,
                        )
                    )
                except Exception as parse_err:
                    raise ToolCallError(
                        f"Failed to parse tool call: {parse_err}",
                        provider=self.PROVIDER_NAME,
                    )

        # Extract token usage
        usage: Optional[TokenUsage] = None
        if response.usage:
            usage = TokenUsage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
            )

        llm_response = LLMResponse(
            content=content,
            tool_calls=tool_calls,
            usage=usage,
            model=response.model,
            provider=self.PROVIDER_NAME,
            latency_ms=round(latency_ms, 2),
        )

        logger.info(
            f"Groq response: model={response.model}, latency={latency_ms:.0f}ms, "
            f"tokens={usage.total_tokens if usage else 'N/A'}, "
            f"tool_calls={len(tool_calls) if tool_calls else 0}"
        )
        return llm_response

    @staticmethod
    def _recover_failed_generation(exc: Exception) -> Optional[List[ToolCall]]:
        """
        Attempt to parse tool calls from Groq's failed_generation error message.

        Llama models on Groq sometimes generate XML tags like `<function=name{...}</function>`
        which trigger Groq's server-side tool_use_failed error. This method recovers
        the structured ToolCall objects from the error payload.
        """
        import re
        import uuid

        failed_gen = ""
        body = getattr(exc, "body", None) or {}
        if isinstance(body, dict):
            err_info = body.get("error", {})
            if isinstance(err_info, dict):
                failed_gen = err_info.get("failed_generation", "")

        if not failed_gen:
            exc_str = str(exc)
            match = re.search(r"['\"]failed_generation['\"]:\s*['\"](.*?)['\"]", exc_str, re.DOTALL)
            if match:
                failed_gen = match.group(1).replace("\\n", "\n").replace('\\"', '"')

        if not failed_gen:
            return None

        # Pattern 1: <function=name{"arg": "val"}</function>
        matches = re.findall(r'<function=([a-zA-Z0-9_-]+)\s*(\{.*?\})</function>', failed_gen, re.DOTALL)
        if not matches:
            # Pattern 2: <function=name>{"arg": "val"}</function>
            matches = re.findall(r'<function=([a-zA-Z0-9_-]+)>(\{.*?\})</function>', failed_gen, re.DOTALL)
        if not matches:
            # Pattern 3: <function=name{"arg": "val"}
            matches = re.findall(r'<function=([a-zA-Z0-9_-]+)\s*(\{.*?\})', failed_gen, re.DOTALL)

        if matches:
            tool_calls = []
            for name, args_str in matches:
                tool_calls.append(
                    ToolCall(
                        id=f"call_{uuid.uuid4().hex[:8]}",
                        name=name.strip(),
                        arguments=args_str.strip(),
                    )
                )
            return tool_calls
        return None

    def _handle_groq_error(self, exc: Exception) -> None:
        """Map Groq SDK exceptions to our custom hierarchy and re-raise."""
        import groq as groq_sdk

        if isinstance(exc, groq_sdk.AuthenticationError):
            raise AuthenticationError(str(exc), provider=self.PROVIDER_NAME) from exc
        if isinstance(exc, groq_sdk.RateLimitError):
            raise RateLimitError(str(exc), provider=self.PROVIDER_NAME) from exc
        if isinstance(exc, groq_sdk.BadRequestError):
            raise InvalidRequestError(str(exc), provider=self.PROVIDER_NAME) from exc
        if isinstance(exc, (groq_sdk.InternalServerError, groq_sdk.APIConnectionError)):
            raise ProviderUnavailableError(str(exc), provider=self.PROVIDER_NAME) from exc
        # Catch-all for unexpected Groq errors
        raise LLMError(f"Unexpected Groq error: {exc}", provider=self.PROVIDER_NAME) from exc


# ─────────────────────────────────────────────────────────────────────────────
# Gemini Client
# ─────────────────────────────────────────────────────────────────────────────

class GeminiClient(BaseLLM):
    """
    LLM client implementation backed by the Google Gemini API.

    Uses the new `google-genai` SDK (google.genai.Client). This client
    serves as the fallback provider when Groq is unavailable or rate-limited.

    Architectural Notes:
        - Gemini uses a Content/Part model instead of the OpenAI-style
          role/content dicts, so we translate between the two formats.
        - Token usage comes from response.usage_metadata.
    """

    PROVIDER_NAME = "gemini"

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None) -> None:
        """
        Initialise the Gemini client.

        Args:
            api_key: Gemini API key. Falls back to settings.gemini_api_key.
            model:   Model identifier. Falls back to settings.gemini_model.
        """
        self._api_key = api_key or settings.gemini_api_key
        self._model = model or settings.gemini_model

        if not self._api_key:
            raise AuthenticationError(
                "Gemini API key is not set. Set GEMINI_API_KEY in your .env file.",
                provider=self.PROVIDER_NAME,
            )

        from google import genai
        self._client = genai.Client(api_key=self._api_key)
        logger.info(f"GeminiClient initialised with model '{self._model}'")

    # ── Public methods ───────────────────────────────────────────────────

    @with_retry()
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> LLMResponse:
        """Single-turn generation via Gemini."""
        messages: List[Message] = []
        if system_prompt:
            messages.append(Message(role=Role.SYSTEM, content=system_prompt))
        messages.append(Message(role=Role.USER, content=prompt))
        return self.chat(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
        )

    @with_retry()
    def chat(
        self,
        messages: List[Message],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> LLMResponse:
        """Multi-turn chat completion via Gemini."""
        from google.genai import types

        temp = temperature if temperature is not None else settings.temperature
        tokens = max_tokens if max_tokens is not None else settings.max_tokens

        # Separate system instruction from conversation messages
        system_instruction, contents = self._build_contents(messages)

        # Convert OpenAI-style tools to Gemini Tool objects
        gemini_tools = None
        if tools:
            function_declarations = []
            for tool in tools:
                if tool.get("type") == "function" and "function" in tool:
                    func = tool["function"]

                    # Convert parameter type strings to uppercase (Google SDK requirement)
                    def _uppercase_types(schema: Any) -> Any:
                        if isinstance(schema, dict):
                            return {
                                k: (v.upper() if k == "type" and isinstance(v, str) else _uppercase_types(v))
                                for k, v in schema.items()
                            }
                        elif isinstance(schema, list):
                            return [_uppercase_types(item) for item in schema]
                        return schema

                    function_declarations.append({
                        "name": func["name"],
                        "description": func.get("description", ""),
                        "parameters": _uppercase_types(func.get("parameters", {})),
                    })

            if function_declarations:
                gemini_tools = [types.Tool(function_declarations=function_declarations)]

        # Build generation config — pass all fields in constructor (SDK 2.x is immutable)
        config = types.GenerateContentConfig(
            temperature=temp,
            max_output_tokens=tokens,
            system_instruction=system_instruction or None,
            tools=gemini_tools,
        )

        logger.info(
            f"Gemini request: model={self._model}, messages={len(messages)}, "
            f"temp={temp}, max_tokens={tokens}, tools={'yes' if tools else 'no'}"
        )

        start = time.perf_counter()
        try:
            response = self._client.models.generate_content(
                model=self._model,
                contents=contents,
                config=config,
            )
        except Exception as exc:
            self._handle_gemini_error(exc)

        latency_ms = (time.perf_counter() - start) * 1000

        return self._parse_response(response, latency_ms)

    # ── Private helpers ──────────────────────────────────────────────────

    @staticmethod
    def _build_contents(messages: List[Message]) -> tuple:
        """
        Split messages into (system_instruction, contents).

        Gemini separates system instructions from the conversation.
        System messages become the system instruction; user/assistant/tool
        messages become Content objects with native parts.

        Returns:
            (system_instruction_str_or_None, list_of_content_dicts)
        """
        from google.genai import types

        system_parts: List[str] = []
        contents: List[types.Content] = []

        for msg in messages:
            if msg.role == Role.SYSTEM:
                if msg.content:
                    system_parts.append(msg.content)
            elif msg.role == Role.USER:
                contents.append(
                    types.Content(
                        role="user",
                        parts=[types.Part.from_text(text=msg.content or "")],
                    )
                )
            elif msg.role == Role.ASSISTANT:
                parts = []
                if msg.content:
                    parts.append(types.Part.from_text(text=msg.content))
                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        try:
                            args_dict = json.loads(tc.arguments) if isinstance(tc.arguments, str) else tc.arguments
                            if not isinstance(args_dict, dict):
                                args_dict = {"query": tc.arguments}
                        except Exception:
                            args_dict = {"args": tc.arguments}
                        parts.append(
                            types.Part.from_function_call(
                                name=tc.name,
                                args=args_dict
                            )
                        )
                contents.append(
                    types.Content(
                        role="model",
                        parts=parts,
                    )
                )
            elif msg.role == Role.TOOL:
                # Find matching tool name from history by tool_call_id
                name = "unknown_tool"
                for prev_msg in reversed(messages):
                    if prev_msg.tool_calls:
                        for tc in prev_msg.tool_calls:
                            if tc.id == msg.tool_call_id:
                                name = tc.name
                                break
                try:
                    resp_obj = json.loads(msg.content or "{}")
                    if not isinstance(resp_obj, dict):
                        resp_obj = {"result": resp_obj}
                except Exception:
                    resp_obj = {"result": msg.content or ""}

                contents.append(
                    types.Content(
                        role="tool",
                        parts=[
                            types.Part.from_function_response(
                                name=name,
                                response=resp_obj
                            )
                        ]
                    )
                )

        system_instruction = "\n".join(system_parts) if system_parts else None
        return system_instruction, contents


    def _parse_response(self, response: Any, latency_ms: float) -> LLMResponse:
        """Extract our unified LLMResponse from a Gemini SDK response object."""
        # Extract text content
        content: Optional[str] = None
        tool_calls: Optional[List[ToolCall]] = None

        if response.candidates and response.candidates[0].content:
            parts = response.candidates[0].content.parts or []
            text_parts: List[str] = []
            tc_list: List[ToolCall] = []

            for part in parts:
                if part.text:
                    text_parts.append(part.text)
                if part.function_call:
                    fc = part.function_call
                    try:
                        tc_list.append(
                            ToolCall(
                                id=fc.id or f"gemini-{int(time.time())}",
                                name=fc.name,
                                arguments=json.dumps(dict(fc.args)) if fc.args else "{}",
                            )
                        )
                    except Exception as parse_err:
                        raise ToolCallError(
                            f"Failed to parse Gemini tool call: {parse_err}",
                            provider=self.PROVIDER_NAME,
                        )

            content = "\n".join(text_parts) if text_parts else None
            tool_calls = tc_list if tc_list else None

        # Extract token usage
        usage: Optional[TokenUsage] = None
        if response.usage_metadata:
            um = response.usage_metadata
            usage = TokenUsage(
                prompt_tokens=getattr(um, "prompt_token_count", None),
                completion_tokens=getattr(um, "candidates_token_count", None),
                total_tokens=getattr(um, "total_token_count", None),
            )

        llm_response = LLMResponse(
            content=content,
            tool_calls=tool_calls,
            usage=usage,
            model=self._model,
            provider=self.PROVIDER_NAME,
            latency_ms=round(latency_ms, 2),
        )

        logger.info(
            f"Gemini response: model={self._model}, latency={latency_ms:.0f}ms, "
            f"tokens={usage.total_tokens if usage else 'N/A'}, "
            f"tool_calls={len(tool_calls) if tool_calls else 0}"
        )
        return llm_response

    def _handle_gemini_error(self, exc: Exception) -> None:
        """Map Google API exceptions to our custom hierarchy and re-raise."""
        exc_str = str(exc).lower()

        # Check common Google API error patterns
        if "401" in exc_str or "403" in exc_str or "api key" in exc_str:
            raise AuthenticationError(str(exc), provider=self.PROVIDER_NAME) from exc
        if "429" in exc_str or "resource exhausted" in exc_str:
            raise RateLimitError(str(exc), provider=self.PROVIDER_NAME) from exc
        if "400" in exc_str or "invalid" in exc_str:
            raise InvalidRequestError(str(exc), provider=self.PROVIDER_NAME) from exc
        if any(code in exc_str for code in ("500", "502", "503", "504", "unavailable")):
            raise ProviderUnavailableError(str(exc), provider=self.PROVIDER_NAME) from exc

        # Catch-all
        raise LLMError(
            f"Unexpected Gemini error: {exc}", provider=self.PROVIDER_NAME
        ) from exc


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

class LLMFactory:
    """
    Factory class that creates the appropriate BaseLLM implementation
    based on the configured provider name.

    Why a factory:
        The agent and API layers should never import GroqClient or
        GeminiClient directly. They call LLMFactory.create() and receive
        a BaseLLM. This keeps provider details encapsulated and makes
        testing easy (you can register a mock provider).

    Usage:
        llm = LLMFactory.create()          # uses settings.llm_provider
        llm = LLMFactory.create("gemini")  # explicit override
    """

    # Registry mapping provider names → client classes
    _registry: Dict[str, Type[BaseLLM]] = {
        "groq": GroqClient,
        "gemini": GeminiClient,
    }

    @classmethod
    def create(cls, provider: Optional[str] = None, **kwargs: Any) -> BaseLLM:
        """
        Create and return a configured LLM client.

        Args:
            provider: Provider name (e.g. "groq", "gemini").
                      Defaults to settings.llm_provider.
            **kwargs: Additional keyword arguments forwarded to the client
                      constructor (e.g. api_key, model).

        Returns:
            A configured BaseLLM instance.

        Raises:
            ValueError: If the provider name is not in the registry.
        """
        name = (provider or settings.llm_provider).lower().strip()

        if name not in cls._registry:
            available = ", ".join(sorted(cls._registry.keys()))
            raise ValueError(
                f"Unknown LLM provider '{name}'. Available: {available}"
            )

        logger.info(f"LLMFactory creating client for provider: '{name}'")
        return cls._registry[name](**kwargs)

    @classmethod
    def register(cls, name: str, client_class: Type[BaseLLM]) -> None:
        """
        Register a new provider at runtime.

        This allows extending the factory without modifying this file —
        useful for plugins, testing mocks, or adding providers like
        OpenAI or Anthropic in the future.

        Args:
            name:          Short name for the provider (e.g. "openai").
            client_class:  A class that inherits from BaseLLM.
        """
        cls._registry[name.lower().strip()] = client_class
        logger.info(f"LLMFactory registered new provider: '{name}'")
