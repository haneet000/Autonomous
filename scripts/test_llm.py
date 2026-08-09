"""
Standalone LLM Verification Script

This script verifies the LLM abstraction layer by:
    1. Instantiating the configured (or overridden) provider via LLMFactory.
    2. Sending a simple prompt.
    3. Printing the response, latency, provider, model, and token usage.

Usage:
    # Use default provider from .env (LLM_PROVIDER)
    PYTHONPATH=research-agent python3 research-agent/scripts/test_llm.py

    # Override provider explicitly
    PYTHONPATH=research-agent python3 research-agent/scripts/test_llm.py --provider gemini

    # Custom prompt
    PYTHONPATH=research-agent python3 research-agent/scripts/test_llm.py --prompt "Explain SOLID principles"
"""

import argparse
import sys
from pathlib import Path

# Add project root to path for standalone execution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from app.config import settings  # noqa: E402 — initialises logging
from app.llm.client import LLMFactory  # noqa: E402
from app.llm.exceptions import LLMError  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Test the LLM abstraction layer.")
    parser.add_argument(
        "--provider",
        type=str,
        default=None,
        help="LLM provider to use (groq, gemini). Defaults to LLM_PROVIDER in .env.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="What are three key principles of clean code? Answer concisely.",
        help="Prompt to send to the LLM.",
    )
    parser.add_argument(
        "--system",
        type=str,
        default="You are a helpful senior software engineer.",
        help="System prompt.",
    )
    args = parser.parse_args()

    provider_name = args.provider or settings.llm_provider

    print("=" * 60)
    print(f"  LLM Verification Script")
    print(f"  Provider: {provider_name}")
    print("=" * 60)

    # ── Step 1: Create the LLM client ────────────────────────────────
    try:
        llm = LLMFactory.create(provider=args.provider)
    except (LLMError, ValueError) as err:
        print(f"\n❌ Failed to create LLM client: {err}")
        sys.exit(1)

    print(f"\n✅ LLM client created: {type(llm).__name__}")

    # ── Step 2: Send a prompt ────────────────────────────────────────
    print(f"\n📤 Prompt: \"{args.prompt}\"")
    print(f"   System: \"{args.system}\"")

    try:
        response = llm.generate(
            prompt=args.prompt,
            system_prompt=args.system,
        )
    except LLMError as err:
        print(f"\n❌ LLM call failed: {err}")
        sys.exit(1)

    # ── Step 3: Display results ──────────────────────────────────────
    print("\n" + "─" * 60)
    print("📥 Response:")
    print("─" * 60)
    print(response.content or "(no text content)")
    print("─" * 60)

    print(f"\n📊 Metadata:")
    print(f"   Provider:   {response.provider}")
    print(f"   Model:      {response.model}")
    print(f"   Latency:    {response.latency_ms:.0f}ms")

    if response.usage:
        print(f"   Tokens:")
        print(f"     Prompt:     {response.usage.prompt_tokens}")
        print(f"     Completion: {response.usage.completion_tokens}")
        print(f"     Total:      {response.usage.total_tokens}")
    else:
        print(f"   Tokens:     N/A (provider did not return usage)")

    if response.tool_calls:
        print(f"\n🔧 Tool Calls: {len(response.tool_calls)}")
        for tc in response.tool_calls:
            print(f"     - {tc.name}({tc.arguments})")

    print(f"\n✅ Verification complete.")


if __name__ == "__main__":
    main()
