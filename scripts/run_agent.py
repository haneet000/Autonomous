"""
CLI ReAct Agent Runner

This script provides a command-line interface to execute the autonomous ReAct Research Agent.
It parses arguments, sets up the LLM provider, instantiates the agent, and prints
each step of the Thought-Action-Observation loop with visual formatting.

Usage:
    PYTHONPATH=research-agent python3 research-agent/scripts/run_agent.py --query "Latest status of Gemini 2.0 Flash features"
"""

import argparse
import sys
from pathlib import Path

# Add project root to path for standalone execution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from app.config import settings  # noqa: E402 — initialises logging
from app.llm.client import LLMFactory  # noqa: E402
from app.agent.core import ReActAgent  # noqa: E402
from app.llm.exceptions import LLMError  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the ReAct Research Agent.")
    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="The research query or topic to investigate.",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default=None,
        help="LLM provider (groq, gemini). Defaults to LLM_PROVIDER in .env.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help="Maximum loop iterations. Defaults to MAX_ITERATIONS in .env.",
    )
    parser.add_argument(
        "--job-id",
        type=str,
        default=None,
        help="Unique identifier for the research job.",
    )
    args = parser.parse_args()

    provider_name = args.provider or settings.llm_provider
    iterations = args.max_iterations or settings.max_iterations

    print("=" * 70)
    print(f"🚀 Starting ReAct Agent Run")
    print(f"   Query:      \"{args.query}\"")
    print(f"   Provider:   {provider_name}")
    print(f"   Max Steps:  {iterations}")
    if args.job_id:
        print(f"   Job ID:     {args.job_id}")
    print("=" * 70)

    # 1. Initialize client
    try:
        llm = LLMFactory.create(provider=args.provider)
    except (LLMError, ValueError) as err:
        print(f"\n❌ Failed to create LLM client: {err}")
        sys.exit(1)

    # 2. Initialize Agent
    agent = ReActAgent(llm=llm)

    # 3. Execute Loop
    try:
        result = agent.run(query=args.query, job_id=args.job_id, max_iterations=args.max_iterations)
    except Exception as err:
        print(f"\n❌ Execution failed: {err}")
        sys.exit(1)

    # 4. Display Results
    print("\n" + "=" * 70)
    print("🏁 Execution Summary")
    print("=" * 70)
    print(f"Job ID:            {agent.current_job_id}")
    print(f"Status:            {'✅ Completed' if result.success else '❌ Failed'}")
    print(f"Steps Taken:       {len(result.steps)}")
    print(f"Notes Collected:   {len(result.notes)}")
    print(f"Visited URLs:      {len(result.visited_urls)}")
    print(f"Total Tokens:      {result.total_tokens_used}")
    print(f"Total Latency:     {result.total_latency_ms:.0f}ms")
    print("=" * 70)

    print("\n📝 Notes Collected:")
    for i, note in enumerate(result.notes, 1):
        print(f"  {i}. {note.note}")
        print(f"     Source: {note.source_url}")
        print("-" * 50)

    print("\n🌐 Visited Pages:")
    for url in result.visited_urls:
        print(f"  - {url}")

    print("\n📜 Final Synthesized Markdown Report:")
    print("─" * 70)
    print(result.final_summary)
    print("─" * 70)


if __name__ == "__main__":
    main()

