"""
Standalone Search Verification Script

This script verifies that the search tool functions correctly by executing a test query
against DuckDuckGo and logging/printing the structured results.

Usage:
    python scripts/test_search.py --query "Quantum Computing progress"
"""

import argparse
import sys
from pathlib import Path

# Add project root to path to allow importing app modules when running script directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

# Now import modules
from app.config import settings
from app.tools.search_tool import search_web


def main():
    parser = argparse.ArgumentParser(description="Test the DuckDuckGo Search tool.")
    parser.add_argument(
        "--query",
        type=str,
        default="OpenAI GPT-4o architecture and features",
        help="Search query to run."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=3,
        help="Maximum search results to display."
    )
    args = parser.parse_args()

    print("==================================================")
    print(f"Executing search for: '{args.query}' (Limit: {args.limit})")
    print("==================================================")

    results = search_web(args.query, max_results=args.limit)

    if not results:
        print("No results returned. Check internet connection or DDG rate limits.")
        sys.exit(1)

    for i, result in enumerate(results, start=1):
        print(f"\n[{i}] {result.title}")
        print(f"    URL:  {result.url}")
        print(f"    Snippet: {result.snippet}")
        print("-" * 50)

    print(f"\nVerification Complete. Retranslated {len(results)} results successfully.")


if __name__ == "__main__":
    main()
