"""
Standalone Page Fetching Verification Script

This script verifies that the web page fetching tool retrieves HTML content,
properly cleans it up, strips unnecessary elements, and limits content length.

Usage:
    python scripts/test_fetch.py --url "https://en.wikipedia.org/wiki/Artificial_intelligence"
"""

import argparse
import sys
from pathlib import Path

# Add project root to path to allow importing app modules when running script directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

# Now import modules
from app.tools.fetch_page_tool import fetch_page_content


def main():
    parser = argparse.ArgumentParser(description="Test the Web Page Fetching tool.")
    parser.add_argument(
        "--url",
        type=str,
        default="https://en.wikipedia.org/wiki/Software_engineering",
        help="URL of the webpage to fetch."
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=2000,
        help="Max character count for display output."
    )
    args = parser.parse_args()

    print("==================================================")
    print(f"Fetching webpage: '{args.url}'")
    print(f"Truncating output visual to {args.max_chars} characters for console readability.")
    print("==================================================")

    content = fetch_page_content(args.url, max_chars=args.max_chars)

    if not content:
        print("Empty content returned.")
        sys.exit(1)

    print("\n--- Extracted Content (Truncated Preview) ---")
    print(content)
    print("---------------------------------------------")
    print(f"\nVerification Complete. Page fetch text length: {len(content)} characters.")


if __name__ == "__main__":
    main()
