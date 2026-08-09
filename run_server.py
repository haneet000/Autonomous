"""
Uvicorn Server Launcher

This script launches the Uvicorn ASGI server to host the Autonomous Research Agent API.

Usage:
    python3 run_server.py [--host 0.0.0.0] [--port 8000] [--reload]
"""

import argparse
import sys
import os
import uvicorn

# Ensure project root is in python path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from app.config import settings


def main() -> None:
    parser = argparse.ArgumentParser(description="Start Autonomous Research Agent FastAPI Server.")
    parser.add_argument(
        "--host",
        type=str,
        default=settings.host,
        help=f"Host address to bind (default: {settings.host})."
    )
    parser.add_argument(
        "--port",
        type=int,
        default=settings.port,
        help=f"Port to bind (default: {settings.port})."
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload on code changes."
    )
    args = parser.parse_args()

    print("=" * 70)
    print("🚀 Starting Autonomous Research Agent API Server")
    print(f"   Address: http://{args.host}:{args.port}")
    print(f"   Docs:    http://{args.host}:{args.port}/docs")
    print("=" * 70)

    uvicorn.run(
        "app.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )


if __name__ == "__main__":
    main()
