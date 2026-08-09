"""
Standalone Memory Verification Script

This script verifies that the SQLite memory database, schema initialization,
and repository functions work correctly.

It:
1. Configures a temporary in-memory or custom verification database.
2. Initializes the database tables.
3. Performs insert operations for job, steps, notes, and visited URLs.
4. Loads the job back from the database and asserts the data matches.

Usage:
    PYTHONPATH=research-agent python3 research-agent/scripts/test_memory.py
"""

import sys
import uuid
from pathlib import Path

# Add project root to path for standalone execution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

# Setup temporary database config for testing
import os
os.environ["DATABASE_PATH"] = "test_research_agent.db"

from app.config import settings
from app.memory.database import init_db
from app.memory import repository

def test_sqlite_memory() -> None:
    print("=" * 60)
    print("  SQLite Memory Layer Verification")
    print("=" * 60)

    # Clean up old test DB if it exists
    test_db = Path(settings.database_path)
    if test_db.exists():
        test_db.unlink()

    # 1. Initialize DB
    print("\n🛠️ Step 1: Initializing database tables...")
    try:
        init_db()
        print("✅ Database successfully initialized.")
    except Exception as e:
        print(f"❌ DB Init failed: {e}")
        sys.exit(1)

    # 2. Insert mock data
    job_id = str(uuid.uuid4())
    query = "Latest developments in Quantum Computing"
    print(f"\n📥 Step 2: Creating job entry (ID: {job_id})")
    repository.create_job(job_id, query)

    print("📥 Step 3: Recording ReAct steps...")
    repository.insert_step(
        job_id=job_id,
        step_num=1,
        thought="First step is to search standard references",
        tool_name="search",
        tool_args={"query": "quantum computing 2026 breakthroughs"},
        observation="Found some articles on topological qubits",
        latency_ms=1200.5
    )
    
    repository.insert_step(
        job_id=job_id,
        step_num=2,
        thought="Let's fetch topological qubits details",
        tool_name="fetch_page",
        tool_args={"url": "https://example.com/quantum"},
        observation="Microsoft quantum program reaches topological milestone",
        latency_ms=850.0
    )

    print("📥 Step 4: Storing research notes...")
    repository.insert_note(
        job_id=job_id,
        note="Microsoft topological quantum computing is reaching 1M physical qubits milestone",
        source_url="https://example.com/quantum"
    )

    print("📥 Step 5: Logging visited URLs...")
    repository.insert_visited_url(job_id, "https://example.com/quantum")
    # Test duplicate URL insertion (should ignore silently)
    repository.insert_visited_url(job_id, "https://example.com/quantum")

    print("📥 Step 6: Completing job and saving summary...")
    repository.update_job(
        job_id=job_id,
        success=True,
        final_summary="Quantum computing has advanced in topological qubit stability.",
        total_tokens=1500,
        total_latency_ms=2050.5
    )

    # 3. Retrieve and verify
    print("\n📤 Step 7: Loading job results from database...")
    run_result = repository.load_job_result(job_id)
    
    if not run_result:
        print("❌ Failed: Loaded job result returned None.")
        sys.exit(1)

    print("✅ Job loaded successfully.")
    
    # 4. Assertions
    print("\n🔍 Step 8: Asserting fields match...")
    try:
        assert run_result.query == query, "Query mismatch"
        assert run_result.success is True, "Success mismatch"
        assert len(run_result.steps) == 2, "Steps count mismatch"
        assert run_result.steps[0].tool_name == "search", "Step tool name mismatch"
        assert run_result.steps[0].tool_args == {"query": "quantum computing 2026 breakthroughs"}, "Step arguments mismatch"
        assert len(run_result.notes) == 1, "Notes count mismatch"
        assert run_result.notes[0].note == "Microsoft topological quantum computing is reaching 1M physical qubits milestone", "Note content mismatch"
        assert len(run_result.visited_urls) == 1, "Visited URLs mismatch (duplicates not ignored or URL missing)"
        assert run_result.total_tokens_used == 1500, "Token count mismatch"
        assert run_result.total_latency_ms == 2050.5, "Latency mismatch"
        print("🎉 ALL ASSERTIONS PASSED!")
    except AssertionError as e:
        print(f"❌ Assertion Failed: {e}")
        sys.exit(1)

    # Cleanup test DB
    if test_db.exists():
        test_db.unlink()
        print("\n🗑️ Cleaned up temporary test database.")

    print("\n✅ Verification complete. SQLite memory layer works flawlessly.")


if __name__ == "__main__":
    test_sqlite_memory()
