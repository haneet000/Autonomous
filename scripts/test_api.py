"""
FastAPI REST API Verification Script

This script verifies the REST API endpoints using FastAPI's TestClient.

Tests Performed:
1. Health check endpoint (GET /api/v1/health).
2. Non-existent job lookup error handling (404 Not Found).
3. Research job submission (POST /api/v1/research) -> 202 Accepted.
4. Job detail retrieval (GET /api/v1/research/{job_id}).
5. Job listing endpoint (GET /api/v1/jobs).
6. Markdown report endpoint (GET /api/v1/research/{job_id}/report).

Usage:
    python3 scripts/test_api.py
"""

import sys
import os
import logging
from fastapi.testclient import TestClient

# Add project root to python path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from app.main import app
from app.memory import repository
from app.memory.database import init_db

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("test_api")


def run_all_tests():
    # Explicitly initialize database for tests
    init_db()

    with TestClient(app) as client:
        # Test 1: Health Check
        logger.info("=== Test 1: Health Check Endpoint ===")
        response = client.get("/api/v1/health")
        logger.info(f"Response status: {response.status_code}, body: {response.json()}")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["database"] == "connected"
        logger.info("✅ Test 1 Passed: Health check endpoint works cleanly.\n")

        # Test 2: 404 Error Handling
        logger.info("=== Test 2: 404 Error Handling for Missing Job ===")
        fake_id = "00000000-0000-0000-0000-000000000000"
        response = client.get(f"/api/v1/research/{fake_id}")
        logger.info(f"Response status: {response.status_code}, body: {response.json()}")
        assert response.status_code == 404
        assert "was not found" in response.json()["detail"]
        logger.info("✅ Test 2 Passed: 404 handled cleanly.\n")

        # Test 3: Async Job Submission & Polling
        logger.info("=== Test 3: Job Submission & Polling ===")
        payload = {
            "query": "Impact of Artificial Intelligence on Modern Software Engineering",
            "max_iterations": 2
        }
        response = client.post("/api/v1/research", json=payload)
        logger.info(f"POST Response status: {response.status_code}, body: {response.json()}")
        assert response.status_code == 202
        data = response.json()
        job_id = data["job_id"]
        assert data["status"] == "running"
        assert data["query"] == payload["query"]

        # Poll status immediately
        poll_resp = client.get(f"/api/v1/research/{job_id}")
        logger.info(f"GET Poll status: {poll_resp.status_code}")
        assert poll_resp.status_code == 200
        poll_data = poll_resp.json()
        assert poll_data["query"] == payload["query"]
        logger.info("✅ Test 3 Passed: Async job creation and polling verified.\n")

        # Test 4: Job Listing & Report Download
        logger.info("=== Test 4: Job Listing & Report Download ===")
        mock_id = "test-completed-job-12345"
        repository.create_job(mock_id, "Sample Research Query")
        repository.insert_note(mock_id, "Sample note 1", "https://example.com/source1")
        repository.update_job(
            mock_id,
            success=True,
            final_summary="# Research Report: Sample\n\n## Key Findings\n- Note 1 [1]\n\n## Sources & References\n[1] https://example.com/source1",
            total_tokens=150,
            total_latency_ms=1200.0
        )

        detail_resp = client.get(f"/api/v1/research/{mock_id}")
        assert detail_resp.status_code == 200
        assert detail_resp.json()["success"] is True

        report_resp = client.get(f"/api/v1/research/{mock_id}/report")
        logger.info(f"Report endpoint status: {report_resp.status_code}")
        assert report_resp.status_code == 200
        assert "# Research Report: Sample" in report_resp.text

        list_resp = client.get("/api/v1/jobs?limit=10")
        assert list_resp.status_code == 200
        list_data = list_resp.json()
        assert list_data["total"] >= 1
        logger.info("✅ Test 4 Passed: Listing and report download verified.\n")


if __name__ == "__main__":
    logger.info("Starting FastAPI REST API Verification...")
    run_all_tests()
    logger.info("🎉 All FastAPI REST API tests completed successfully!")
