"""
Integration Tests for FastAPI REST Endpoints

This module tests HTTP status codes, payload validations, async job submission,
status polling, and report exports.
"""

from app.memory import repository


def test_health_check_endpoint(api_client):
    """Verify GET /api/v1/health returns 200 OK and database connectivity."""
    response = api_client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["database"] == "connected"


def test_missing_job_returns_404(api_client):
    """Verify GET /api/v1/research/{job_id} returns 404 for non-existent job."""
    response = api_client.get("/api/v1/research/non-existent-uuid-12345")
    assert response.status_code == 404
    assert "was not found" in response.json()["detail"]


def test_submit_research_job(api_client):
    """Verify POST /api/v1/research accepts query and returns 202 Accepted."""
    payload = {
        "query": "Quantum Computing Applications in Cryptography",
        "max_iterations": 3
    }
    response = api_client.post("/api/v1/research", json=payload)
    assert response.status_code == 202
    data = response.json()
    assert "job_id" in data
    assert data["status"] == "running"
    assert data["query"] == payload["query"]


def test_job_polling_and_report_export(api_client):
    """Verify polling job detail and downloading raw Markdown report."""
    job_id = "test-api-export-999"
    repository.create_job(job_id, "Sample Crypto Query")
    repository.insert_note(job_id, "Post-quantum algorithms standardized.", "https://example.com/nist")
    repository.update_job(
        job_id,
        success=True,
        final_summary="# Research Report: Post Quantum Cryptography\n\n## Key Findings\n- NIST standards.",
        total_tokens=120,
        total_latency_ms=300.0
    )

    # 1. Detail endpoint
    detail_resp = api_client.get(f"/api/v1/research/{job_id}")
    assert detail_resp.status_code == 200
    data = detail_resp.json()
    assert data["success"] is True
    assert len(data["notes"]) == 1

    # 2. Report export endpoint
    report_resp = api_client.get(f"/api/v1/research/{job_id}/report")
    assert report_resp.status_code == 200
    assert "# Research Report: Post Quantum Cryptography" in report_resp.text


def test_list_jobs_endpoint(api_client):
    """Verify GET /api/v1/jobs pagination."""
    response = api_client.get("/api/v1/jobs?limit=5&offset=0")
    assert response.status_code == 200
    data = response.json()
    assert "total" in data
    assert "jobs" in data
