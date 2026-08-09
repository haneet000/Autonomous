# ==============================================================================
# Production Dockerfile — Autonomous Research Agent
# Multi-stage build for minimal image size and enhanced security
# ==============================================================================

# ── Stage 1: Build dependencies ──────────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /app

# Install build tools if necessary
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies into a isolated prefix
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Stage 2: Production runtime image ─────────────────────────────────────────
FROM python:3.12-slim AS runner

# Create a non-root system user and group for security
RUN groupadd -r appgroup && useradd -r -g appgroup -d /app -s /sbin/nologin appuser

WORKDIR /app

# Copy pre-compiled dependencies from builder stage
COPY --from=builder /install /usr/local

# Copy application source code
COPY . /app

# Set ownership to non-root user
RUN chown -R appuser:appgroup /app

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000 \
    HOST=0.0.0.0

# Switch to non-root user
USER appuser

# Expose HTTP API port
EXPOSE 8000

# Native Python container healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Command to launch Uvicorn REST API server
CMD ["python3", "run_server.py", "--host", "0.0.0.0", "--port", "8000"]
