"""
FastAPI Application Entrypoint

This module constructs the main FastAPI application instance, configures CORS
middleware, registers API routers, and manages application startup lifecycle events.

Architectural Decisions:
- Lifespan Context Manager: Ensures database schema is initialized on server startup.
- CORS Middleware: Permits cross-origin requests from frontend dashboards.
- OpenAPI Documentation: Exposes interactive Swagger UI at /docs and ReDoc at /redoc.
"""

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from app.config import settings
from app.memory.database import init_db
from app.api.routes import router as api_router

logger = logging.getLogger("research-agent.main")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan manager handling startup and shutdown procedures.
    """
    logger.info("Initializing Autonomous Research Agent API Server...")
    init_db()
    logger.info("Database schema initialized. Application startup complete.")
    yield
    logger.info("Shutting down Autonomous Research Agent API Server...")


app = FastAPI(
    title="Autonomous Research Agent API",
    description=(
        "Production-ready REST API for the Autonomous Research Agent. "
        "Supports non-blocking async research execution, step-by-step trace polling, "
        "note storage, and publication-ready Markdown report synthesis."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for frontend clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API router
app.include_router(api_router)


@app.get("/", include_in_schema=False)
def root_redirect() -> RedirectResponse:
    """
    Redirect root path requests directly to interactive OpenAPI documentation (/docs).
    """
    return RedirectResponse(url="/docs")
