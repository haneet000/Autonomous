"""
SQLite Database Initializer and Connection Context Manager

This module handles SQLite database setup, table creation, and transactional
connections.

Architectural Decisions:
- Context Manager: Handles connection lifecycle. Commits on successful exit,
  rolls back on exceptions, and always closes the connection.
- Pragmatic Foreign Keys: SQLite disables foreign key enforcement by default.
  We explicitly run `PRAGMA foreign_keys = ON;` on every connection opened.
- Schema Setup: Creates the tables for research jobs, steps, notes, and visited URLs.
"""

import sqlite3
import logging
from contextlib import contextmanager
from typing import Generator
from app.config import settings

logger = logging.getLogger("research-agent.memory.database")


@contextmanager
def get_db_connection() -> Generator[sqlite3.Connection, None, None]:
    """
    Context manager to obtain a connection to the SQLite database.

    Automatically handles commits, rollbacks on error, and connection closing.
    Enforces foreign keys explicitly.
    """
    # Use BASE_DIR path to locate the db relative to project root if it's a relative path
    from app.config import BASE_DIR
    import os
    db_path = settings.database_path
    if not os.path.isabs(db_path):
        db_path = os.path.join(BASE_DIR, db_path)

    # Ensure the parent directory exists (critical for Docker volumes like /app/data)
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)

    conn = sqlite3.connect(db_path)
    # Enable dict factory to easily map rows to dicts
    conn.row_factory = sqlite3.Row
    
    try:
        # Enforce foreign key constraints
        conn.execute("PRAGMA foreign_keys = ON;")
        yield conn
        conn.commit()
    except Exception as e:
        logger.error(f"Database transaction error: {e}. Rolling back.", exc_info=True)
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    """
    Initializes the SQLite database schema if tables do not already exist.
    """
    logger.info("Initializing SQLite database tables...")
    
    create_jobs_table = """
    CREATE TABLE IF NOT EXISTS research_jobs (
        job_id TEXT PRIMARY KEY,
        query TEXT NOT NULL,
        success INTEGER DEFAULT 0,
        final_summary TEXT,
        total_tokens INTEGER DEFAULT 0,
        total_latency_ms REAL DEFAULT 0.0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """
    
    create_steps_table = """
    CREATE TABLE IF NOT EXISTS agent_steps (
        step_id INTEGER PRIMARY KEY AUTOINCREMENT,
        job_id TEXT NOT NULL,
        step_num INTEGER NOT NULL,
        thought TEXT,
        tool_name TEXT NOT NULL,
        tool_args TEXT NOT NULL, -- JSON string
        observation TEXT NOT NULL,
        latency_ms REAL NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (job_id) REFERENCES research_jobs (job_id) ON DELETE CASCADE
    );
    """
    
    create_notes_table = """
    CREATE TABLE IF NOT EXISTS research_notes (
        note_id INTEGER PRIMARY KEY AUTOINCREMENT,
        job_id TEXT NOT NULL,
        note TEXT NOT NULL,
        source_url TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (job_id) REFERENCES research_jobs (job_id) ON DELETE CASCADE
    );
    """
    
    create_visited_urls_table = """
    CREATE TABLE IF NOT EXISTS visited_urls (
        url_id INTEGER PRIMARY KEY AUTOINCREMENT,
        job_id TEXT NOT NULL,
        url TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (job_id) REFERENCES research_jobs (job_id) ON DELETE CASCADE,
        UNIQUE (job_id, url)
    );
    """

    try:
        with get_db_connection() as conn:
            conn.execute(create_jobs_table)
            conn.execute(create_steps_table)
            conn.execute(create_notes_table)
            conn.execute(create_visited_urls_table)
        logger.info("Database tables initialized successfully.")
    except Exception as e:
        logger.critical(f"Failed to initialize database: {e}", exc_info=True)
        raise
