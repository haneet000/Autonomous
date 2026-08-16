"""
Configuration Module

This module defines the configuration settings for the Autonomous Research Agent
using Pydantic BaseSettings. It loads environment variables from the project .env file
and handles validation, type coercion, and setting up the central logging system.

Architectural Decisions:
- Pydantic Settings is used to parse, validate, and enforce types for configuration inputs.
- A central configuration object is created and shared as a singleton.
- Central logging configuration is established here to ensure uniform formatting across all modules.
"""

import logging
import os
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Define the root of the project to locate the .env file reliably
BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """
    Application Settings schema. Loads from environment variables or a local .env file.
    """
    model_config = SettingsConfigDict(
        env_file=os.path.join(BASE_DIR, ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # API Keys (leave as empty string if not set, handled dynamically or via validation)
    groq_api_key: str = Field(default="", env="GROQ_API_KEY")
    gemini_api_key: str = Field(default="", env="GEMINI_API_KEY")

    # LLM Configuration
    llm_provider: str = Field(default="groq", env="LLM_PROVIDER")
    default_model: str = Field(default="llama-3.1-8b-instant", env="DEFAULT_MODEL")
    gemini_model: str = Field(default="gemini-2.5-flash", env="GEMINI_MODEL")
    temperature: float = Field(default=0.0, env="TEMPERATURE")
    max_tokens: int = Field(default=4096, env="MAX_TOKENS")
    max_retries: int = Field(default=3, env="MAX_RETRIES")

    # Guardrails and settings
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    max_iterations: int = Field(default=10, env="MAX_ITERATIONS")
    request_timeout: int = Field(default=15, env="REQUEST_TIMEOUT")
    database_path: str = Field(default="research_agent.db", env="DATABASE_PATH")

    # Server configs
    port: int = Field(default=8000, env="PORT")
    host: str = Field(default="0.0.0.0", env="HOST")


# Instantiate settings singleton
settings = Settings()

# Setup logging configuration dynamically based on settings
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

logger = logging.getLogger("research-agent")
logger.info("Configuration successfully loaded.")
