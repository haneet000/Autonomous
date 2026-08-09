"""
Agent Package

This package implements the autonomous ReAct agent, including the core execution loop,
system prompting, interaction schemas, and research report synthesis.
"""

from app.agent.core import ReActAgent
from app.agent.synthesizer import ReportSynthesizer

__all__ = ["ReActAgent", "ReportSynthesizer"]

