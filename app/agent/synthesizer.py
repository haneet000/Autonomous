"""
Report Synthesizer Module

This module provides the ReportSynthesizer class, which converts raw research
notes, visited URLs, and agent summaries into a structured, publication-quality
Markdown research report complete with inline citations and a references section.

Architectural Rationale:
- Single Responsibility Principle (SRP): Decouples synthesis and Markdown formatting
  from the ReAct action-observation loop.
- Resilience & Fallbacks: If LLM generation fails or no notes were captured,
  it executes a deterministic fallback generator ensuring valid output is returned.
- Citation Mapping: Automatically maps unique source URLs to numerical footnote
  indices ([1], [2], etc.) for reliable reference tracking.
"""

import logging
from typing import List, Dict, Optional
from app.llm.client import BaseLLM
from app.llm.schemas import Message, Role
from app.agent.schemas import Note
from app.agent.prompts import SYNTHESIS_SYSTEM_PROMPT

logger = logging.getLogger("research-agent.agent.synthesizer")


class ReportSynthesizer:
    """
    Synthesizes research findings into a structured Markdown document using BaseLLM.
    """

    def __init__(self, llm: BaseLLM) -> None:
        """
        Initialize the synthesizer with an LLM provider client.

        Args:
            llm: BaseLLM instance (e.g. Groq or Gemini Flash fallback).
        """
        self.llm = llm

    def synthesize(
        self,
        query: str,
        notes: List[Note],
        visited_urls: List[str],
        raw_summary: Optional[str] = None
    ) -> str:
        """
        Generate a structured Markdown research report.

        Args:
            query: The original user research query.
            notes: List of Note Pydantic models collected during research.
            visited_urls: List of URLs fetched during execution.
            raw_summary: Optional initial finish summary provided by the agent.

        Returns:
            A string containing the full Markdown formatted report.
        """
        logger.info(f"Synthesizing research report for query: '{query}' with {len(notes)} notes.")

        if not notes and not raw_summary:
            logger.warning("No notes or summary available for report synthesis.")
            return self._build_empty_fallback(query)

        # 1. Build Source URL -> Citation Index Mapping
        url_to_citation: Dict[str, int] = {}
        citation_counter = 1

        # Collect unique sources from notes first
        for note in notes:
            url = note.source_url.strip()
            if url and url not in url_to_citation:
                url_to_citation[url] = citation_counter
                citation_counter += 1

        # Collect any remaining visited URLs
        for url in visited_urls:
            url_clean = url.strip()
            if url_clean and url_clean not in url_to_citation:
                url_to_citation[url_clean] = citation_counter
                citation_counter += 1

        # 2. Format Notes and Sources for Prompt
        formatted_notes_lines: List[str] = []
        for idx, note in enumerate(notes, 1):
            cite_num = url_to_citation.get(note.source_url.strip(), "?")
            formatted_notes_lines.append(
                f"Note {idx}: {note.note} (Source [{cite_num}]: {note.source_url})"
            )
        formatted_notes_str = "\n".join(formatted_notes_lines) if formatted_notes_lines else "No specific notes recorded."

        formatted_sources_lines: List[str] = []
        for url, cite_num in sorted(url_to_citation.items(), key=lambda x: x[1]):
            formatted_sources_lines.append(f"[{cite_num}] {url}")
        formatted_sources_str = "\n".join(formatted_sources_lines) if formatted_sources_lines else "No sources recorded."

        # 3. Construct LLM User Prompt
        user_prompt = f"""
Research Query: {query}

Raw Agent Summary:
{raw_summary or "None provided."}

Collected Research Notes:
{formatted_notes_str}

Sources Index:
{formatted_sources_str}

Please generate the final research report in Markdown format following the system instructions. Ensure all citations reference the numerical indices [1], [2], etc., corresponding to the Sources Index above.
"""

        # 4. Invoke LLM Layer
        try:
            response = self.llm.generate(
                prompt=user_prompt,
                system_prompt=SYNTHESIS_SYSTEM_PROMPT,
                temperature=0.3
            )
            report = response.content.strip()

            
            # Ensure the report contains a Sources & References section
            if "Sources & References" not in report and formatted_sources_lines:
                report += "\n\n## Sources & References\n" + formatted_sources_str

            logger.info("Successfully synthesized Markdown research report.")
            return report

        except Exception as e:
            logger.error(f"LLM synthesis failed: {e}. Falling back to deterministic report generator.", exc_info=True)
            return self._build_deterministic_fallback(query, notes, url_to_citation, raw_summary)

    def _build_empty_fallback(self, query: str) -> str:
        """
        Fallback report when no facts were collected.
        """
        return f"""# Research Report: {query}

## Executive Summary
Research was initiated for the query: **{query}**. However, no factual notes or webpage data could be gathered during execution.

## Key Findings
- No facts recorded.

## Sources & References
- No external sources were visited or cited.
"""

    def _build_deterministic_fallback(
        self,
        query: str,
        notes: List[Note],
        url_to_citation: Dict[str, int],
        raw_summary: Optional[str]
    ) -> str:
        """
        Fallback generator when LLM fails. Formats raw notes into clean Markdown.
        """
        lines = [
            f"# Research Report: {query}",
            "",
            "## Executive Summary",
            raw_summary or f"Research report synthesized from {len(notes)} collected findings for the topic: {query}.",
            "",
            "## Key Findings"
        ]

        for note in notes:
            cite_num = url_to_citation.get(note.source_url.strip(), "?")
            lines.append(f"- {note.note} [{cite_num}]")

        lines.extend(["", "## Sources & References"])
        for url, cite_num in sorted(url_to_citation.items(), key=lambda x: x[1]):
            lines.append(f"[{cite_num}] {url}")

        return "\n".join(lines)
