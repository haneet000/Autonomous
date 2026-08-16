"""
DuckDuckGo Search Tool

This module implements the web search tool using the duckduckgo-search library.
It performs queries and returns a structured list of results consisting of the
title, URL, and a snippet for each result.

Architectural Decisions:
- Utilizes the `duckduckgo-search` (DDGS) library which provides a simple, API-key-free access to search results.
- Returns data validated by Pydantic models to guarantee type safety and schema consistency.
- Implements graceful exception handling and comprehensive logging to ensure agent resilience.
"""

import logging
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from duckduckgo_search import DDGS

logger = logging.getLogger("research-agent.tools.search")


class SearchResult(BaseModel):
    """
    Model representing a single structured search result.
    """
    title: str = Field(description="The title of the search result page")
    url: str = Field(description="The source URL of the page")
    snippet: str = Field(description="A short summary/snippet of the page content")


def search_web(query: str, max_results: int = 5) -> List[SearchResult]:
    """
    Searches the web using DuckDuckGo for the given query.

    Args:
        query: The search term or question.
        max_results: The maximum number of results to return.

    Returns:
        A list of SearchResult models.
    """
    logger.info(f"Executing web search for query: '{query}' with max_results: {max_results}")
    
    if not query.strip():
        logger.warning("Empty search query provided.")
        return []

    results: List[SearchResult] = []
    
    # 1. Try DuckDuckGo Search (HTML backend first for stability)
    try:
        with DDGS() as ddgs:
            response = list(ddgs.text(query, backend="html", max_results=max_results))
            if not response:
                response = list(ddgs.text(query, backend="lite", max_results=max_results))
            if not response:
                response = list(ddgs.text(query, max_results=max_results))

            for item in response:
                title = item.get("title", "")
                url = item.get("href", "")
                snippet = item.get("body", "")
                if url:
                    results.append(SearchResult(title=title, url=url, snippet=snippet))
    except Exception as ddg_err:
        logger.warning(f"DuckDuckGo search encountered error: {ddg_err}. Falling back to Wikipedia API...")

    # 2. Fallback to Wikipedia API if search returned no results
    if not results:
        try:
            import requests
            import re
            logger.info(f"Using Wikipedia search fallback for query: '{query}'")
            resp = requests.get(
                "https://en.wikipedia.org/w/api.php",
                params={
                    "action": "query",
                    "list": "search",
                    "srsearch": query,
                    "format": "json",
                    "utf8": 1
                },
                headers={"User-Agent": "AutonomousResearchAgent/1.0 (https://github.com/autonomous-agent)"},
                timeout=8
            )
            if resp.status_code == 200:
                data = resp.json()
                for item in data.get("query", {}).get("search", [])[:max_results]:
                    title = item.get("title", "")
                    clean_snippet = re.sub(r"<[^>]+>", "", item.get("snippet", ""))
                    page_url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
                    results.append(SearchResult(title=title, url=page_url, snippet=clean_snippet))
        except Exception as wiki_err:
            logger.warning(f"Wikipedia fallback search error: {wiki_err}")

    if not results:
        logger.info(f"No results found for query: '{query}' with any provider.")
        return []

    logger.info(f"Successfully retrieved {len(results)} search results.")
    return results
