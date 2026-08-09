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
    try:
        # DDGS client is a context manager that handles connection pools and clean shutdown
        with DDGS() as ddgs:
            # text search retrieves standard search index results
            response = ddgs.text(query, max_results=max_results)
            
            # Fallback to HTML backend if default backend returns empty (common in cloud environments)
            if not response:
                logger.info("Default backend returned no results. Trying HTML backend fallback...")
                response = ddgs.text(query, backend="html", max_results=max_results)
            
            if not response:
                logger.info(f"No results found for query: '{query}' with any backend.")
                return []

            for item in response:
                # duckduckgo-search returns dicts containing:
                # - 'title': title of the result
                # - 'href': URL of the result
                # - 'body': text snippet of the result
                title = item.get("title", "")
                url = item.get("href", "")
                snippet = item.get("body", "")

                # Validate data schema using our Pydantic model
                result_item = SearchResult(
                    title=title,
                    url=url,
                    snippet=snippet
                )
                results.append(result_item)
                
        logger.info(f"Successfully retrieved {len(results)} search results.")
        return results

    except Exception as e:
        logger.error(f"Failed to execute DuckDuckGo search: {str(e)}", exc_info=True)
        # Return empty list in case of failure to maintain agent loop stability (graceful failure)
        return []
