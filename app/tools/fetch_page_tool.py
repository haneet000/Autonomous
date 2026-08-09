"""
Web Page Fetching Tool

This module implements page fetching using the `requests` library to fetch HTML
content and `BeautifulSoup4` to parse, filter, and extract clean plain text.

Architectural Decisions:
- Leverages the synchronous `requests` client with a configurable timeout.
- Standardizes headers (specifically User-Agent) to minimize blocking by web servers.
- Uses BeautifulSoup to strip non-informational components (scripts, stylesheets,
  navigation bars, footers) to maximize text relevance and minimize token counts.
- Implements default safety truncation (e.g., 8,000 characters) to protect the LLM
  context window.
"""

import logging
import requests
from bs4 import BeautifulSoup
from app.config import settings

logger = logging.getLogger("research-agent.tools.fetch_page")

# Common user agent header to prevent simple request blockages
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}


def fetch_page_content(url: str, max_chars: int = 8000) -> str:
    """
    Fetches the content of a webpage and extracts clean, readable text.

    Args:
        url: The absolute HTTP/HTTPS URL of the page to retrieve.
        max_chars: The maximum character count for the output text.

    Returns:
        The extracted clean text or an error message if the fetch failed.
    """
    logger.info(f"Initiating fetch for URL: '{url}'")

    if not url.startswith(("http://", "https://")):
        logger.warning(f"Invalid URL protocol provided: '{url}'")
        return "Error: Invalid URL. Must start with http:// or https://"

    try:
        # Use timeout from global settings
        timeout = settings.request_timeout
        logger.debug(f"Request timeout set to {timeout} seconds")
        
        response = requests.get(url, headers=HEADERS, timeout=timeout)
        
        # Raise HTTPError if response code is 4xx or 5xx
        response.raise_for_status()
        
    except requests.Timeout:
        logger.error(f"Timeout occurred while fetching URL: '{url}'")
        return f"Error: Request timed out after {settings.request_timeout} seconds."
    except requests.RequestException as e:
        logger.error(f"RequestException during fetch for URL: '{url}': {str(e)}")
        return f"Error: Failed to retrieve page. Details: {str(e)}"

    try:
        html_content = response.text
        soup = BeautifulSoup(html_content, "html.parser")

        # Decompose elements that do not contain core article/informational text
        for element in soup(["script", "style", "header", "footer", "nav", "aside", "form", "iframe", "noscript"]):
            element.decompose()

        # Extract text separated by newlines
        raw_text = soup.get_text(separator="\n")

        # Clean up whitespace and empty lines
        lines = (line.strip() for line in raw_text.splitlines())
        # Split multi-whitespace occurrences
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        # Filter out empty lines
        clean_text = "\n".join(chunk for chunk in chunks if chunk)

        # Truncate content to protect context window size limit
        if len(clean_text) > max_chars:
            logger.info(f"Page text exceeds max length ({len(clean_text)} > {max_chars}). Truncating.")
            truncated_msg = f"\n\n[Content truncated to {max_chars} characters...]"
            clean_text = clean_text[:max_chars] + truncated_msg

        logger.info(f"Successfully fetched and cleaned page. Extracted {len(clean_text)} characters.")
        return clean_text

    except Exception as e:
        logger.error(f"HTML Parsing error for URL '{url}': {str(e)}", exc_info=True)
        return f"Error: HTML parsing failed. Details: {str(e)}"
