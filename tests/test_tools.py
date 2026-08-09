"""
Unit Tests for Web Tools Layer

This module tests the DuckDuckGo search tool and HTML page fetch/clean tool.
"""

from unittest.mock import patch, MagicMock
from app.tools.search_tool import search_web, SearchResult
from app.tools.fetch_page_tool import fetch_page_content


def test_search_empty_query():
    """Verify search returns empty list when given empty query."""
    results = search_web("")
    assert results == []


@patch("app.tools.search_tool.DDGS")
def test_search_web_success(mock_ddgs_cls):
    """Verify search tool returns formatted SearchResult objects."""
    mock_instance = MagicMock()
    mock_instance.text.return_value = [
        {
            "title": "Python 3.12 Release Notes",
            "href": "https://docs.python.org/3.12/",
            "body": "Python 3.12 introduces new language features."
        }
    ]
    mock_ddgs_cls.return_value.__enter__.return_value = mock_instance

    results = search_web("Python 3.12")
    assert len(results) == 1
    assert isinstance(results[0], SearchResult)
    assert results[0].title == "Python 3.12 Release Notes"
    assert results[0].url == "https://docs.python.org/3.12/"
    assert "new language features" in results[0].snippet


def test_fetch_page_invalid_url():
    """Verify fetch_page returns an error message on invalid URL format."""
    result = fetch_page_content("ftp://invalid-schema-url")
    assert result.startswith("Error:")


@patch("app.tools.fetch_page_tool.requests.get")
def test_fetch_page_html_stripping(mock_get):
    """Verify HTML cleaning strips script, style, nav tags and returns clean text."""
    mock_response = MagicMock()
    mock_response.text = """
    <html>
        <head>
            <style>body { background: red; }</style>
            <script>alert('hack');</script>
        </head>
        <body>
            <nav><a href="#">Home</a></nav>
            <h1>Artificial Intelligence Progress</h1>
            <p>AI models have achieved major milestones in science and coding.</p>
        </body>
    </html>
    """
    mock_response.raise_for_status.return_value = None
    mock_get.return_value = mock_response

    text = fetch_page_content("https://example.com/ai-news")
    assert "Artificial Intelligence Progress" in text
    assert "AI models have achieved major milestones" in text
    assert "<script>" not in text
    assert "<style>" not in text
    assert "alert('hack')" not in text
