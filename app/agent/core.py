"""
ReAct Agent Core Loop

This module implements the core ReAct (Reasoning and Action) loop.
It coordinates LLM inference (thought generation and tool selection) with
local tool execution (search, page fetching, note saving, and completion).

Architectural Decisions:
- State Encapsulation: All run-time state is stored on the ReActAgent instance,
  making it thread-safe and reusable.
- Graceful Failures: Tool errors are caught and returned as observations to
  the LLM so it can learn and adapt its strategy rather than crash the agent.
- Duplicate Fetch Prevention: Set-based URL tracking prevents fetching the
  same content twice.
- Format Translation: Converts our unified Pydantic Messages to/from the LLM
  layer seamlessly.
"""

import json
import logging
import time
import uuid
from typing import List, Dict, Any, Optional, Set
from datetime import datetime

from app.config import settings
from app.llm.client import BaseLLM
from app.llm.schemas import Message, Role, ToolCall, LLMResponse
from app.agent.schemas import Note, AgentStep, AgentRunResult
from app.agent.prompts import REACT_SYSTEM_PROMPT
from app.agent.synthesizer import ReportSynthesizer
from app.tools.search_tool import search_web
from app.tools.fetch_page_tool import fetch_page_content
from app.memory.database import init_db
from app.memory import repository

logger = logging.getLogger("research-agent.agent.core")


class ReActAgent:
    """
    Autonomous ReAct (Reasoning and Action) Agent.

    Coordinates the loop:
      1. Prompt LLM with conversation history and tool definitions.
      2. LLM outputs a Thought and requests one or more tool executions (Actions).
      3. Agent executes the requested tools and retrieves the results (Observations).
      4. Agent appends the Action and Observation to the history.
      5. Repeat until the LLM calls the `finish` tool or iterations are exhausted.
    """

    def __init__(self, llm: BaseLLM) -> None:
        """
        Initialize the ReAct agent.

        Args:
            llm: A provider-agnostic BaseLLM client.
        """
        self.llm = llm
        self.synthesizer = ReportSynthesizer(llm)
        # Session state variables
        self.notes: List[Note] = []
        self.visited_urls: Set[str] = set()
        self.steps: List[AgentStep] = []
        self.messages: List[Message] = []
        self.finished = False
        self.final_summary = ""
        self.total_tokens_used = 0
        self.total_latency_ms = 0.0
        self.current_job_id = ""

    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        """
        Return OpenAI-compatible JSON Schema definitions of available tools.
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Search the web using DuckDuckGo for the given query and retrieve a list of pages with titles, URLs, and snippets.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query to execute."
                            }
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "fetch_page",
                    "description": "Download and extract clean plain text from a specific URL. Always use search first to find URLs before calling this.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "url": {
                                "type": "string",
                                "description": "The absolute HTTP/HTTPS URL of the page to retrieve."
                            }
                        },
                        "required": ["url"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "save_note",
                    "description": "Save a key fact, note, or research finding extracted from a webpage source URL to memory.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "note": {
                                "type": "string",
                                "description": "The concise fact or piece of information to save."
                            },
                            "source_url": {
                                "type": "string",
                                "description": "The URL of the webpage where this fact was found."
                            }
                        },
                        "required": ["note", "source_url"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "finish",
                    "description": "Conclude the research process. You must call this tool once you have collected sufficient information to answer the user query.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "final_summary": {
                                "type": "string",
                                "description": "A detailed, factual final summary of your findings. It must synthesize all notes collected and directly answer the query, citing source URLs."
                            }
                        },
                        "required": ["final_summary"]
                    }
                }
            }
        ]

    def run(self, query: str, job_id: Optional[str] = None, max_iterations: Optional[int] = None) -> AgentRunResult:
        """
        Execute the ReAct loop for a given query and persist step traces in SQLite memory.

        Args:
            query:          The research question/topic.
            job_id:         Optional unique research job identifier. Automatically generated if not set.
            max_iterations: Maximum number of cycles before forcing completion.
                            Falls back to settings.max_iterations.

        Returns:
            An AgentRunResult model summarizing the execution.
        """
        limit = max_iterations or settings.max_iterations
        self.current_job_id = job_id or str(uuid.uuid4())
        
        logger.info(f"Starting ReAct Agent run (Job ID: {self.current_job_id}) for query: '{query}' (limit={limit} iterations)")

        # Ensure database and tables exist
        init_db()

        # Create the initial job record in SQLite DB
        repository.create_job(self.current_job_id, query)

        # Reset session state
        self.notes = []
        self.visited_urls = set()
        self.steps = []
        self.finished = False
        self.final_summary = ""
        self.total_tokens_used = 0
        self.total_latency_ms = 0.0

        # Construct initial messages
        self.messages = [
            Message(role=Role.SYSTEM, content=REACT_SYSTEM_PROMPT),
            Message(role=Role.USER, content=f"Research Query: {query}")
        ]

        for step_num in range(1, limit + 1):
            if self.finished:
                break

            logger.info(f"--- ReAct Iteration {step_num}/{limit} ---")
            
            # Step 1: Prompt LLM
            try:
                response = self.llm.chat(
                    messages=self.messages,
                    tools=self.get_tool_definitions(),
                )
            except Exception as e:
                logger.error(f"LLM call failed in ReAct step {step_num}: {e}", exc_info=True)
                final_err = f"Research failed due to LLM error: {str(e)}"
                # Save failure status in DB
                repository.update_job(
                    self.current_job_id,
                    success=False,
                    final_summary=final_err,
                    total_tokens=self.total_tokens_used,
                    total_latency_ms=self.total_latency_ms
                )
                return AgentRunResult(
                    query=query,
                    success=False,
                    final_summary=final_err,
                    steps=self.steps,
                    notes=self.notes,
                    visited_urls=list(self.visited_urls),
                    total_tokens_used=self.total_tokens_used,
                    total_latency_ms=self.total_latency_ms,
                )

            # Accumulate metrics
            if response.usage and response.usage.total_tokens:
                self.total_tokens_used += response.usage.total_tokens
            self.total_latency_ms += response.latency_ms

            thought = response.content or ""
            if thought:
                logger.info(f"Thought: {thought.strip()}")

            # Step 2: Handle Tool Calls (Actions)
            if not response.tool_calls:
                logger.warning("LLM response did not contain any tool call.")
                obs = "Error: You did not call a tool. If you have finished, call the 'finish' tool. Otherwise, use 'search' or 'fetch_page' to continue."
                self.messages.append(Message(role=Role.ASSISTANT, content=thought))
                self.messages.append(Message(role=Role.USER, content=obs))
                
                # Record step locally
                step_obj = AgentStep(
                    step_num=step_num,
                    thought=thought,
                    tool_name="none",
                    tool_args={},
                    observation=obs,
                    latency_ms=response.latency_ms,
                )
                self.steps.append(step_obj)
                
                # Record step in SQLite DB
                repository.insert_step(
                    job_id=self.current_job_id,
                    step_num=step_num,
                    thought=thought,
                    tool_name="none",
                    tool_args={},
                    observation=obs,
                    latency_ms=response.latency_ms
                )
                continue

            # Append LLM's thought and action request to message history
            self.messages.append(
                Message(
                    role=Role.ASSISTANT,
                    content=response.content,
                    tool_calls=response.tool_calls,
                )
            )

            # Execute each tool call
            for tool_call in response.tool_calls:
                tool_name = tool_call.name
                tool_args = self._parse_arguments(tool_call.arguments)

                logger.info(f"Action: Call tool '{tool_name}' with args {tool_args}")
                
                # Step 3: Execute Action
                observation = self._execute_tool(tool_name, tool_args)
                logger.info(f"Observation: {observation[:200]}..." if len(observation) > 200 else f"Observation: {observation}")

                # Append tool observation back to the message history
                self.messages.append(
                    Message(
                        role=Role.TOOL,
                        content=observation,
                        tool_call_id=tool_call.id,
                    )
                )

                # Record step metrics locally
                step_obj = AgentStep(
                    step_num=step_num,
                    thought=thought,
                    tool_name=tool_name,
                    tool_args=tool_args,
                    observation=observation,
                    latency_ms=response.latency_ms,
                )
                self.steps.append(step_obj)

                # Record step in SQLite DB
                repository.insert_step(
                    job_id=self.current_job_id,
                    step_num=step_num,
                    thought=thought,
                    tool_name=tool_name,
                    tool_args=tool_args,
                    observation=observation,
                    latency_ms=response.latency_ms
                )

                if tool_name == "finish":
                    self.finished = True
                    self.final_summary = tool_args.get("final_summary", "No summary provided.")
                    break

        # If the iteration limit was hit without a finish call, set fallback summary flag
        if not self.finished:
            logger.warning(f"Iteration limit ({limit}) reached without explicit 'finish'.")
            self.finished = True

        # Synthesize final Markdown research report from gathered notes and summary
        logger.info("Synthesizing final Markdown research report...")
        self.final_summary = self.synthesizer.synthesize(
            query=query,
            notes=self.notes,
            visited_urls=list(self.visited_urls),
            raw_summary=self.final_summary
        )

        # Update final job summary in SQLite DB
        repository.update_job(
            job_id=self.current_job_id,
            success=True,
            final_summary=self.final_summary,
            total_tokens=self.total_tokens_used,
            total_latency_ms=self.total_latency_ms
        )

        logger.info(f"ReAct Agent run complete. Notes collected: {len(self.notes)}, Visited URLs: {len(self.visited_urls)}")
        
        return AgentRunResult(
            query=query,
            success=True,
            final_summary=self.final_summary,
            steps=self.steps,
            notes=self.notes,
            visited_urls=list(self.visited_urls),
            total_tokens_used=self.total_tokens_used,
            total_latency_ms=self.total_latency_ms,
        )

    # ── Private Tool Execution Router ──────────────────────────────────────

    def _execute_tool(self, name: str, args: Dict[str, Any]) -> str:
        """
        Routes the tool name to the appropriate function.

        Args:
            name: Name of the tool.
            args: Dict of arguments.

        Returns:
            String output of the tool execution.
        """
        try:
            if name == "search":
                query = args.get("query", "").strip()
                if not query:
                    return "Error: Empty search query."
                results = search_web(query)
                if not results:
                    return f"No results found for search query: '{query}'."
                
                # Format search results nicely for LLM context
                formatted = []
                for idx, r in enumerate(results, 1):
                    formatted.append(f"[{idx}] Title: {r.title}\n    URL: {r.url}\n    Snippet: {r.snippet}")
                return "\n\n".join(formatted)

            elif name == "fetch_page":
                url = args.get("url", "").strip()
                if not url:
                    return "Error: URL argument missing."
                
                # Duplicate prevention guardrail
                if url in self.visited_urls:
                    logger.warning(f"Duplicate fetch prevented for URL: {url}")
                    return f"Warning: You have already fetched this URL ({url}). Do not fetch it again. Use the saved notes or try a different URL."
                
                self.visited_urls.add(url)
                # Store visited URL in database
                repository.insert_visited_url(self.current_job_id, url)
                content = fetch_page_content(url)
                return content

            elif name == "save_note":
                note_text = args.get("note", "").strip()
                source_url = args.get("source_url", "").strip()
                
                if not note_text:
                    return "Error: 'note' field cannot be empty."
                if not source_url:
                    return "Error: 'source_url' field cannot be empty."

                note_obj = Note(
                    note=note_text,
                    source_url=source_url,
                    timestamp=datetime.utcnow()
                )
                self.notes.append(note_obj)
                # Store note in database
                repository.insert_note(self.current_job_id, note_text, source_url)
                return f"Success: Note saved. Content: '{note_text}' (Source: {source_url})"

            elif name == "finish":
                final_summary = args.get("final_summary", "").strip()
                if not final_summary:
                    return "Error: 'final_summary' field cannot be empty when calling finish."
                return "Success: Concluding research."

            else:
                return f"Error: Unknown tool '{name}'."

        except Exception as e:
            logger.error(f"Error executing tool '{name}': {e}", exc_info=True)
            return f"Error: Tool execution failed. Details: {str(e)}"

    @staticmethod
    def _parse_arguments(arguments: str) -> Dict[str, Any]:
        """
        Safely parse JSON arguments returned by the LLM.
        """
        if not arguments:
            return {}
        if isinstance(arguments, dict):
            return arguments
        try:
            return json.loads(arguments)
        except Exception as e:
            logger.warning(f"Failed to parse tool arguments JSON: {arguments}. Error: {e}")
            # Try some basic regex or fallback if necessary, but returning empty or dict is standard
            return {"raw_value": arguments}

    def _synthesize_final_result(self, query: str) -> str:
        """
        Fall-back synthesis when the iteration limit is reached.
        Consolidates all notes to build a summary.
        """
        if not self.notes:
            return "Research reached iteration limit. No research notes were saved."
        
        summary_lines = [
            f"Research reached iteration limit without explicit finish for query: '{query}'.",
            "Here is a summary based on collected findings:",
            ""
        ]
        for idx, note in enumerate(self.notes, 1):
            summary_lines.append(f"- {note.note} (Source: {note.source_url})")
            
        return "\n".join(summary_lines)
