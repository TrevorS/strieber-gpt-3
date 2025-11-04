"""
Universal progress tracking for MCP tools using SQLite backend.

MCP tools run in separate subprocesses and cannot share Python objects with the
parent process. This module provides a SQLite-based progress tracking system that
allows tools to write progress updates that the main process can poll and stream
to the frontend via SSE.

Architecture:
    Tool subprocess → writes progress → SQLite DB
    Main process → polls SQLite → emits SSE events → Frontend

Usage in MCP tools:
    progress = ToolProgress(tool_call_id) if tool_call_id else NullProgress()
    progress.update("searching", "Fetching results...", 50)
    progress.complete("Done!")

Usage in main process:
    # Poll for new progress
    updates = get_progress_since(tool_call_id, last_id)

    # Cleanup after tool completes
    cleanup_progress(tool_call_id)
"""

import sqlite3
import time
from pathlib import Path
from typing import List, Dict, Any, Optional


# Database path (same directory as sessions.db)
DB_PATH = Path(__file__).parent.parent.parent / "data" / "tool_progress.db"


def init_progress_db():
    """Initialize progress tracking database.

    Creates the tool_progress table if it doesn't exist. Should be called
    once during application startup.

    Schema:
        - id: Auto-incrementing primary key
        - tool_call_id: Identifier for the tool execution
        - status: Status keyword (searching, filtering, completed, error, etc.)
        - detail: Human-readable detail message
        - progress_pct: Optional progress percentage (0-100)
        - timestamp: Unix timestamp when progress was recorded
    """
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS tool_progress (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tool_call_id TEXT NOT NULL,
            status TEXT NOT NULL,
            detail TEXT,
            progress_pct INTEGER,
            timestamp REAL NOT NULL
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_tool_call_id
        ON tool_progress(tool_call_id)
    """)
    conn.commit()
    conn.close()


def emit_progress(
    tool_call_id: str,
    status: str,
    detail: str = "",
    progress_pct: Optional[int] = None
):
    """Write progress update to SQLite database.

    This is the low-level function called by ToolProgress. In most cases,
    you should use the ToolProgress class instead of calling this directly.

    Args:
        tool_call_id: Unique identifier for this tool execution
        status: Status keyword (e.g., "searching", "filtering", "completed")
        detail: Human-readable detail message
        progress_pct: Optional progress percentage (0-100)

    Example:
        emit_progress("call_123", "searching", "Query 1/3", 30)
    """
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute(
        """
        INSERT INTO tool_progress (tool_call_id, status, detail, progress_pct, timestamp)
        VALUES (?, ?, ?, ?, ?)
        """,
        (tool_call_id, status, detail, progress_pct, time.time())
    )
    conn.commit()
    conn.close()


def get_progress_since(tool_call_id: str, last_id: int = 0) -> List[Dict[str, Any]]:
    """Retrieve new progress updates for a tool execution.

    Polls the database for progress updates with id > last_id. This is used
    by the main process to check for new progress while a tool is executing.

    Args:
        tool_call_id: Identifier for the tool execution
        last_id: ID of the last progress entry retrieved (default: 0)

    Returns:
        List of progress dictionaries with keys: id, status, detail, progress_pct, timestamp

    Example:
        # Poll every 100ms
        last_id = 0
        while tool_running:
            updates = get_progress_since("call_123", last_id)
            for update in updates:
                emit_sse_event(update)
                last_id = update["id"]
            await asyncio.sleep(0.1)
    """
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row  # Return rows as dictionaries

    cursor = conn.execute(
        """
        SELECT id, status, detail, progress_pct, timestamp
        FROM tool_progress
        WHERE tool_call_id = ? AND id > ?
        ORDER BY id ASC
        """,
        (tool_call_id, last_id)
    )

    results = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return results


def cleanup_progress(tool_call_id: str):
    """Delete all progress entries for a completed tool execution.

    Should be called after a tool completes to prevent database growth.
    Progress is only needed during tool execution.

    Args:
        tool_call_id: Identifier for the tool execution to clean up

    Example:
        try:
            result = await tool.ainvoke(args)
        finally:
            cleanup_progress(tool_call_id)
    """
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute(
        "DELETE FROM tool_progress WHERE tool_call_id = ?",
        (tool_call_id,)
    )
    conn.commit()
    conn.close()


class ToolProgress:
    """Progress tracker for MCP tools using SQLite backend.

    This class provides a convenient interface for tools to report progress.
    Each instance is bound to a specific tool_call_id, so tools don't need
    to pass it with every update.

    Attributes:
        tool_call_id: Unique identifier for this tool execution

    Example:
        @mcp.tool()
        async def web_search(query: str, _tool_call_id: str = None) -> str:
            progress = ToolProgress(_tool_call_id) if _tool_call_id else NullProgress()

            progress.update("searching", f"Searching for: {query}")
            # ... perform search ...

            progress.update("filtering", "Filtering results...", 50)
            # ... filter results ...

            progress.complete(f"Found {len(results)} results")
            return format_results(results)
    """

    def __init__(self, tool_call_id: str):
        """Initialize progress tracker for a specific tool execution.

        Args:
            tool_call_id: Unique identifier for this tool execution
        """
        self.tool_call_id = tool_call_id

    def update(self, status: str, detail: str = "", progress_pct: Optional[int] = None):
        """Update progress with new status.

        Args:
            status: Status keyword (e.g., "searching", "filtering", "completed")
            detail: Human-readable detail about what's happening
            progress_pct: Optional progress percentage (0-100)
        """
        emit_progress(self.tool_call_id, status, detail, progress_pct)

    def complete(self, detail: str = ""):
        """Mark operation as completed.

        Args:
            detail: Optional completion message
        """
        self.update("completed", detail, 100)

    def error(self, error_msg: str):
        """Mark operation as errored.

        Args:
            error_msg: Error message
        """
        self.update("error", error_msg)


class NullProgress:
    """No-op progress tracker (null object pattern).

    Used when progress tracking is not needed (e.g., when tool is called
    directly without a tool_call_id). All methods are no-ops, allowing
    tool code to call progress methods unconditionally.

    Example:
        # Tool code doesn't need to check if progress exists
        def my_tool(param: str, _tool_call_id: str = None):
            progress = ToolProgress(_tool_call_id) if _tool_call_id else NullProgress()

            progress.update("step1", "Doing work...")  # Safe to call
            # ... do work ...

            progress.complete("Done!")  # No defensive check needed
    """

    def update(self, status: str, detail: str = "", progress_pct: Optional[int] = None):
        """No-op update."""
        pass

    def complete(self, detail: str = ""):
        """No-op completion."""
        pass

    def error(self, error_msg: str):
        """No-op error."""
        pass
