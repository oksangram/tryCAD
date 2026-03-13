"""
Tool executor middleware.

Intercepts tool calls from the LLM response, executes the corresponding
Python tool, and returns results. Manages the ToolSession state across
multi-turn tool calls.
"""

from __future__ import annotations
import json
import sys
import traceback
from pathlib import Path
from typing import Optional

# Ensure project root on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.session import ToolSession
from tools.registry import execute_tool, TOOLS_SCHEMA
from validation.validator import validate


class ToolExecutor:
    """Executes tool calls from LLM responses and manages session state."""

    def __init__(self):
        self.session = ToolSession()
        self.call_history: list[dict] = []

    def reset(self):
        """Reset session for a new design."""
        self.session = ToolSession()
        self.call_history = []

    def execute_tool_calls(self, tool_calls: list[dict]) -> list[dict]:
        """
        Execute a batch of tool calls and return results.

        Args:
            tool_calls: List of tool call dicts from LLM response
                Each has {id, type, function: {name, arguments}}

        Returns:
            List of tool result messages
        """
        results = []
        for tc in tool_calls:
            call_id = tc.get("id", "unknown")
            func = tc.get("function", {})
            tool_name = func.get("name", "")
            args_str = func.get("arguments", "{}")

            try:
                args = json.loads(args_str) if isinstance(args_str, str) else args_str
                result = execute_tool(self.session, tool_name, args)

                self.call_history.append({
                    "id": call_id,
                    "tool": tool_name,
                    "args": args,
                    "result": result,
                    "success": True,
                })

                results.append({
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": result,
                })

            except Exception as e:
                error_msg = f"Error executing {tool_name}: {str(e)}"
                self.call_history.append({
                    "id": call_id,
                    "tool": tool_name,
                    "args": args_str,
                    "result": error_msg,
                    "success": False,
                })
                results.append({
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": error_msg,
                })

        return results

    def get_script(self) -> str:
        """Get the current STAAD Pro script from the session."""
        return self.session.to_dsl()

    def validate_current(self) -> dict:
        """Validate the current session state."""
        script = self.session.to_script()
        result = validate(script)
        return {
            "is_valid": result.is_valid,
            "errors": result.errors,
            "warnings": result.warnings,
        }

    def get_tools_schema(self) -> list[dict]:
        """Get the OpenAI function-calling schema for all tools."""
        return TOOLS_SCHEMA

    def get_session_summary(self) -> dict:
        """Get a summary of the current session state."""
        return {
            "n_joints": len(self.session._joints),
            "n_members": len(self.session._members),
            "n_sections": len(self.session._sections),
            "n_loads": len(self.session._load_cases),
            "n_supports": len(self.session._supports),
            "n_tool_calls": len(self.call_history),
        }
