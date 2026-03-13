"""
Local agentic loop for structural design.

Runs on your machine, connects to RunPod (or any OpenAI-compatible API)
to generate designs. Handles the multi-turn tool call cycle locally
or delegates to the serverless handler.

Usage:
  python -m agent.cli "Design a 3-bay portal frame, 6m spans, 4m height"
  python -m agent.cli --interactive
  python -m agent.cli --batch specs.yaml --output designs/
"""

from __future__ import annotations
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from deploy.tool_executor import ToolExecutor
from tools.registry import TOOLS_SCHEMA


SYSTEM_PROMPT = (
    "You are a structural engineering assistant. You design structures by "
    "reasoning step-by-step about the requirements and calling tools to compute "
    "exact geometry. You never write coordinates or joint numbers directly — "
    "you delegate all spatial computation to Python tools. "
    "After calling all necessary tools, call assemble_script to produce "
    "a validated STAAD Pro script."
)


class StructuralAgent:
    """Agent that orchestrates LLM + tool calls for structural design."""

    def __init__(
        self,
        api_url: str = "https://api.runpod.ai/v2/{endpoint_id}/runsync",
        api_key: Optional[str] = None,
        endpoint_id: Optional[str] = None,
        max_rounds: int = 10,
        max_tokens: int = 4096,
        use_serverless: bool = True,
    ):
        self.api_url = api_url
        self.api_key = api_key or os.environ.get("RUNPOD_API_KEY", "")
        self.endpoint_id = endpoint_id or os.environ.get("RUNPOD_ENDPOINT_ID", "")
        self.max_rounds = max_rounds
        self.max_tokens = max_tokens
        self.use_serverless = use_serverless
        self.executor = ToolExecutor()

    def design(self, spec: str) -> dict:
        """
        Design a structure from a natural language specification.

        Returns dict with:
            script: The final STAAD Pro script
            validation: Validation results
            messages: Full conversation history
            elapsed: Time taken
        """
        self.executor.reset()

        if self.use_serverless:
            return self._design_serverless(spec)
        else:
            return self._design_local(spec)

    def _design_serverless(self, spec: str) -> dict:
        """Send to RunPod serverless for full pipeline execution."""
        import requests

        url = self.api_url.format(endpoint_id=self.endpoint_id)
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "input": {
                "prompt": spec,
                "max_tool_rounds": self.max_rounds,
                "max_tokens": self.max_tokens,
            }
        }

        start = time.time()
        response = requests.post(url, json=payload, headers=headers, timeout=300)
        response.raise_for_status()
        result = response.json()

        if "error" in result:
            return {"error": result["error"]}

        output = result.get("output", result)
        output["elapsed"] = round(time.time() - start, 2)
        return output

    def _design_local(self, spec: str) -> dict:
        """
        Run the agentic loop locally using OpenAI-compatible API.
        Useful for testing with local vLLM or any OpenAI-compatible endpoint.
        """
        import requests

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": spec},
        ]

        start = time.time()

        for round_idx in range(self.max_rounds):
            # Call LLM
            response = self._call_llm(messages)
            assistant_msg = response

            messages.append(assistant_msg)

            # Check for tool calls
            tool_calls = assistant_msg.get("tool_calls", [])
            if not tool_calls:
                break

            # Execute tools locally
            tool_results = self.executor.execute_tool_calls(tool_calls)
            messages.extend(tool_results)

        elapsed = round(time.time() - start, 2)

        # Extract script
        script = ""
        final = messages[-1].get("content", "") if messages else ""
        if "```" in final:
            s = final.find("```\n") + 4
            e = final.find("\n```", s)
            if s > 3 and e > s:
                script = final[s:e]

        if not script:
            script = self.executor.get_script()

        validation = self.executor.validate_current()

        return {
            "script": script,
            "validation": validation,
            "session": self.executor.get_session_summary(),
            "rounds": round_idx + 1,
            "elapsed": elapsed,
            "messages": messages,
        }

    def _call_llm(self, messages: list[dict]) -> dict:
        """Call OpenAI-compatible API endpoint."""
        import requests

        url = self.api_url.rstrip("/") + "/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": "unsloth/Qwen2.5-72B-Instruct-bnb-4bit",
            "messages": messages,
            "tools": TOOLS_SCHEMA,
            "max_tokens": 2048,
            "temperature": 0.1,
            "stream": True,
        }

        print("\n[AI Thinking...]\n", end="", flush=True)
        response = requests.post(url, json=payload, headers=headers, timeout=300, stream=True)
        response.raise_for_status()

        full_content = ""
        tool_calls = []

        import json
        for line in response.iter_lines():
            if line:
                line = line.decode("utf-8")
                if line.startswith("data: ") and line != "data: [DONE]":
                    data = json.loads(line[6:])
                    delta = data["choices"][0].get("delta", {})

                    if "content" in delta and delta["content"]:
                        content = delta["content"]
                        full_content += content
                        print(content, end="", flush=True)

                    if "tool_calls" in delta and delta["tool_calls"]:
                        for tc_delta in delta["tool_calls"]:
                            index = tc_delta.get("index", 0)
                            while len(tool_calls) <= index:
                                tool_calls.append({"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
                            
                            if "id" in tc_delta and tc_delta["id"]:
                                tool_calls[index]["id"] = tc_delta["id"]
                            if "function" in tc_delta:
                                fn_delta = tc_delta["function"]
                                if "name" in fn_delta and fn_delta["name"]:
                                    tool_calls[index]["function"]["name"] = fn_delta["name"]
                                if "arguments" in fn_delta and fn_delta["arguments"]:
                                    tool_calls[index]["function"]["arguments"] += fn_delta["arguments"]

        print()

        msg = {"role": "assistant"}
        if full_content:
            msg["content"] = full_content
        else:
            msg["content"] = ""
            
        if tool_calls:
            msg["tool_calls"] = tool_calls
            
        return msg
