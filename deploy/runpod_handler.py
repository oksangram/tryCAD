"""
RunPod Serverless handler.

Handles incoming requests, runs the agentic loop (LLM → tool calls → results),
and returns the final validated STAAD Pro script.

Deploy with:
  runpodctl deploy --handler deploy/runpod_handler.py
"""

from __future__ import annotations
import json
import os
import sys
import time
from pathlib import Path

# Ensure project root on path
sys.path.insert(0, str(Path(__file__).parent.parent))

import runpod
from deploy.tool_executor import ToolExecutor
from tools.registry import TOOLS_SCHEMA

# ── Configuration ──
MODEL_PATH = os.environ.get("MODEL_PATH", "/models/structural_dsl_awq")
MAX_TOOL_ROUNDS = int(os.environ.get("MAX_TOOL_ROUNDS", "10"))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "4096"))

# System prompt
SYSTEM_PROMPT = (
    "You are a structural engineering assistant. You design structures by "
    "reasoning step-by-step about the requirements and calling tools to compute "
    "exact geometry. You never write coordinates or joint numbers directly — "
    "you delegate all spatial computation to Python tools. "
    "After calling all necessary tools, call assemble_script to produce "
    "a validated STAAD Pro script."
)

# ── Global model (loaded once at cold start) ──
engine = None


def load_model():
    """Load vLLM engine at cold start."""
    global engine
    if engine is not None:
        return

    from vllm import LLM, SamplingParams

    engine = LLM(
        model=MODEL_PATH,
        quantization="awq",
        dtype="half",
        max_model_len=8192,
        gpu_memory_utilization=0.90,
        enable_prefix_caching=True,
    )
    print(f"Model loaded from {MODEL_PATH}")


def handler(job: dict) -> dict:
    """
    RunPod serverless handler.

    Input:
        {
            "input": {
                "prompt": "Design a 3-bay portal frame...",
                "max_tool_rounds": 10,  # optional
                "max_tokens": 4096,     # optional
            }
        }

    Output:
        {
            "script": "STAAD SPACE\n...\nFINISH",
            "validation": {"is_valid": true, ...},
            "tool_calls": [...],
            "messages": [...],
        }
    """
    load_model()

    job_input = job.get("input", {})
    user_prompt = job_input.get("prompt", "")
    max_rounds = job_input.get("max_tool_rounds", MAX_TOOL_ROUNDS)
    max_tokens = job_input.get("max_tokens", MAX_TOKENS)

    if not user_prompt:
        return {"error": "No prompt provided"}

    # Initialize
    executor = ToolExecutor()
    tools_schema = executor.get_tools_schema()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    start_time = time.time()
    total_tool_calls = 0

    # ── Agentic loop ──
    for round_idx in range(max_rounds):
        # Generate response
        response = _generate(messages, tools_schema, max_tokens)

        assistant_msg = response["message"]
        messages.append(assistant_msg)

        # Check if there are tool calls
        tool_calls = assistant_msg.get("tool_calls", [])

        if not tool_calls:
            # No tool calls → final response
            break

        # Execute tool calls
        tool_results = executor.execute_tool_calls(tool_calls)
        messages.extend(tool_results)
        total_tool_calls += len(tool_calls)

    elapsed = time.time() - start_time

    # Extract final script
    script = ""
    final_content = messages[-1].get("content", "") if messages else ""
    if "```" in final_content:
        start = final_content.find("```\n") + 4
        end = final_content.find("\n```", start)
        if start > 3 and end > start:
            script = final_content[start:end]

    # If no script in final message, try to get it from session
    if not script:
        script = executor.get_script()

    # Validate
    validation = executor.validate_current()

    return {
        "script": script,
        "validation": validation,
        "session_summary": executor.get_session_summary(),
        "tool_calls_count": total_tool_calls,
        "rounds": round_idx + 1,
        "elapsed_seconds": round(elapsed, 2),
        "messages": messages,
    }


def _generate(messages: list[dict], tools: list[dict], max_tokens: int) -> dict:
    """
    Generate a response from the LLM.

    Uses vLLM with the Qwen2.5 chat template. In a full deployment,
    this would use the OpenAI-compatible API endpoint.
    """
    from vllm import SamplingParams

    # Format messages using the chat template
    # vLLM handles this internally when using the chat API
    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=0.1,
        top_p=0.95,
        stop=["<|im_end|>"],
    )

    # For serverless, we use the engine directly
    prompt = _format_messages_to_prompt(messages)
    outputs = engine.generate([prompt], sampling_params)

    generated_text = outputs[0].outputs[0].text.strip()

    # Parse tool calls from response
    message = _parse_response(generated_text)

    return {"message": message}


def _format_messages_to_prompt(messages: list[dict]) -> str:
    """Format messages into Qwen2.5 chat template string."""
    parts = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")

        if role == "system":
            parts.append(f"<|im_start|>system\n{content}<|im_end|>")
        elif role == "user":
            parts.append(f"<|im_start|>user\n{content}<|im_end|>")
        elif role == "assistant":
            tc = msg.get("tool_calls", [])
            if tc:
                tc_parts = []
                for t in tc:
                    name = t["function"]["name"]
                    args = t["function"]["arguments"]
                    if isinstance(args, str):
                        args = json.loads(args)
                    tc_parts.append(
                        f'<tool_call>\n{{"name": "{name}", "arguments": {json.dumps(args)}}}\n</tool_call>'
                    )
                tool_str = "\n".join(tc_parts)
                parts.append(f"<|im_start|>assistant\n{content}\n{tool_str}<|im_end|>")
            else:
                parts.append(f"<|im_start|>assistant\n{content}<|im_end|>")
        elif role == "tool":
            parts.append(f"<|im_start|>tool\n{content}<|im_end|>")

    # Add generation prompt
    parts.append("<|im_start|>assistant\n")

    return "\n".join(parts)


def _parse_response(text: str) -> dict:
    """Parse LLM response text into a message dict with optional tool calls."""
    tool_calls = []
    content = text

    # Check for tool calls in the response
    if "<tool_call>" in text:
        import re
        # Extract tool calls
        tc_pattern = r'<tool_call>\s*(\{.*?\})\s*</tool_call>'
        matches = re.findall(tc_pattern, text, re.DOTALL)

        for i, match in enumerate(matches):
            try:
                tc_data = json.loads(match)
                tool_calls.append({
                    "id": f"call_{i + 1}",
                    "type": "function",
                    "function": {
                        "name": tc_data.get("name", ""),
                        "arguments": json.dumps(tc_data.get("arguments", {})),
                    },
                })
            except json.JSONDecodeError:
                continue

        # Remove tool call blocks from content
        content = re.sub(tc_pattern, "", text, flags=re.DOTALL).strip()

    message = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls

    return message


# ── RunPod entry point ──
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
