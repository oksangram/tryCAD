"""
CLI interface for the structural design agent.

Usage:
  # Single design
  python -m agent.cli "Design a 3-bay portal frame, 6m spans, 4m height"

  # Interactive mode
  python -m agent.cli --interactive

  # Batch mode
  python -m agent.cli --batch specs.yaml --output designs/

  # Local mode (requires local vLLM server)
  python -m agent.cli --local --api-url http://localhost:8000/v1 "Design..."
"""

from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.agent import StructuralAgent


def main():
    parser = argparse.ArgumentParser(
        description="Structural DSL Design Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            '  python -m agent.cli "Design a 3-bay portal frame"\n'
            "  python -m agent.cli --interactive\n"
            '  python -m agent.cli --local --api-url http://localhost:8000/v1 "Design..."\n'
        ),
    )
    parser.add_argument("prompt", nargs="?", help="Design specification")
    parser.add_argument("--interactive", "-i", action="store_true",
                        help="Interactive mode")
    parser.add_argument("--batch", type=str, help="YAML file with batch specs")
    parser.add_argument("--output", "-o", type=str, default=".",
                        help="Output directory for generated scripts")
    parser.add_argument("--api-key", type=str,
                        default=os.environ.get("RUNPOD_API_KEY", ""),
                        help="RunPod API key")
    parser.add_argument("--endpoint-id", type=str,
                        default=os.environ.get("RUNPOD_ENDPOINT_ID", ""),
                        help="RunPod endpoint ID")
    parser.add_argument("--api-url", type=str,
                        default="https://api.runpod.ai/v2/{endpoint_id}/runsync",
                        help="API URL (use http://localhost:8000/v1 for local)")
    parser.add_argument("--local", action="store_true",
                        help="Use local OpenAI-compatible API instead of serverless")
    parser.add_argument("--max-rounds", type=int, default=10,
                        help="Maximum tool call rounds")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show detailed output including tool calls")
    args = parser.parse_args()

    agent = StructuralAgent(
        api_url=args.api_url,
        api_key=args.api_key,
        endpoint_id=args.endpoint_id,
        max_rounds=args.max_rounds,
        use_serverless=not args.local,
    )

    if args.interactive:
        _interactive_mode(agent, args)
    elif args.batch:
        _batch_mode(agent, args)
    elif args.prompt:
        result = _design_single(agent, args.prompt, args)
        _save_result(result, args.output, "design")
    else:
        parser.print_help()


def _design_single(agent: StructuralAgent, spec: str, args) -> dict:
    """Run a single design and display results."""
    print(f"\n{'='*60}")
    print("STRUCTURAL DESIGN AGENT")
    print(f"{'='*60}")
    print(f"\nSpec: {spec[:100]}{'...' if len(spec) > 100 else ''}")
    print(f"\nGenerating design...")

    result = agent.design(spec)

    if "error" in result:
        print(f"\n❌ Error: {result['error']}")
        return result

    # Display results
    validation = result.get("validation", {})
    is_valid = validation.get("is_valid", False)
    status = "✅ VALID" if is_valid else "❌ INVALID"

    print(f"\n{status}")
    print(f"Rounds: {result.get('rounds', '?')}")
    print(f"Time:   {result.get('elapsed', '?')}s")

    if result.get("session"):
        s = result["session"]
        print(f"Joints: {s.get('n_joints', 0)}")
        print(f"Members: {s.get('n_members', 0)}")

    if validation.get("errors"):
        print(f"\nErrors:")
        for err in validation["errors"]:
            print(f"  ⚠ {err}")

    if validation.get("warnings"):
        print(f"\nWarnings:")
        for warn in validation["warnings"]:
            print(f"  ⚡ {warn}")

    script = result.get("script", "")
    if script:
        print(f"\n{'─'*60}")
        print("STAAD Pro Script:")
        print(f"{'─'*60}")
        print(script)
        print(f"{'─'*60}")

    if args.verbose and result.get("messages"):
        print(f"\n{'─'*60}")
        print("Full Conversation:")
        print(f"{'─'*60}")
        for msg in result["messages"]:
            role = msg.get("role", "")
            content = msg.get("content", "")[:200]
            tc = msg.get("tool_calls", [])
            print(f"\n[{role.upper()}]")
            if content:
                print(f"  {content}")
            if tc:
                for t in tc:
                    print(f"  → {t['function']['name']}({t['function']['arguments'][:80]}...)")

    return result


def _interactive_mode(agent: StructuralAgent, args):
    """Interactive REPL mode."""
    print(f"\n{'='*60}")
    print("STRUCTURAL DESIGN AGENT — Interactive Mode")
    print(f"{'='*60}")
    print("Type your design specifications. Type 'quit' to exit.\n")

    design_count = 0
    while True:
        try:
            spec = input("Design> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not spec or spec.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        design_count += 1
        result = _design_single(agent, spec, args)
        _save_result(result, args.output, f"design_{design_count}")


def _batch_mode(agent: StructuralAgent, args):
    """Process multiple specs from a YAML file."""
    try:
        import yaml
    except ImportError:
        print("PyYAML required for batch mode: pip install pyyaml")
        sys.exit(1)

    with open(args.batch, "r") as f:
        specs = yaml.safe_load(f)

    if not isinstance(specs, list):
        specs = specs.get("designs", [specs])

    os.makedirs(args.output, exist_ok=True)

    for i, spec_item in enumerate(specs):
        if isinstance(spec_item, str):
            spec = spec_item
            name = f"design_{i + 1}"
        else:
            spec = spec_item.get("spec", spec_item.get("prompt", ""))
            name = spec_item.get("name", f"design_{i + 1}")

        print(f"\n[{i + 1}/{len(specs)}] {name}")
        result = _design_single(agent, spec, args)
        _save_result(result, args.output, name)


def _save_result(result: dict, output_dir: str, name: str):
    """Save design result to file."""
    script = result.get("script", "")
    if not script:
        return

    os.makedirs(output_dir, exist_ok=True)

    # Save STAAD script
    script_path = os.path.join(output_dir, f"{name}.std")
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(script)
    print(f"\nSaved: {script_path}")

    # Save full result as JSON
    result_path = os.path.join(output_dir, f"{name}.json")
    # Remove messages from saved result to keep it small
    save_data = {k: v for k, v in result.items() if k != "messages"}
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
