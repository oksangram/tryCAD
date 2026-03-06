"""
Assembler tool — validates and outputs the final STAAD Pro script.
"""

from __future__ import annotations
from .session import ToolSession


def assemble_script(session: ToolSession, title: str = "Structure",
                    units: str = "METER KN") -> str:
    """
    Assemble all accumulated elements into a validated STAAD Pro script.

    Runs the validation engine and returns the script if valid,
    or returns error messages if validation fails.

    Args:
        session: The active tool session.
        title: Structure title for the header.
        units: Unit system ("METER KN", etc.).

    Returns:
        The complete DSL script text, or an error report.
    """
    script = session.to_script()

    # Run validation
    from validation.validator import validate
    result = validate(script)

    dsl_text = session.to_dsl()

    if result.is_valid:
        return (
            f"SCRIPT ASSEMBLED SUCCESSFULLY\n"
            f"Validation: PASSED ({len(result.warnings)} warnings)\n"
            f"{session.get_summary()}\n"
            f"\n--- SCRIPT ---\n{dsl_text}\n--- END SCRIPT ---"
        )
    else:
        error_list = "\n".join(f"  - {e}" for e in result.errors)
        warning_list = "\n".join(f"  - {w}" for w in result.warnings) if result.warnings else "  None"
        return (
            f"VALIDATION FAILED — {len(result.errors)} error(s)\n"
            f"Errors:\n{error_list}\n"
            f"Warnings:\n{warning_list}\n"
            f"\n--- CURRENT SCRIPT (with errors) ---\n{dsl_text}\n--- END SCRIPT ---"
        )
