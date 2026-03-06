"""
STAAD Pro DSL Parser.

Reads DSL text and produces an AST (Script object) using Lark.
Also provides error reporting with line numbers.
"""

from __future__ import annotations
from pathlib import Path
from lark import Lark, Transformer, v_args, UnexpectedInput
from typing import Optional

from .ast_nodes import (
    Script, Joint, Member, SectionAssignment, Support, SupportType,
    MemberLoad, JointLoad, SelfWeight, LoadCase, LoadCombination,
    LoadType, LoadDirection, StructureType,
)


GRAMMAR_PATH = Path(__file__).parent / "grammar.lark"


class ParseError(Exception):
    """Raised when the DSL text cannot be parsed."""
    def __init__(self, message: str, line: Optional[int] = None, column: Optional[int] = None):
        self.line = line
        self.column = column
        location = ""
        if line is not None:
            location = f" (line {line}"
            if column is not None:
                location += f", col {column}"
            location += ")"
        super().__init__(f"{message}{location}")


# ──────────────────────────────────────────────────────────────
# Lark Transformer: Parse Tree → AST
# ──────────────────────────────────────────────────────────────

class StaadTransformer(Transformer):
    """Transforms Lark parse tree into our AST node objects."""

    def __init__(self):
        super().__init__()
        self._script = Script()

    # ── Header ──

    def staad_line(self, items):
        self._script.structure_type = StructureType(str(items[0]))

    def input_width_line(self, items):
        self._script.input_width = int(items[0])

    def unit_line(self, items):
        self._script.unit_length = str(items[0])
        self._script.unit_force = str(items[1])

    def header_section(self, items):
        pass

    # ── Joints ──

    def joint_line(self, items):
        jid, x, y, z = items[0], items[1], items[2], items[3]
        self._script.joints.append(Joint(
            id=int(jid), x=float(x), y=float(y), z=float(z)
        ))

    def joint_section(self, items):
        pass

    # ── Members ──

    def member_line(self, items):
        mid, start, end = items[0], items[1], items[2]
        self._script.members.append(Member(
            id=int(mid), start_joint=int(start), end_joint=int(end)
        ))

    def member_section(self, items):
        pass

    # ── Shared: list of integer IDs ──

    def int_list(self, items):
        return [int(i) for i in items]

    # ── Properties ──

    def property_line(self, items):
        member_ids = items[0]  # from int_list
        spec_type = str(items[1])
        section_name = str(items[2])
        self._script.section_assignments.append(SectionAssignment(
            member_ids=member_ids,
            section_name=section_name,
            spec_type=spec_type,
        ))

    def property_section(self, items):
        pass

    # ── Supports ──

    def release_clause(self, items):
        return [str(d) for d in items]

    def released_dof(self, items):
        return str(items[0])

    def support_line(self, items):
        joint_ids = items[0]  # from int_list
        support_type = SupportType(str(items[1]))
        released = items[2] if len(items) > 2 else None
        self._script.supports.append(Support(
            joint_ids=joint_ids,
            support_type=support_type,
            released_dofs=released,
        ))

    def support_section(self, items):
        pass

    # ── Loading ──

    def load_title(self, items):
        return str(items[0]).strip()

    def selfweight_line(self, items):
        return SelfWeight(
            direction=LoadDirection(str(items[0])),
            factor=float(items[1]),
        )

    def member_load_line(self, items):
        member_ids = items[0]  # from int_list
        load_type = LoadType(str(items[1]))
        direction = LoadDirection(str(items[2]))
        value = float(items[3])
        distance = float(items[4]) if len(items) > 4 else None
        return MemberLoad(
            member_ids=member_ids,
            load_type=load_type,
            direction=direction,
            value=value,
            distance=distance,
        )

    def member_load_block(self, items):
        return [i for i in items if isinstance(i, MemberLoad)]

    def joint_load_line(self, items):
        joint_id = int(items[0])
        fx = float(items[1])
        fy = float(items[2])
        fz = float(items[3])
        mx = float(items[4]) if len(items) > 4 else 0.0
        my = float(items[5]) if len(items) > 5 else 0.0
        mz = float(items[6]) if len(items) > 6 else 0.0
        return JointLoad(
            joint_id=joint_id,
            fx=fx, fy=fy, fz=fz,
            mx=mx, my=my, mz=mz,
        )

    def joint_load_block(self, items):
        return [i for i in items if isinstance(i, JointLoad)]

    def load_content(self, items):
        return items[0]

    def loading_section(self, items):
        case_id = int(items[0])
        title = ""
        member_loads = []
        joint_loads = []
        self_weight = None

        for arg in items[1:]:
            if isinstance(arg, str):
                title = arg
            elif isinstance(arg, SelfWeight):
                self_weight = arg
            elif isinstance(arg, list):
                for item in arg:
                    if isinstance(item, MemberLoad):
                        member_loads.append(item)
                    elif isinstance(item, JointLoad):
                        joint_loads.append(item)

        self._script.load_cases.append(LoadCase(
            id=case_id,
            title=title,
            member_loads=member_loads,
            joint_loads=joint_loads,
            self_weight=self_weight,
        ))

    # ── Load Combinations ──

    def combo_title(self, items):
        return str(items[0]).strip()

    def combo_factor_line(self, items):
        return (int(items[0]), float(items[1]))

    def load_combination_section(self, items):
        combo_id = int(items[0])
        title = ""
        factors = {}
        for arg in items[1:]:
            if isinstance(arg, str):
                title = arg
            elif isinstance(arg, tuple):
                case_id, factor = arg
                factors[case_id] = factor

        self._script.load_combinations.append(LoadCombination(
            id=combo_id,
            title=title,
            factors=factors,
        ))

    # ── Analysis ──

    def analysis_section(self, items):
        self._script.perform_analysis = True

    # ── Top-level ──

    def body_section(self, items):
        return items[0] if items else None

    def body_sections(self, items):
        pass

    def start(self, items):
        return self._script


# ──────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────

_parser_cache: Optional[Lark] = None


def get_parser() -> Lark:
    """Get the cached Lark parser instance."""
    global _parser_cache
    if _parser_cache is None:
        with open(GRAMMAR_PATH, "r") as f:
            grammar_text = f.read()
        _parser_cache = Lark(
            grammar_text,
            parser="earley",
            propagate_positions=True,
        )
    return _parser_cache


def parse(text: str) -> Script:
    """
    Parse STAAD Pro DSL text into a Script AST.

    Args:
        text: The DSL script text.

    Returns:
        Script: The parsed AST.

    Raises:
        ParseError: If the text cannot be parsed.
    """
    parser = get_parser()

    try:
        tree = parser.parse(text)
        transformer = StaadTransformer()
        return transformer.transform(tree)
    except UnexpectedInput as e:
        line = getattr(e, 'line', None)
        column = getattr(e, 'column', None)
        msg = "Syntax error: unexpected input"
        if hasattr(e, 'expected') and e.expected:
            expected = ", ".join(sorted(e.expected)[:5])
            msg = f"Syntax error: expected one of [{expected}]"
        raise ParseError(msg, line=line, column=column) from e
    except Exception as e:
        raise ParseError(f"Failed to parse script: {e}") from e


def parse_safe(text: str) -> tuple[Optional[Script], Optional[ParseError]]:
    """
    Parse DSL text, returning (script, None) on success or (None, error) on failure.
    """
    try:
        return parse(text), None
    except ParseError as e:
        return None, e
