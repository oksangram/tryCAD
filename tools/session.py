"""
Stateful tool session that accumulates joints, members, section assignments,
supports, and loads across multiple tool calls.

This is the shared state that all tools operate on during a single design session.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import math

from dsl.ast_nodes import (
    Script, Joint, Member, SectionAssignment, Support, SupportType,
    MemberLoad, JointLoad, LoadCase, LoadType, LoadDirection,
    StructureType,
)
from dsl.writer import write


@dataclass
class ToolSession:
    """
    Accumulates structural elements across multiple tool calls.
    Handles joint deduplication and sequential ID assignment.
    """
    _joints: dict[int, Joint] = field(default_factory=dict)     # id → Joint
    _members: dict[int, Member] = field(default_factory=dict)   # id → Member
    _sections: list[SectionAssignment] = field(default_factory=list)
    _supports: list[Support] = field(default_factory=list)
    _load_cases: list[LoadCase] = field(default_factory=list)
    _next_joint_id: int = 1
    _next_member_id: int = 1
    _coord_index: dict[tuple, int] = field(default_factory=dict)  # (x,y,z) → joint_id
    _dedup_tolerance: float = 0.001  # meters

    # ── Joint management (with deduplication) ──

    def add_joint(self, x: float, y: float, z: float) -> int:
        """
        Add a joint or return existing joint ID if one exists at these coordinates.
        Coordinates are rounded to avoid floating-point duplicates.
        """
        key = (round(x, 3), round(y, 3), round(z, 3))
        if key in self._coord_index:
            return self._coord_index[key]

        jid = self._next_joint_id
        self._next_joint_id += 1
        self._joints[jid] = Joint(id=jid, x=key[0], y=key[1], z=key[2])
        self._coord_index[key] = jid
        return jid

    def get_joint(self, joint_id: int) -> Optional[Joint]:
        return self._joints.get(joint_id)

    # ── Member management ──

    def add_member(self, start_joint: int, end_joint: int) -> int:
        """Add a member connecting two joints. Returns member ID."""
        mid = self._next_member_id
        self._next_member_id += 1
        self._members[mid] = Member(id=mid, start_joint=start_joint, end_joint=end_joint)
        return mid

    # ── Section assignment ──

    def assign_section(self, member_ids: list[int], section_name: str,
                       spec_type: str = "ST"):
        """Assign a section to a list of members."""
        self._sections.append(SectionAssignment(
            member_ids=member_ids,
            section_name=section_name,
            spec_type=spec_type,
        ))

    # ── Supports ──

    def add_supports(self, joint_ids: list[int], support_type: str = "FIXED"):
        """Add boundary conditions to joints."""
        self._supports.append(Support(
            joint_ids=joint_ids,
            support_type=SupportType(support_type),
        ))

    # ── Loads ──

    def add_load_case(self, case_id: int, title: str = "") -> LoadCase:
        """Create a new load case and add it to the session."""
        lc = LoadCase(id=case_id, title=title)
        self._load_cases.append(lc)
        return lc

    def add_member_load(self, case_id: int, member_ids: list[int],
                        load_type: str, direction: str, value: float):
        """Add a member load to the specified load case."""
        lc = self._get_or_create_load_case(case_id)
        lc.member_loads.append(MemberLoad(
            member_ids=member_ids,
            load_type=LoadType(load_type),
            direction=LoadDirection(direction),
            value=value,
        ))

    def add_joint_load(self, case_id: int, joint_id: int,
                       fx: float = 0, fy: float = 0, fz: float = 0):
        """Add a joint load to the specified load case."""
        lc = self._get_or_create_load_case(case_id)
        lc.joint_loads.append(JointLoad(
            joint_id=joint_id, fx=fx, fy=fy, fz=fz,
        ))

    def _get_or_create_load_case(self, case_id: int) -> LoadCase:
        for lc in self._load_cases:
            if lc.id == case_id:
                return lc
        lc = LoadCase(id=case_id)
        self._load_cases.append(lc)
        return lc

    # ── Query helpers ──

    def find_joints_at_elevation(self, y: float) -> list[int]:
        """Find all joint IDs at a given Y elevation."""
        return [j.id for j in self._joints.values()
                if abs(j.y - y) < self._dedup_tolerance]

    def find_joints_in_range(self, x_range: tuple = None, y_range: tuple = None,
                              z_range: tuple = None) -> list[int]:
        """Find joints within coordinate ranges."""
        results = []
        for j in self._joints.values():
            if x_range and not (x_range[0] - 0.001 <= j.x <= x_range[1] + 0.001):
                continue
            if y_range and not (y_range[0] - 0.001 <= j.y <= y_range[1] + 0.001):
                continue
            if z_range and not (z_range[0] - 0.001 <= j.z <= z_range[1] + 0.001):
                continue
            results.append(j.id)
        return sorted(results)

    def member_length(self, member_id: int) -> float:
        """Calculate the length of a member."""
        m = self._members.get(member_id)
        if not m:
            return 0.0
        j1 = self._joints[m.start_joint]
        j2 = self._joints[m.end_joint]
        return j1.distance_to(j2)

    # ── Assembly ──

    def to_script(self) -> Script:
        """Convert the session state to a Script AST."""
        return Script(
            structure_type=StructureType.SPACE,
            joints=list(self._joints.values()),
            members=list(self._members.values()),
            section_assignments=self._sections.copy(),
            supports=self._supports.copy(),
            load_cases=self._load_cases.copy(),
        )

    def to_dsl(self) -> str:
        """Convert the session state to DSL text."""
        return write(self.to_script())

    # ── Summary for LLM ──

    def get_summary(self) -> str:
        """Human-readable summary of the current session state."""
        n_joints = len(self._joints)
        n_members = len(self._members)
        n_sections = sum(len(sa.member_ids) for sa in self._sections)
        n_supports = sum(len(s.joint_ids) for s in self._supports)
        n_loads = sum(len(lc.member_loads) + len(lc.joint_loads)
                      for lc in self._load_cases)

        summary = f"Session state: {n_joints} joints, {n_members} members"
        if n_sections:
            summary += f", {len(self._sections)} section groups"
        if n_supports:
            summary += f", {n_supports} supported joints"
        if n_loads:
            summary += f", {n_loads} loads in {len(self._load_cases)} cases"

        if self._joints:
            xs = [j.x for j in self._joints.values()]
            ys = [j.y for j in self._joints.values()]
            zs = [j.z for j in self._joints.values()]
            summary += (f"\nBounding box: X=[{min(xs):.1f}, {max(xs):.1f}], "
                       f"Y=[{min(ys):.1f}, {max(ys):.1f}], "
                       f"Z=[{min(zs):.1f}, {max(zs):.1f}]")

        return summary
