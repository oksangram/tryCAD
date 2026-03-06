"""
Tests for the STAAD Pro DSL parser, writer, and round-trip.
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from dsl import parse, parse_safe, write, ParseError, Script


TEST_DATA = Path(__file__).parent / "test_data"
VALID_SCRIPTS = TEST_DATA / "valid_scripts"
INVALID_SCRIPTS = TEST_DATA / "invalid_scripts"


# ──────────────────────────────────────────────────────────────
# Basic Parsing Tests
# ──────────────────────────────────────────────────────────────

class TestParseBasic:
    """Test basic parsing of well-formed scripts."""

    def test_parse_portal_frame(self):
        """Parse the 3-bay portal frame test script."""
        text = (VALID_SCRIPTS / "portal_frame_3bay.std").read_text()
        script = parse(text)

        assert script.structure_type.value == "SPACE"
        assert script.unit_length == "METER"
        assert script.unit_force == "KN"

    def test_joint_count(self):
        text = (VALID_SCRIPTS / "portal_frame_3bay.std").read_text()
        script = parse(text)
        assert len(script.joints) == 8

    def test_joint_coordinates(self):
        text = (VALID_SCRIPTS / "portal_frame_3bay.std").read_text()
        script = parse(text)

        j1 = script.get_joint(1)
        assert j1 is not None
        assert j1.x == 0.0
        assert j1.y == 0.0
        assert j1.z == 0.0

        j8 = script.get_joint(8)
        assert j8 is not None
        assert j8.x == 18.0
        assert j8.y == 4.0
        assert j8.z == 0.0

    def test_member_count(self):
        text = (VALID_SCRIPTS / "portal_frame_3bay.std").read_text()
        script = parse(text)
        assert len(script.members) == 11

    def test_member_connectivity(self):
        text = (VALID_SCRIPTS / "portal_frame_3bay.std").read_text()
        script = parse(text)

        m1 = script.get_member(1)
        assert m1 is not None
        assert m1.start_joint == 1
        assert m1.end_joint == 5

    def test_section_assignments(self):
        text = (VALID_SCRIPTS / "portal_frame_3bay.std").read_text()
        script = parse(text)
        assert len(script.section_assignments) == 3

        # Columns: W12X65
        sa_cols = script.section_assignments[0]
        assert sa_cols.section_name == "W12X65"
        assert sa_cols.member_ids == [1, 2, 3, 4]

    def test_supports(self):
        text = (VALID_SCRIPTS / "portal_frame_3bay.std").read_text()
        script = parse(text)
        assert len(script.supports) == 1
        assert script.supports[0].joint_ids == [1, 2, 3, 4]
        assert script.supports[0].support_type.value == "FIXED"

    def test_loading(self):
        text = (VALID_SCRIPTS / "portal_frame_3bay.std").read_text()
        script = parse(text)
        assert len(script.load_cases) == 1

        lc = script.load_cases[0]
        assert lc.id == 1
        assert len(lc.member_loads) == 1

        ml = lc.member_loads[0]
        assert ml.member_ids == [5, 6, 7]
        assert ml.load_type.value == "UNI"
        assert ml.direction.value == "GY"
        assert ml.value == -7.0

    def test_perform_analysis(self):
        text = (VALID_SCRIPTS / "portal_frame_3bay.std").read_text()
        script = parse(text)
        assert script.perform_analysis is True


# ──────────────────────────────────────────────────────────────
# AST Utilities
# ──────────────────────────────────────────────────────────────

class TestASTUtilities:
    """Test convenience methods on AST nodes."""

    def test_joint_distance(self):
        from dsl.ast_nodes import Joint
        j1 = Joint(1, 0, 0, 0)
        j2 = Joint(2, 3, 4, 0)
        assert abs(j1.distance_to(j2) - 5.0) < 1e-6

    def test_bounding_box(self):
        text = (VALID_SCRIPTS / "portal_frame_3bay.std").read_text()
        script = parse(text)
        bb_min, bb_max = script.bounding_box()
        assert bb_min == (0.0, 0.0, 0.0)
        assert bb_max == (18.0, 4.0, 0.0)

    def test_joint_ids_set(self):
        text = (VALID_SCRIPTS / "portal_frame_3bay.std").read_text()
        script = parse(text)
        assert script.joint_ids() == {1, 2, 3, 4, 5, 6, 7, 8}

    def test_max_ids(self):
        text = (VALID_SCRIPTS / "portal_frame_3bay.std").read_text()
        script = parse(text)
        assert script.max_joint_id() == 8
        assert script.max_member_id() == 11


# ──────────────────────────────────────────────────────────────
# Writer Tests
# ──────────────────────────────────────────────────────────────

class TestWriter:
    """Test the DSL writer."""

    def test_write_produces_valid_output(self):
        text = (VALID_SCRIPTS / "portal_frame_3bay.std").read_text()
        script = parse(text)
        output = write(script)

        assert "STAAD SPACE" in output
        assert "JOINT COORDINATES" in output
        assert "MEMBER INCIDENCES" in output
        assert "FINISH" in output

    def test_write_contains_all_joints(self):
        text = (VALID_SCRIPTS / "portal_frame_3bay.std").read_text()
        script = parse(text)
        output = write(script)

        assert "18.000" in output  # Max X coordinate
        assert "4.000" in output   # Max Y coordinate

    def test_write_contains_sections(self):
        text = (VALID_SCRIPTS / "portal_frame_3bay.std").read_text()
        script = parse(text)
        output = write(script)

        assert "W12X65" in output
        assert "W16X36" in output
        assert "L4X4X3/8" in output


# ──────────────────────────────────────────────────────────────
# Round-Trip Tests (parse → write → parse → compare)
# ──────────────────────────────────────────────────────────────

class TestRoundTrip:
    """Test parse → write → parse round-trip consistency."""

    def test_round_trip_preserves_joints(self):
        text = (VALID_SCRIPTS / "portal_frame_3bay.std").read_text()
        script1 = parse(text)
        output = write(script1)
        script2 = parse(output)

        assert len(script2.joints) == len(script1.joints)
        for j1, j2 in zip(
            sorted(script1.joints, key=lambda j: j.id),
            sorted(script2.joints, key=lambda j: j.id),
        ):
            assert j1.id == j2.id
            assert abs(j1.x - j2.x) < 1e-6
            assert abs(j1.y - j2.y) < 1e-6
            assert abs(j1.z - j2.z) < 1e-6

    def test_round_trip_preserves_members(self):
        text = (VALID_SCRIPTS / "portal_frame_3bay.std").read_text()
        script1 = parse(text)
        output = write(script1)
        script2 = parse(output)

        assert len(script2.members) == len(script1.members)
        for m1, m2 in zip(
            sorted(script1.members, key=lambda m: m.id),
            sorted(script2.members, key=lambda m: m.id),
        ):
            assert m1.id == m2.id
            assert m1.start_joint == m2.start_joint
            assert m1.end_joint == m2.end_joint


# ──────────────────────────────────────────────────────────────
# Error Handling Tests
# ──────────────────────────────────────────────────────────────

class TestErrorHandling:
    """Test parser error reporting."""

    def test_parse_safe_returns_error(self):
        script, error = parse_safe("INVALID GARBAGE TEXT")
        assert script is None
        assert error is not None
        assert isinstance(error, ParseError)

    def test_empty_script_fails(self):
        with pytest.raises(ParseError):
            parse("")

    def test_missing_finish_fails(self):
        text = "STAAD SPACE\nUNIT METER KN\nJOINT COORDINATES\n1 0 0 0\nMEMBER INCIDENCES\n1 1 1\n"
        with pytest.raises(ParseError):
            parse(text)


# ──────────────────────────────────────────────────────────────
# Programmatic Script Construction (for tools & datagen)
# ──────────────────────────────────────────────────────────────

class TestProgrammaticConstruction:
    """Test building scripts programmatically (not via parsing)."""

    def test_build_simple_frame(self):
        from dsl.ast_nodes import (
            Script, Joint, Member, SectionAssignment,
            Support, SupportType, LoadCase, MemberLoad,
            LoadType, LoadDirection, StructureType,
        )

        script = Script(structure_type=StructureType.SPACE)
        script.joints = [
            Joint(1, 0, 0, 0), Joint(2, 6, 0, 0),
            Joint(3, 0, 4, 0), Joint(4, 6, 4, 0),
        ]
        script.members = [
            Member(1, 1, 3), Member(2, 2, 4), Member(3, 3, 4),
        ]
        script.section_assignments = [
            SectionAssignment([1, 2], "W12X65"),
            SectionAssignment([3], "W16X36"),
        ]
        script.supports = [
            Support([1, 2], SupportType.FIXED),
        ]
        script.load_cases = [
            LoadCase(
                id=1, title="DL+LL",
                member_loads=[
                    MemberLoad([3], LoadType.UNI, LoadDirection.GY, -7.0),
                ],
            ),
        ]

        output = write(script)
        assert "STAAD SPACE" in output
        assert "W12X65" in output
        assert "-7.0" in output

        # Verify it can be parsed back
        script2 = parse(output)
        assert len(script2.joints) == 4
        assert len(script2.members) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
