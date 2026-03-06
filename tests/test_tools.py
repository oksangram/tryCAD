"""
Integration tests for the structural tool kit.

Tests the full pipeline: session → tools → validation → DSL output.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from tools.session import ToolSession
from tools.column_grid import create_column_grid
from tools.beam_grid import create_beam_grid, create_beam_spans
from tools.bracing import add_bracing
from tools.loads import add_supports, add_member_load, place_equipment
from tools.assembler import assemble_script
from tools.registry import execute_tool
from validation.validator import validate
from dsl.parser import parse


class TestColumnGrid:
    """Test the column grid tool."""

    def test_single_column(self):
        session = ToolSession()
        result = create_column_grid(session, [0], [0], 0.0, 4.0, "W12X65")
        assert "1 columns" in result
        assert len(session._joints) == 2
        assert len(session._members) == 1

    def test_2x2_grid(self):
        session = ToolSession()
        result = create_column_grid(session, [0, 6], [0, 8], 0.0, 4.0, "W12X65")
        assert "4 columns" in result
        assert len(session._joints) == 8  # 4 base + 4 top
        assert len(session._members) == 4

    def test_3_bay_frame(self):
        session = ToolSession()
        result = create_column_grid(session, [0, 6, 12, 18], [0], 0.0, 4.0, "W12X65")
        assert "4 columns" in result
        assert len(session._joints) == 8


class TestBeamGrid:
    """Test the beam grid tool."""

    def test_simple_beam_spans(self):
        session = ToolSession()
        # Create columns first
        create_column_grid(session, [0, 6, 12], [0], 0.0, 4.0, "W12X65")
        # Beam tops are joints 4, 5, 6 (top)
        result = create_beam_spans(session, [(4, 5), (5, 6)], "W16X36")
        assert "2 beam spans" in result

    def test_2d_beam_grid(self):
        session = ToolSession()
        create_column_grid(session, [0, 6], [0, 8], 0.0, 4.0, "W12X65")
        result = create_beam_grid(session, 4.0, [0, 6], [0, 8], "W16X36")
        assert "beam grid" in result.lower()


class TestBracing:
    """Test bracing tools."""

    def test_x_bracing(self):
        session = ToolSession()
        create_column_grid(session, [0, 6], [0], 0.0, 4.0, "W12X65")
        # Bay 1: joints 1(base left), 2(base right), 3(top left), 4(top right)
        result = add_bracing(session, [[1, 2, 3, 4]], "X", "L4X4X3/8")
        assert "X-bracing" in result
        assert "2 brace members" in result

    def test_chevron_bracing(self):
        session = ToolSession()
        create_column_grid(session, [0, 6], [0], 0.0, 4.0, "W12X65")
        result = add_bracing(session, [[1, 2, 3, 4]], "CHEVRON", "L4X4X3/8")
        assert "CHEVRON-bracing" in result


class TestLoads:
    """Test load tools."""

    def test_supports(self):
        session = ToolSession()
        create_column_grid(session, [0, 6], [0], 0.0, 4.0, "W12X65")
        result = add_supports(session, [1, 2], "FIXED")
        assert "FIXED supports" in result
        assert "2 joints" in result

    def test_member_load(self):
        session = ToolSession()
        create_column_grid(session, [0, 6], [0], 0.0, 4.0, "W12X65")
        create_beam_spans(session, [(3, 4)], "W16X36")
        result = add_member_load(session, [3], "UNI", "GY", -7.0)
        assert "UNI load" in result

    def test_equipment_load(self):
        session = ToolSession()
        create_column_grid(session, [0, 6], [0, 8], 0.0, 4.0, "W12X65")
        result = place_equipment(session, 50.0, 3.0, 4.0, 4.0)
        assert "50.0 kN" in result


class TestValidation:
    """Test the validation engine."""

    def test_valid_simple_frame(self):
        session = ToolSession()
        # Creates: j1(0,0,0), j2(0,4,0), j3(6,0,0), j4(6,4,0)
        # Members: m1(1→2), m2(3→4)
        create_column_grid(session, [0, 6], [0], 0.0, 4.0, "W12X65")
        # Beam connecting tops: j2 → j4
        create_beam_spans(session, [(2, 4)], "W16X36")
        add_supports(session, [1, 3], "FIXED")  # base joints
        add_member_load(session, [3], "UNI", "GY", -7.0)

        script = session.to_script()
        result = validate(script)
        assert result.is_valid, f"Errors: {result.errors}"

    def test_missing_support_warns(self):
        session = ToolSession()
        create_column_grid(session, [0, 6], [0], 0.0, 4.0, "W12X65")

        script = session.to_script()
        result = validate(script)
        assert any("No supports" in w for w in result.warnings)


class TestEndToEnd:
    """Full end-to-end pipeline tests."""

    def test_portal_frame_3bay(self):
        """Build a complete 3-bay portal frame with X-bracing."""
        session = ToolSession()

        # Step 1: Columns
        # j1(0,0,0),j2(0,4,0), j3(6,0,0),j4(6,4,0), j5(12,0,0),j6(12,4,0), j7(18,0,0),j8(18,4,0)
        # m1(1→2), m2(3→4), m3(5→6), m4(7→8)
        create_column_grid(session, [0, 6, 12, 18], [0], 0.0, 4.0, "W12X65")
        assert len(session._joints) == 8

        # Step 2: Beams connecting tops: j2→j4, j4→j6, j6→j8
        create_beam_spans(session, [(2, 4), (4, 6), (6, 8)], "W16X36")

        # Step 3: X-bracing in end bays
        # Bay 1: BL=j1, BR=j3, TL=j2, TR=j4
        add_bracing(session, [[1, 3, 2, 4]], "X", "L4X4X3/8")
        # Bay 3: BL=j5, BR=j7, TL=j6, TR=j8
        add_bracing(session, [[5, 7, 6, 8]], "X", "L4X4X3/8")

        # Step 4: Supports at base joints
        add_supports(session, [1, 3, 5, 7], "FIXED")

        # Step 5: Loads on beam members (5,6,7)
        add_member_load(session, [5, 6, 7], "UNI", "GY", -7.0)

        # Step 6: Validate
        script = session.to_script()
        result = validate(script)
        assert result.is_valid, f"Errors: {result.errors}"

        # Step 7: Generate DSL and verify it parses back
        dsl_text = session.to_dsl()
        parsed_back = parse(dsl_text)
        assert len(parsed_back.joints) == 8
        assert len(parsed_back.members) == 11  # 4 cols + 3 beams + 4 braces

    def test_2d_platform(self):
        """Build a 2D platform with beam grid."""
        session = ToolSession()

        # Columns on a 2x2 grid
        create_column_grid(session, [0, 6, 12], [0, 8], 0.0, 5.0, "W14X90")

        # Beam grid at top
        create_beam_grid(session, 5.0, [0, 6, 12], [0, 8], "W16X36")

        # Supports
        add_supports(session, [1, 2, 3, 4, 5, 6], "FIXED")

        # Validate
        script = session.to_script()
        result = validate(script)
        assert result.is_valid, f"Errors: {result.errors}"

        # Check output parses
        dsl_text = session.to_dsl()
        parsed = parse(dsl_text)
        assert len(parsed.joints) >= 12  # At least 6 base + 6 top
        assert len(parsed.supports) > 0

    def test_registry_execution(self):
        """Test executing tools via the registry."""
        session = ToolSession()
        result = execute_tool(session, "create_column_grid", {
            "grid_x": [0, 6],
            "grid_z": [0],
            "base_y": 0.0,
            "top_y": 4.0,
            "section": "W12X65",
        })
        assert "2 columns" in result
        assert "ERROR" not in result

    def test_unknown_tool(self):
        session = ToolSession()
        result = execute_tool(session, "nonexistent_tool", {})
        assert "ERROR" in result

    def test_joint_deduplication(self):
        """Verify that joints at the same position get deduplicated."""
        session = ToolSession()
        # Create columns — joints at base
        create_column_grid(session, [0, 6], [0], 0.0, 4.0, "W12X65")
        n_joints_before = len(session._joints)

        # Beam grid at same elevation — should reuse top joints
        create_beam_grid(session, 4.0, [0, 6], [0], "W16X36")
        n_joints_after = len(session._joints)

        # Should NOT create new joints at (0,4,0) and (6,4,0) — already exist
        assert n_joints_after == n_joints_before, (
            f"Expected no new joints but got {n_joints_after - n_joints_before} new ones"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
