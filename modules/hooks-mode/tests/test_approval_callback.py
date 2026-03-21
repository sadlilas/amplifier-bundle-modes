"""Tests for approval.needs_check callback registered in mount().

The approval.needs_check capability is a callable registered at mount time.
It takes a tool name and returns True if the current active mode's confirm_tools
includes that tool.
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from amplifier_module_hooks_mode import mount


def _make_coordinator() -> MagicMock:
    """Create a mock coordinator with session_state and capability storage."""
    coordinator = MagicMock()
    coordinator.session_state = {
        "active_mode": None,
        "require_approval_tools": set(),
    }
    coordinator.hooks = MagicMock()

    # Capability storage with side_effects so register/get work together
    _capabilities: dict = {}

    def _register_capability(key: str, value: object) -> None:
        _capabilities[key] = value

    def _get_capability(key: str, default: object = None) -> object:
        return _capabilities.get(key, default)

    coordinator.register_capability = MagicMock(side_effect=_register_capability)
    coordinator.get_capability = MagicMock(side_effect=_get_capability)
    return coordinator


def _create_mode_file_with_confirm(
    path: Path, name: str, confirm_tools: list[str] | None = None
) -> Path:
    """Helper: create a mode .md file with confirm_tools specified."""
    confirm_tools = confirm_tools or []
    confirm_str = ", ".join(f'"{t}"' for t in confirm_tools) if confirm_tools else ""
    confirm_section = f"        confirm: [{confirm_str}]\n" if confirm_tools else ""

    mode_file = path / f"{name}.md"
    mode_file.write_text(
        textwrap.dedent(f"""\
            ---
            mode:
              name: {name}
              description: "{name} mode"
              tools:
                safe: [read_file, grep]
                {confirm_section.strip()}
              default_action: block
            ---
            # {name.title()} Mode
            You are in {name} mode.
        """),
        encoding="utf-8",
    )
    return mode_file


def _create_mode_file_with_confirm_raw(
    path: Path, name: str, confirm_tools: list[str]
) -> Path:
    """Helper: create a mode .md file with confirm_tools in raw YAML."""
    confirm_list = "[" + ", ".join(confirm_tools) + "]"
    mode_file = path / f"{name}.md"
    mode_file.write_text(
        textwrap.dedent(f"""\
            ---
            mode:
              name: {name}
              description: "{name} mode"
              tools:
                safe: [read_file, grep]
                confirm: {confirm_list}
              default_action: block
            ---
            # {name.title()} Mode
            You are in {name} mode.
        """),
        encoding="utf-8",
    )
    return mode_file


class TestApprovalNeedsCheckCallback:
    """Tests for the approval.needs_check capability registered by mount()."""

    @pytest.mark.asyncio
    async def test_callback_registered_at_mount(self, tmp_path: Path) -> None:
        """mount() must register approval.needs_check as a callable capability."""
        modes_dir = tmp_path / "modes"
        modes_dir.mkdir()

        coordinator = _make_coordinator()

        await mount(coordinator, {"search_paths": [str(modes_dir)]})

        callback = coordinator.get_capability("approval.needs_check")
        assert callback is not None, (
            "mount() must register a capability with key 'approval.needs_check'"
        )
        assert callable(callback), (
            "approval.needs_check must be a callable, got: %s" % type(callback)
        )

    @pytest.mark.asyncio
    async def test_callback_returns_true_for_confirm_tool(self, tmp_path: Path) -> None:
        """Callback returns True when the active mode's confirm_tools includes the tool."""
        modes_dir = tmp_path / "modes"
        modes_dir.mkdir()
        _create_mode_file_with_confirm_raw(modes_dir, "careful", ["bash", "write_file"])

        coordinator = _make_coordinator()
        coordinator.session_state["active_mode"] = "careful"

        await mount(coordinator, {"search_paths": [str(modes_dir)]})

        callback = coordinator.get_capability("approval.needs_check")
        assert callback("bash") is True, (
            "callback('bash') must return True when 'bash' is in the active mode's confirm_tools"
        )
        assert callback("write_file") is True, (
            "callback('write_file') must return True when 'write_file' is in confirm_tools"
        )

    @pytest.mark.asyncio
    async def test_callback_returns_false_for_safe_tool(self, tmp_path: Path) -> None:
        """Callback returns False for tools that are not in confirm_tools."""
        modes_dir = tmp_path / "modes"
        modes_dir.mkdir()
        _create_mode_file_with_confirm_raw(modes_dir, "careful", ["bash"])

        coordinator = _make_coordinator()
        coordinator.session_state["active_mode"] = "careful"

        await mount(coordinator, {"search_paths": [str(modes_dir)]})

        callback = coordinator.get_capability("approval.needs_check")
        assert callback("read_file") is False, (
            "callback('read_file') must return False — 'read_file' is safe, not in confirm_tools"
        )
        assert callback("grep") is False, (
            "callback('grep') must return False — 'grep' is safe, not in confirm_tools"
        )

    @pytest.mark.asyncio
    async def test_callback_returns_false_when_no_mode_active(
        self, tmp_path: Path
    ) -> None:
        """Callback returns False for any tool when no mode is active."""
        modes_dir = tmp_path / "modes"
        modes_dir.mkdir()

        coordinator = _make_coordinator()
        # active_mode is None by default

        await mount(coordinator, {"search_paths": [str(modes_dir)]})

        callback = coordinator.get_capability("approval.needs_check")
        assert callback("bash") is False, (
            "callback('bash') must return False when no mode is active"
        )
        assert callback("write_file") is False, (
            "callback('write_file') must return False when no mode is active"
        )

    @pytest.mark.asyncio
    async def test_callback_reflects_current_mode_not_snapshot(
        self, tmp_path: Path
    ) -> None:
        """Callback reflects the CURRENT active mode, not a snapshot taken at mount time.

        This verifies that the callback is a live closure reading from session_state,
        not a value captured at mount() time.
        """
        modes_dir = tmp_path / "modes"
        modes_dir.mkdir()
        _create_mode_file_with_confirm_raw(modes_dir, "strict", ["bash", "write_file"])
        # Create a second mode without confirm_tools
        second_mode_file = modes_dir / "safe.md"
        second_mode_file.write_text(
            textwrap.dedent("""\
                ---
                mode:
                  name: safe
                  description: "Safe mode"
                  tools:
                    safe: [read_file, grep, bash]
                  default_action: allow
                ---
                # Safe Mode
                You are in safe mode.
            """),
            encoding="utf-8",
        )

        coordinator = _make_coordinator()
        # Start with no mode active
        coordinator.session_state["active_mode"] = None

        await mount(coordinator, {"search_paths": [str(modes_dir)]})

        callback = coordinator.get_capability("approval.needs_check")

        # No mode active — should return False
        assert callback("bash") is False, (
            "callback('bash') must return False when no mode active"
        )

        # Activate strict mode — bash is now in confirm_tools
        coordinator.session_state["active_mode"] = "strict"
        assert callback("bash") is True, (
            "callback('bash') must return True after activating 'strict' mode "
            "which has 'bash' in confirm_tools"
        )

        # Switch to safe mode — bash is no longer in confirm_tools
        coordinator.session_state["active_mode"] = "safe"
        assert callback("bash") is False, (
            "callback('bash') must return False after switching to 'safe' mode "
            "which has no confirm_tools"
        )
