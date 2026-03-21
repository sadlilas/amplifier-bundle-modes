"""Tests for ModeHooks and mount() behavior."""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from amplifier_module_hooks_mode import ModeDiscovery, ModeHooks


def _create_mode_file(path: Path, name: str, description: str = "") -> Path:
    """Helper: create a minimal mode .md file with valid YAML frontmatter."""
    mode_file = path / f"{name}.md"
    mode_file.write_text(
        textwrap.dedent(f"""\
            ---
            mode:
              name: {name}
              description: "{description or name + " mode"}"
              tools:
                safe: [read_file, grep]
              default_action: block
            ---
            # {name.title()} Mode
            You are in {name} mode.
        """),
        encoding="utf-8",
    )
    return mode_file


def _make_coordinator(active_mode: str | None = None) -> MagicMock:
    """Create a mock coordinator with capability storage.

    active_mode is stored in capabilities as 'modes.active_mode',
    not in session_state (which is now capabilities-only).
    session_state is kept as an empty dict for tests that verify
    session_state is not mutated.
    """
    coordinator = MagicMock()
    coordinator.session_state = {}
    coordinator.hooks = MagicMock()

    # Capability storage with side_effects so register/get work together
    # active_mode is stored here as 'modes.active_mode' (not in session_state)
    _capabilities: dict[str, Any] = {}
    if active_mode is not None:
        _capabilities["modes.active_mode"] = active_mode

    def _register_capability(key: str, value: object) -> None:
        _capabilities[key] = value

    def _get_capability(key: str, default: object = None) -> object:
        return _capabilities.get(key, default)

    coordinator.register_capability = MagicMock(side_effect=_register_capability)
    coordinator.get_capability = MagicMock(side_effect=_get_capability)
    return coordinator


class TestMountEventRegistration:
    """Fix 1: mount() must register context injection on provider:request."""

    @pytest.mark.asyncio
    async def test_mount_registers_on_provider_request(self, tmp_path: Path) -> None:
        """The context injection handler must be registered on 'provider:request',
        NOT 'prompt:submit'."""
        modes_dir = tmp_path / "modes"
        modes_dir.mkdir()
        _create_mode_file(modes_dir, "plan")

        coordinator = _make_coordinator()

        from amplifier_module_hooks_mode import mount

        await mount(coordinator, {"search_paths": [str(modes_dir)]})

        register_calls = coordinator.hooks.register.call_args_list

        context_registration = None
        for c in register_calls:
            args, kwargs = c
            if kwargs.get("name") == "mode-context":
                context_registration = c
                break

        assert context_registration is not None, (
            "Expected a hooks.register call with name='mode-context'"
        )

        args, kwargs = context_registration
        event_name = args[0]
        assert event_name == "provider:request", (
            f"mode-context handler must be registered on 'provider:request', "
            f"but was registered on '{event_name}'"
        )


class TestHandlerMethodName:
    """Fix 1: The handler method should be named handle_provider_request."""

    def test_mode_hooks_has_handle_provider_request(self) -> None:
        """ModeHooks must have handle_provider_request method."""
        assert hasattr(ModeHooks, "handle_provider_request"), (
            "ModeHooks must have a 'handle_provider_request' method"
        )

    def test_mode_hooks_no_handle_prompt_submit(self) -> None:
        """The old handle_prompt_submit method must not exist."""
        assert not hasattr(ModeHooks, "handle_prompt_submit"), (
            "ModeHooks must NOT have the old 'handle_prompt_submit' method -- "
            "it should be renamed to 'handle_provider_request'"
        )


class TestInfrastructureToolsBypass:
    """Fix 2: Infrastructure tools must bypass the mode tool cascade."""

    @pytest.mark.asyncio
    async def test_mode_tool_allowed_by_default(self, tmp_path: Path) -> None:
        """The 'mode' tool must be allowed even when default_action is 'block'
        and 'mode' is not in safe_tools."""
        modes_dir = tmp_path / "modes"
        modes_dir.mkdir()
        _create_mode_file(modes_dir, "strict")

        coordinator = _make_coordinator(active_mode="strict")
        discovery = ModeDiscovery(search_paths=[modes_dir])
        hooks = ModeHooks(coordinator, discovery)

        result = await hooks.handle_tool_pre("tool:pre", {"tool_name": "mode"})
        assert result.action == "continue", (
            f"'mode' tool must be allowed (infrastructure tool), "
            f"but got action='{result.action}'"
        )

    @pytest.mark.asyncio
    async def test_todo_tool_allowed_by_default(self, tmp_path: Path) -> None:
        """The 'todo' tool must be allowed even when default_action is 'block'
        and 'todo' is not in safe_tools."""
        modes_dir = tmp_path / "modes"
        modes_dir.mkdir()
        _create_mode_file(modes_dir, "notodo")

        coordinator = _make_coordinator(active_mode="notodo")
        discovery = ModeDiscovery(search_paths=[modes_dir])
        hooks = ModeHooks(coordinator, discovery)

        result = await hooks.handle_tool_pre("tool:pre", {"tool_name": "todo"})
        assert result.action == "continue", (
            f"'todo' tool must be allowed (infrastructure tool), "
            f"but got action='{result.action}'"
        )

    @pytest.mark.asyncio
    async def test_non_infrastructure_tool_still_blocked(self, tmp_path: Path) -> None:
        """Tools NOT in infrastructure_tools must still follow the cascade."""
        modes_dir = tmp_path / "modes"
        modes_dir.mkdir()
        _create_mode_file(modes_dir, "strict")

        coordinator = _make_coordinator(active_mode="strict")
        discovery = ModeDiscovery(search_paths=[modes_dir])
        hooks = ModeHooks(coordinator, discovery)

        result = await hooks.handle_tool_pre("tool:pre", {"tool_name": "write_file"})
        assert result.action == "deny", (
            f"'write_file' must still be blocked by default_action, "
            f"but got action='{result.action}'"
        )


class TestInfrastructureToolsConfig:
    """Fix 2: infrastructure_tools must be configurable."""

    @pytest.mark.asyncio
    async def test_custom_infrastructure_tools(self, tmp_path: Path) -> None:
        """When infrastructure_tools is set to a custom list, only those tools bypass."""
        modes_dir = tmp_path / "modes"
        modes_dir.mkdir()
        _create_mode_file(modes_dir, "custom")

        coordinator = _make_coordinator(active_mode="custom")
        discovery = ModeDiscovery(search_paths=[modes_dir])
        hooks = ModeHooks(coordinator, discovery, infrastructure_tools={"mode"})

        # "mode" should still be allowed
        result = await hooks.handle_tool_pre("tool:pre", {"tool_name": "mode"})
        assert result.action == "continue"

        # "todo" should now be blocked (not in custom list)
        result = await hooks.handle_tool_pre("tool:pre", {"tool_name": "todo"})
        assert result.action == "deny"

    @pytest.mark.asyncio
    async def test_empty_infrastructure_tools_blocks_mode(self, tmp_path: Path) -> None:
        """When infrastructure_tools is empty, even the mode tool is blocked."""
        modes_dir = tmp_path / "modes"
        modes_dir.mkdir()
        _create_mode_file(modes_dir, "locked")

        coordinator = _make_coordinator(active_mode="locked")
        discovery = ModeDiscovery(search_paths=[modes_dir])
        hooks = ModeHooks(coordinator, discovery, infrastructure_tools=set())

        result = await hooks.handle_tool_pre("tool:pre", {"tool_name": "mode"})
        assert result.action == "deny", (
            "With empty infrastructure_tools, 'mode' must be blocked"
        )


class TestModeActiveSignal:
    """Fix 3: Context injection must include an explicit MODE ACTIVE banner."""

    @pytest.mark.asyncio
    async def test_context_has_mode_active_banner(self, tmp_path: Path) -> None:
        """Injected context must start with 'MODE ACTIVE: {name}' inside the tags."""
        modes_dir = tmp_path / "modes"
        modes_dir.mkdir()
        _create_mode_file(modes_dir, "plan", "Plan mode")

        coordinator = _make_coordinator(active_mode="plan")
        discovery = ModeDiscovery(search_paths=[modes_dir])
        hooks = ModeHooks(coordinator, discovery)

        result = await hooks.handle_provider_request("provider:request", {})

        assert result.action == "inject_context"
        content = result.context_injection
        assert content is not None
        assert "MODE ACTIVE: plan" in content

    @pytest.mark.asyncio
    async def test_context_has_do_not_reactivate_warning(self, tmp_path: Path) -> None:
        """Injected context must warn the agent not to re-activate the current mode."""
        modes_dir = tmp_path / "modes"
        modes_dir.mkdir()
        _create_mode_file(modes_dir, "brainstorm", "Brainstorm mode")

        coordinator = _make_coordinator(active_mode="brainstorm")
        discovery = ModeDiscovery(search_paths=[modes_dir])
        hooks = ModeHooks(coordinator, discovery)

        result = await hooks.handle_provider_request("provider:request", {})
        content = result.context_injection
        assert content is not None
        assert "do NOT call" in content or "do not call" in content.lower()
        assert "brainstorm" in content

    @pytest.mark.asyncio
    async def test_context_still_contains_mode_content(self, tmp_path: Path) -> None:
        """The mode's markdown body must still be included after the banner."""
        modes_dir = tmp_path / "modes"
        modes_dir.mkdir()
        _create_mode_file(modes_dir, "plan", "Plan mode")

        coordinator = _make_coordinator(active_mode="plan")
        discovery = ModeDiscovery(search_paths=[modes_dir])
        hooks = ModeHooks(coordinator, discovery)

        result = await hooks.handle_provider_request("provider:request", {})
        content = result.context_injection
        assert content is not None
        assert "You are in plan mode." in content

    @pytest.mark.asyncio
    async def test_context_wrapped_in_system_reminder_tags(
        self, tmp_path: Path
    ) -> None:
        """Context must be wrapped in <system-reminder> tags with mode source."""
        modes_dir = tmp_path / "modes"
        modes_dir.mkdir()
        _create_mode_file(modes_dir, "plan", "Plan mode")

        coordinator = _make_coordinator(active_mode="plan")
        discovery = ModeDiscovery(search_paths=[modes_dir])
        hooks = ModeHooks(coordinator, discovery)

        result = await hooks.handle_provider_request("provider:request", {})
        content = result.context_injection
        assert content is not None
        assert content.startswith('<system-reminder source="mode-plan">')
        assert content.rstrip().endswith("</system-reminder>")


class TestMountRegistersCapabilities:
    """mount() must register modes.discovery and modes.hooks via register_capability."""

    @pytest.mark.asyncio
    async def test_mount_registers_modes_discovery(self, tmp_path: Path) -> None:
        """mount() must call register_capability('modes.discovery', discovery_instance)."""
        modes_dir = tmp_path / "modes"
        modes_dir.mkdir()
        _create_mode_file(modes_dir, "plan")

        coordinator = _make_coordinator()

        from amplifier_module_hooks_mode import mount

        await mount(coordinator, {"search_paths": [str(modes_dir)]})

        calls = coordinator.register_capability.call_args_list
        discovery_calls = [c for c in calls if c.args[0] == "modes.discovery"]
        assert len(discovery_calls) == 1, (
            "mount() must call register_capability exactly once with key 'modes.discovery'"
        )
        registered_value = discovery_calls[0].args[1]
        from amplifier_module_hooks_mode import ModeDiscovery

        assert isinstance(registered_value, ModeDiscovery), (
            f"Value registered for 'modes.discovery' must be a ModeDiscovery instance, "
            f"got {type(registered_value)}"
        )

    @pytest.mark.asyncio
    async def test_mount_registers_modes_hooks(self, tmp_path: Path) -> None:
        """mount() must call register_capability('modes.hooks', hooks_instance)."""
        modes_dir = tmp_path / "modes"
        modes_dir.mkdir()
        _create_mode_file(modes_dir, "plan")

        coordinator = _make_coordinator()

        from amplifier_module_hooks_mode import mount

        await mount(coordinator, {"search_paths": [str(modes_dir)]})

        calls = coordinator.register_capability.call_args_list
        hooks_calls = [c for c in calls if c.args[0] == "modes.hooks"]
        assert len(hooks_calls) == 1, (
            "mount() must call register_capability exactly once with key 'modes.hooks'"
        )
        registered_value = hooks_calls[0].args[1]
        from amplifier_module_hooks_mode import ModeHooks

        assert isinstance(registered_value, ModeHooks), (
            f"Value registered for 'modes.hooks' must be a ModeHooks instance, "
            f"got {type(registered_value)}"
        )


class TestGetActiveModeIsPureLookup:
    """Task 7: _get_active_mode() must be a pure lookup with no side effects."""

    def test_get_active_mode_no_side_effects_with_active_mode(
        self, tmp_path: Path
    ) -> None:
        """_get_active_mode() must NOT write require_approval_tools to session_state."""
        modes_dir = tmp_path / "modes"
        modes_dir.mkdir()
        _create_mode_file(modes_dir, "plan")

        coordinator = _make_coordinator(active_mode="plan")
        # Ensure require_approval_tools is absent so we can detect if it gets set
        coordinator.session_state.pop("require_approval_tools", None)

        discovery = ModeDiscovery(search_paths=[modes_dir])
        hooks = ModeHooks(coordinator, discovery)

        hooks._get_active_mode()

        assert "require_approval_tools" not in coordinator.session_state, (
            "_get_active_mode() must not write 'require_approval_tools' to session_state; "
            "this side-effect is handled by the approval callback registered in mount()"
        )

    def test_get_active_mode_no_side_effects_without_active_mode(self) -> None:
        """_get_active_mode() must NOT write require_approval_tools when no mode is active."""
        coordinator = _make_coordinator(active_mode=None)
        coordinator.session_state.pop("require_approval_tools", None)

        discovery = ModeDiscovery(search_paths=[])
        hooks = ModeHooks(coordinator, discovery)

        hooks._get_active_mode()

        assert "require_approval_tools" not in coordinator.session_state, (
            "_get_active_mode() must not write 'require_approval_tools' when no mode is active"
        )

    def test_get_active_mode_returns_none_when_no_active_mode(self) -> None:
        """_get_active_mode() returns None when session_state has no active_mode."""
        coordinator = _make_coordinator(active_mode=None)
        discovery = ModeDiscovery(search_paths=[])
        hooks = ModeHooks(coordinator, discovery)

        result = hooks._get_active_mode()

        assert result is None

    def test_get_active_mode_returns_mode_definition(self, tmp_path: Path) -> None:
        """_get_active_mode() returns the ModeDefinition when a mode is active."""
        modes_dir = tmp_path / "modes"
        modes_dir.mkdir()
        _create_mode_file(modes_dir, "plan")

        coordinator = _make_coordinator(active_mode="plan")
        discovery = ModeDiscovery(search_paths=[modes_dir])
        hooks = ModeHooks(coordinator, discovery)

        result = hooks._get_active_mode()

        assert result is not None
        assert result.name == "plan"

    def test_get_active_mode_returns_none_for_unknown_mode(self) -> None:
        """_get_active_mode() returns None when active_mode doesn't match any known mode."""
        coordinator = _make_coordinator(active_mode="nonexistent")
        discovery = ModeDiscovery(search_paths=[])
        hooks = ModeHooks(coordinator, discovery)

        result = hooks._get_active_mode()

        assert result is None


class TestMountInitializesActiveModeViaCapability:
    """Task 9: mount() must initialize active_mode via register_capability, not session_state."""

    @pytest.mark.asyncio
    async def test_mount_registers_active_mode_capability_as_none(
        self, tmp_path: Path
    ) -> None:
        """mount() must call register_capability('modes.active_mode', None) on first mount."""
        modes_dir = tmp_path / "modes"
        modes_dir.mkdir()
        _create_mode_file(modes_dir, "plan")

        coordinator = _make_coordinator()

        from amplifier_module_hooks_mode import mount

        await mount(coordinator, {"search_paths": [str(modes_dir)]})

        calls = coordinator.register_capability.call_args_list
        active_mode_calls = [c for c in calls if c.args[0] == "modes.active_mode"]
        assert len(active_mode_calls) == 1, (
            "mount() must call register_capability exactly once with key 'modes.active_mode'"
        )
        assert active_mode_calls[0].args[1] is None, (
            "mount() must register 'modes.active_mode' with None as the initial value"
        )

    @pytest.mark.asyncio
    async def test_mount_skips_register_when_active_mode_capability_already_set(
        self, tmp_path: Path
    ) -> None:
        """mount() must NOT overwrite an existing 'modes.active_mode' capability."""
        modes_dir = tmp_path / "modes"
        modes_dir.mkdir()
        _create_mode_file(modes_dir, "plan")

        coordinator = _make_coordinator()
        # Pre-set the capability so get_capability("modes.active_mode") returns non-None
        coordinator.register_capability("modes.active_mode", "plan")
        coordinator.register_capability.reset_mock()

        from amplifier_module_hooks_mode import mount

        await mount(coordinator, {"search_paths": [str(modes_dir)]})

        calls = coordinator.register_capability.call_args_list
        active_mode_calls = [c for c in calls if c.args[0] == "modes.active_mode"]
        assert len(active_mode_calls) == 0, (
            "mount() must NOT call register_capability('modes.active_mode', ...) when already set"
        )

    @pytest.mark.asyncio
    async def test_mount_no_session_state_init_for_active_mode(
        self, tmp_path: Path
    ) -> None:
        """mount() must not add active_mode to session_state during initialization."""
        modes_dir = tmp_path / "modes"
        modes_dir.mkdir()
        _create_mode_file(modes_dir, "plan")

        coordinator = _make_coordinator()
        # Capture the initial session_state to verify mount() doesn't mutate it
        initial_session_state = dict(coordinator.session_state)

        from amplifier_module_hooks_mode import mount

        await mount(coordinator, {"search_paths": [str(modes_dir)]})

        # mount() must NOT add 'active_mode' to session_state
        assert coordinator.session_state == initial_session_state, (
            "mount() must not add 'active_mode' to session_state; "
            "use register_capability('modes.active_mode', None) instead"
        )
