"""
sandbox.py
───────────
Safe execution environment for validating tool calls.
Runs function calls in a sandboxed context to check for execution errors
without touching real network infrastructure.

In production, replace `_mock_execute` with actual telco API stubs.
"""

from __future__ import annotations

import json
import re
import traceback
from typing import Any, Callable

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Mock telecom function implementations
# (Replace with real stubs or simulation layer)
# ──────────────────────────────────────────────────────────────────────────────

def _default_mock(function_name: str, arguments: dict) -> dict:
    """Default mock: validates argument types and returns a dummy response."""
    return {
        "status": "success",
        "function": function_name,
        "result":   f"Mock result for {function_name}({arguments})",
    }


class Sandbox:
    """
    Sandboxed function executor.

    Usage
    ─────
    sandbox = Sandbox(function_library)
    ok = sandbox.execute(response_string)
    ok = sandbox.execute({"function": "fn", "arguments": {...}})
    """

    def __init__(
        self,
        function_library: dict,
        mocks: dict[str, Callable] | None = None,
        timeout_seconds: float = 5.0,
    ):
        self.library    = function_library
        self.mocks      = mocks or {}
        self.timeout    = timeout_seconds
        self._call_log: list[dict] = []

    # ── Public API ────────────────────────────────────────────────────────────

    def execute(self, call_input: str | dict | None) -> bool:
        """
        Execute a function call.  Returns True on success, False on any error.

        Parameters
        ──────────
        call_input : either a raw model response string containing <call>...</call>
                     OR a pre-parsed call dict {"function": ..., "arguments": ...}
        """
        if call_input is None:
            return False

        call = self._resolve_call(call_input)
        if call is None:
            return False

        return self._run_call(call)

    def execute_all(self, response: str) -> list[bool]:
        """Execute all <call> blocks in a multi-step response."""
        from src.reward.base_reward import extract_all_calls
        calls = extract_all_calls(response)
        return [self._run_call(c) for c in calls]

    def get_call_log(self) -> list[dict]:
        return self._call_log.copy()

    def clear_log(self) -> None:
        self._call_log.clear()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _resolve_call(self, call_input: str | dict) -> dict | None:
        if isinstance(call_input, dict):
            return call_input
        from src.reward.base_reward import extract_call
        return extract_call(call_input)

    def _run_call(self, call: dict) -> bool:
        func_name = call.get("function")
        arguments = call.get("arguments", {})

        if not func_name:
            logger.debug("[Sandbox] No function name in call.")
            return False

        # Validate against library schema
        if func_name not in self.library:
            logger.debug(f"[Sandbox] Unknown function: {func_name}")
            self._log(func_name, arguments, "error", "function_not_found")
            return False

        schema = self.library[func_name]
        if not self._validate_args(func_name, arguments, schema):
            return False

        # Execute mock or default
        try:
            mock_fn = self.mocks.get(func_name, _default_mock)
            if mock_fn == _default_mock:
                result = _default_mock(func_name, arguments)
            else:
                result = mock_fn(**arguments)
            self._log(func_name, arguments, "success", result)
            return True
        except Exception as exc:
            logger.debug(f"[Sandbox] Execution error for {func_name}: {exc}")
            self._log(func_name, arguments, "error", str(exc))
            return False

    def _validate_args(self, func_name: str, arguments: dict, schema: dict) -> bool:
        """
        Check that all required parameters are present and
        values respect declared constraints.
        """
        params = schema.get("parameters", {})
        constraints = schema.get("constraints", {})

        # Check required parameters
        for pname, pinfo in params.items():
            if pinfo.get("required", False) and pname not in arguments:
                logger.debug(f"[Sandbox] Missing required param '{pname}' for {func_name}")
                self._log(func_name, arguments, "error", f"missing_required:{pname}")
                return False

        # Check constraints
        for pname, con in constraints.items():
            if pname not in arguments:
                continue
            val = arguments[pname]
            if "min" in con and val < con["min"]:
                logger.debug(f"[Sandbox] {pname}={val} below min={con['min']}")
                return False
            if "max" in con and val > con["max"]:
                logger.debug(f"[Sandbox] {pname}={val} above max={con['max']}")
                return False
            if "enum" in con and val not in con["enum"]:
                logger.debug(f"[Sandbox] {pname}={val} not in enum {con['enum']}")
                return False

        return True

    def _log(self, func_name: str, arguments: dict, status: str, result: Any) -> None:
        self._call_log.append({
            "function":  func_name,
            "arguments": arguments,
            "status":    status,
            "result":    result,
        })