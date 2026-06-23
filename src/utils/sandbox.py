from typing import Any, Callable

from src.reward.base_reward import extract_call, extract_all_calls


def _default_mock(function_name: str, arguments: dict) -> dict:
    return {
        "status": "success",
        "function": function_name,
        "result": f"Mock result for {function_name}({arguments})",
    }


class Sandbox:
    def __init__(
        self,
        function_library: dict,
        mocks: dict[str, Callable] | None = None,
        timeout_seconds: float = 5.0,
    ):
        self.library = function_library
        self.mocks = mocks or {}
        self.timeout = timeout_seconds
        self._call_log: list[dict] = []
        self._state: dict = {}

    def execute(self, call_input: str | dict | None) -> bool:
        if call_input is None:
            return False
        call = self._resolve_call(call_input)
        if call is None:
            return False
        return self._run_call(call)

    def execute_all(self, response: str) -> list[bool]:
        calls = extract_all_calls(response)
        return [self._run_call(c) for c in calls]

    def get_call_log(self) -> list[dict]:
        return self._call_log.copy()

    def get_state(self) -> dict:
        return self._state.copy()

    def clear(self) -> None:
        self._call_log.clear()
        self._state.clear()

    def clear_log(self) -> None:
        self.clear()

    def _resolve_call(self, call_input: str | dict) -> dict | None:
        if isinstance(call_input, dict):
            return call_input
        return extract_call(call_input)

    def _run_call(self, call: dict) -> bool:
        func_name = call.get("function")
        arguments = call.get("arguments", {})
        if not func_name:
            return False
        if func_name not in self.library:
            self._log(func_name, arguments, "error", "function_not_found")
            return False
        schema = self.library[func_name]
        if not self._validate_args(func_name, arguments, schema):
            return False
        try:
            mock_fn = self.mocks.get(func_name, _default_mock)
            if mock_fn is _default_mock:
                result = _default_mock(func_name, arguments)
            else:
                result = mock_fn(**arguments)
            self._log(func_name, arguments, "success", result)
            self._state[func_name] = {"arguments": arguments, "result": result}
            return True
        except Exception as exc:
            self._log(func_name, arguments, "error", str(exc))
            return False

    def _validate_args(self, func_name: str, arguments: dict, schema: dict) -> bool:
        params = schema.get("parameters", {})
        constraints = schema.get("constraints", {})
        for pname, pinfo in params.items():
            if isinstance(pinfo, dict) and pinfo.get("required", False) and pname not in arguments:
                self._log(func_name, arguments, "error", f"missing_required:{pname}")
                return False
        for pname, con in constraints.items():
            if pname not in arguments:
                continue
            val = arguments[pname]
            if "min" in con and val < con["min"]:
                self._log(func_name, arguments, "error",
                          f"constraint_violation:{pname}<{con['min']}")
                return False
            if "max" in con and val > con["max"]:
                self._log(func_name, arguments, "error",
                          f"constraint_violation:{pname}>{con['max']}")
                return False
            if "enum" in con and val not in con["enum"]:
                self._log(func_name, arguments, "error",
                          f"constraint_violation:{pname} not in enum")
                return False
        return True

    def _log(self, func_name: str, arguments: dict, status: str, result: Any) -> None:
        self._call_log.append({
            "function": func_name,
            "arguments": arguments,
            "status": status,
            "result": result,
        })


def build_gold_state(gold_calls: list[dict], function_library: dict) -> dict:
    sb = Sandbox(function_library)
    for call in gold_calls:
        sb.execute(call)
    return sb.get_state()
