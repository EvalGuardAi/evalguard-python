"""DSPy integration for EvalGuard.

Two faithful entry points for the way DSPy programs actually run:

1. :class:`EvalGuardDSPyCallback` — a drop-in for DSPy's native callback
   system. DSPy invokes ``on_module_start`` / ``on_module_end`` around every
   ``Module.__call__`` (``dspy.Predict``, ``dspy.ChainOfThought``, custom
   modules). Register it once and every module call is guarded + traced::

       import dspy
       from evalguard.dspy import EvalGuardDSPyCallback

       dspy.configure(callbacks=[EvalGuardDSPyCallback(api_key="eg_...")])

2. :func:`guard_module` — wrap a single module in place by patching its
   ``forward`` so the input is checked before the call and the output is
   traced after::

       qa = guard_module(dspy.Predict("question -> answer"), api_key="eg_...")
       qa(question="...")

Both forward to :class:`~evalguard.guardrails.GuardrailClient` and raise
:class:`~evalguard.guardrails.GuardrailViolation` when ``block_on_violation``
is set and a check fails. Neither imports ``dspy`` — the callback is duck-typed
against DSPy's ``BaseCallback`` interface, so EvalGuard stays an optional
dependency.
"""

from __future__ import annotations

import functools
import time
from typing import Any, Callable, Dict, List, Optional

from .guardrails import GuardrailClient, GuardrailViolation


def _text_from_inputs(inputs: Any) -> str:
    """Best-effort extraction of the user-facing text from DSPy module inputs.

    DSPy passes a dict like ``{"args": (...), "kwargs": {...}}`` to callbacks
    and plain kwargs to ``forward``. We concatenate the string-valued fields.
    """
    parts: List[str] = []

    def _collect(value: Any) -> None:
        if isinstance(value, str):
            if value.strip():
                parts.append(value)
        elif isinstance(value, dict):
            for v in value.values():
                _collect(v)
        elif isinstance(value, (list, tuple)):
            for v in value:
                _collect(v)

    _collect(inputs)
    return "\n".join(parts)[:8000]


def _text_from_output(output: Any) -> str:
    """Stringify a DSPy ``Prediction`` / output for trace logging."""
    if output is None:
        return ""
    # dspy.Prediction exposes fields via attributes; fall back to str().
    for attr in ("answer", "response", "output", "text"):
        val = getattr(output, attr, None)
        if isinstance(val, str) and val.strip():
            return val[:2000]
    return str(output)[:2000]


class EvalGuardDSPyCallback:
    """DSPy ``BaseCallback``-compatible guard + tracer.

    Implements the subset of the callback protocol EvalGuard needs:
    ``on_module_start`` (pre-check the inputs) and ``on_module_end`` (trace the
    outputs + latency). Other DSPy callback hooks are accepted as no-ops so the
    object satisfies the interface regardless of DSPy version.
    """

    def __init__(
        self,
        api_key: str,
        project_id: Optional[str] = None,
        base_url: str = "https://evalguard.ai/api",
        rules: Optional[List[str]] = None,
        block_on_violation: bool = True,
        timeout: float = 5.0,
    ) -> None:
        self._guard = GuardrailClient(
            api_key=api_key,
            base_url=base_url,
            project_id=project_id,
            timeout=timeout,
        )
        self._rules = rules
        self._block = block_on_violation
        self._starts: Dict[str, float] = {}

    # ── DSPy callback protocol ────────────────────────────────────────
    def on_module_start(self, call_id: str, instance: Any, inputs: Dict[str, Any]) -> None:
        self._starts[call_id] = time.monotonic()
        text = _text_from_inputs(inputs)
        if not text:
            return
        check = self._guard.check_input(
            text,
            rules=self._rules,
            metadata={"framework": "dspy", "module": type(instance).__name__},
        )
        if not check.get("allowed", True) and self._block:
            raise GuardrailViolation(check.get("violations", []))

    def on_module_end(self, call_id: str, outputs: Any, exception: Optional[Exception]) -> None:
        start = self._starts.pop(call_id, None)
        elapsed_ms = round((time.monotonic() - start) * 1000, 2) if start is not None else None
        self._guard.log_trace(
            {
                "provider": "dspy",
                "output": _text_from_output(outputs),
                "llm_latency_ms": elapsed_ms,
                "error": str(exception) if exception else None,
            }
        )

    # ── no-op hooks for full BaseCallback compatibility ───────────────
    def on_lm_start(self, *_args: Any, **_kwargs: Any) -> None:  # pragma: no cover
        pass

    def on_lm_end(self, *_args: Any, **_kwargs: Any) -> None:  # pragma: no cover
        pass

    def on_tool_start(self, *_args: Any, **_kwargs: Any) -> None:  # pragma: no cover
        pass

    def on_tool_end(self, *_args: Any, **_kwargs: Any) -> None:  # pragma: no cover
        pass


def guard_module(
    module: Any,
    *,
    api_key: str,
    project_id: Optional[str] = None,
    base_url: str = "https://evalguard.ai/api",
    rules: Optional[List[str]] = None,
    block_on_violation: bool = True,
    timeout: float = 5.0,
) -> Any:
    """Patch a single DSPy ``Module`` so each call is guarded + traced.

    Wraps the module's ``forward`` (falling back to ``__call__``): the input
    fields are checked before execution and the prediction is traced after.
    Returns the same module instance.
    """
    guard = GuardrailClient(
        api_key=api_key,
        base_url=base_url,
        project_id=project_id,
        timeout=timeout,
    )

    target_name = "forward" if callable(getattr(module, "forward", None)) else "__call__"
    original = getattr(module, target_name, None)
    if not callable(original):
        return module

    @functools.wraps(original)
    def guarded(*args: Any, **kwargs: Any) -> Any:
        text = _text_from_inputs({"args": args, "kwargs": kwargs})
        check = guard.check_input(
            text,
            rules=rules,
            metadata={"framework": "dspy", "module": type(module).__name__},
        )
        if not check.get("allowed", True) and block_on_violation:
            raise GuardrailViolation(check.get("violations", []))

        start = time.monotonic()
        result = original(*args, **kwargs)
        elapsed_ms = round((time.monotonic() - start) * 1000, 2)

        guard.log_trace(
            {
                "provider": "dspy",
                "module": type(module).__name__,
                "input": text[:2000],
                "output": _text_from_output(result),
                "llm_latency_ms": elapsed_ms,
                "violations": check.get("violations", []),
            }
        )
        return result

    setattr(module, target_name, guarded)
    return module


__all__: List[str] = ["EvalGuardDSPyCallback", "guard_module"]
