"""R9-1: Pydantic validator span emission.

Registers EvalGuard as a Pydantic plugin so every `validate_python`,
`validate_json`, and `validate_strings` call emits an EvalGuard span with:

    - input data (size-capped, depth-capped via tracing._safe_serialize)
    - success/failure status
    - per-validation error count + first error path/type
    - schema model name + type (typed-dict / dataclass / model)
    - elapsed wall time

Plus an in-process counter `pydantic.validations` (success/failure/total)
exposed via :func:`get_validation_counters`.

Configuration is per-model via Pydantic's plugin_settings::

    from pydantic import BaseModel, ConfigDict

    class Order(BaseModel):
        model_config = ConfigDict(
            plugin_settings={"evalguard": {"record": "failure"}},
        )
        item: str
        qty: int

Supported ``record`` values:
    ``"all"``      — emit a span for every validation (default if not set).
    ``"failure"``  — emit only when validation raises ValidationError.
    ``"metrics"``  — bump counters only, never emit a span.
    ``"off"``      — disable EvalGuard for this model entirely.

Global default (when a model doesn't set plugin_settings) is controlled by
:func:`configure_pydantic`. Defaults to ``"all"``.

Activation:
    EvalGuard registers an entry point ``pydantic.plugins = evalguard`` in
    setup.py. Once the SDK is installed, Pydantic auto-loads this plugin.
    Plugin loading can be disabled per-process by setting
    ``PYDANTIC_DISABLE_PLUGINS=__all__`` or
    ``EVALGUARD_PYDANTIC_DISABLED=1``.

Why this lands here, not in `tracing.py`:
    Pydantic's plugin protocol returns event handlers, not a context
    manager. We translate the on_enter/on_success/on_error hooks into
    spans here and forward them to the same `_batcher` that `traceable`
    uses, so all observability lands in a single pipeline.
"""

from __future__ import annotations

import logging
import os
import threading
import time
import uuid
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

from . import tracing  # noqa: F401 — for _batcher + _safe_serialize

logger = logging.getLogger("evalguard.pydantic_integration")


# ── Counters ────────────────────────────────────────────────────────────


@dataclass
class _Counters:
    total: int = 0
    success: int = 0
    failure: int = 0
    error: int = 0  # non-ValidationError exceptions
    by_model: Dict[str, Dict[str, int]] = field(default_factory=dict)

    def bump(self, model: str, outcome: str) -> None:
        self.total += 1
        if outcome == "success":
            self.success += 1
        elif outcome == "failure":
            self.failure += 1
        else:
            self.error += 1
        by = self.by_model.setdefault(model, {"success": 0, "failure": 0, "error": 0})
        by[outcome] = by.get(outcome, 0) + 1


_counters = _Counters()
_counters_lock = threading.Lock()


def get_validation_counters() -> Dict[str, Any]:
    """Return a snapshot of in-process pydantic.validations counters."""
    with _counters_lock:
        return {
            "total": _counters.total,
            "success": _counters.success,
            "failure": _counters.failure,
            "error": _counters.error,
            "by_model": {k: dict(v) for k, v in _counters.by_model.items()},
        }


def reset_validation_counters() -> None:
    """Test helper — wipe the counter snapshot."""
    global _counters
    with _counters_lock:
        _counters = _Counters()


# ── Configuration ───────────────────────────────────────────────────────


_RECORD_VALUES = ("all", "failure", "metrics", "off")
_default_record: ContextVar[str] = ContextVar("_default_record", default="all")


def configure_pydantic(*, record: str = "all") -> None:
    """Set the global default plugin_settings.evalguard.record value.

    Per-model plugin_settings override this. Call once at startup.
    """
    if record not in _RECORD_VALUES:
        raise ValueError(
            f"record must be one of {_RECORD_VALUES!r}, got {record!r}",
        )
    _default_record.set(record)


def _resolve_record(plugin_settings: Optional[Dict[str, Any]]) -> str:
    """Pick the effective `record` value for a validator instance."""
    if plugin_settings is not None:
        eg_settings = plugin_settings.get("evalguard")
        if isinstance(eg_settings, dict):
            v = eg_settings.get("record")
            if isinstance(v, str) and v in _RECORD_VALUES:
                return v
    return _default_record.get()


# ── Plugin event handlers ───────────────────────────────────────────────


_in_flight: ContextVar[Optional[Dict[str, Any]]] = ContextVar(
    "_in_flight", default=None,
)


class _ValidationHandler:
    """Pydantic v2 plugin protocol expects an OBJECT with on_enter,
    on_success, on_error, on_exception methods — NOT a 4-tuple of
    functions. This class implements that protocol.

    One instance per (model, method) pair. State (start_time + input)
    is held on a ContextVar so concurrent validations on the same
    validator instance get isolated per-call state.
    """

    __slots__ = ("model_name", "method", "emit_span", "emit_on_success")

    def __init__(self, model_name: str, method: str, record: str) -> None:
        self.model_name = model_name
        self.method = method
        self.emit_span = record in ("all", "failure")
        self.emit_on_success = record == "all"

    def on_enter(self, input_value: Any, **_kwargs: Any) -> None:
        state: Dict[str, Any] = {
            "model": self.model_name,
            "method": self.method,
            "start_time": time.time(),
            "input": input_value,
        }
        _in_flight.set(state)

    def on_success(self, result: Any) -> None:
        state = _in_flight.get()
        _in_flight.set(None)

        with _counters_lock:
            _counters.bump(self.model_name, "success")

        if self.emit_span and self.emit_on_success and state is not None:
            _emit_span(
                state,
                outcome="success",
                error_count=0,
                first_error=None,
                result=result,
            )

    def on_error(self, error: Any) -> None:
        state = _in_flight.get()
        _in_flight.set(None)

        with _counters_lock:
            _counters.bump(self.model_name, "failure")

        if not self.emit_span or state is None:
            return

        try:
            errors_list = list(error.errors())
        except Exception:
            errors_list = []
        try:
            err_count = error.error_count()
        except Exception:
            err_count = len(errors_list)
        first = errors_list[0] if errors_list else None
        first_error: Optional[Dict[str, Any]] = None
        if isinstance(first, dict):
            first_error = {
                "type": first.get("type"),
                "loc": list(first.get("loc", ())),
                "msg": first.get("msg"),
            }

        _emit_span(
            state,
            outcome="failure",
            error_count=err_count,
            first_error=first_error,
            result=None,
        )

    def on_exception(self, exception: BaseException) -> None:
        state = _in_flight.get()
        _in_flight.set(None)

        with _counters_lock:
            _counters.bump(self.model_name, "error")

        if not self.emit_span or state is None:
            return

        _emit_span(
            state,
            outcome="exception",
            error_count=1,
            first_error={
                "type": type(exception).__name__,
                "loc": [],
                "msg": str(exception)[:512],
            },
            result=None,
        )


def _make_handlers(
    schema: Any,
    schema_type: Any,
    schema_type_path: Any,
    schema_kind: str,
    config: Any,
    plugin_settings: Optional[Dict[str, Any]],
    method: str,
) -> Optional[_ValidationHandler]:
    """Build the validator handler instance for one of the three
    validator methods. Returns None when record=='off' so Pydantic skips
    this validator entirely.

    Back-compat: callers that previously expected a 4-tuple of
    (on_enter, on_success, on_error, on_exception) can read the
    handler's bound methods — see `unpack_handler()`.
    """
    record = _resolve_record(plugin_settings)
    if record == "off":
        return None

    model_name = _resolve_model_name(schema_type, schema_type_path)
    return _ValidationHandler(model_name, method, record)


def unpack_handler(
    h: Optional[_ValidationHandler],
) -> Optional[Tuple[Any, Any, Any, Any]]:
    """Helper for unit tests: returns the 4-tuple of bound methods that
    older tests asserted against. Returns None when h is None."""
    if h is None:
        return None
    return (h.on_enter, h.on_success, h.on_error, h.on_exception)


def _resolve_model_name(schema_type: Any, schema_type_path: Any) -> str:
    """Best-effort stable model name for span metadata."""
    if isinstance(schema_type, type):
        mod = getattr(schema_type, "__module__", "") or ""
        qn = getattr(schema_type, "__qualname__", schema_type.__name__)
        return f"{mod}.{qn}" if mod else qn
    if hasattr(schema_type_path, "__str__"):
        try:
            return str(schema_type_path)
        except Exception:
            pass
    return "unknown"


def _emit_span(
    state: Dict[str, Any],
    *,
    outcome: str,
    error_count: int,
    first_error: Optional[Dict[str, Any]],
    result: Any,
) -> None:
    """Build a Span dict and enqueue it via the shared batcher."""
    start = state["start_time"]
    end = time.time()
    span_dict: Dict[str, Any] = {
        "spanId": uuid.uuid4().hex[:16],
        "traceId": tracing._current_trace_id.get() or uuid.uuid4().hex,
        "name": f"pydantic.{state['method']}",
        "startTime": start,
        "endTime": end,
        "durationMs": (end - start) * 1000.0,
        "status": "ok" if outcome == "success" else "error",
        "inputs": {
            "model": state["model"],
            "method": state["method"],
            "input": tracing._safe_serialize(state["input"]),
        },
        "metadata": {
            "pydantic.outcome": outcome,
            "pydantic.error_count": error_count,
        },
    }
    parent = tracing._current_span.get()
    if parent is not None:
        span_dict["parentSpanId"] = parent.span_id

    if outcome == "success":
        # Outputs only present on success — bound to safe-serialize.
        span_dict["outputs"] = tracing._safe_serialize(result)
    elif first_error is not None:
        span_dict["error"] = first_error.get("msg") or "validation failed"
        span_dict["metadata"]["pydantic.first_error_type"] = first_error.get("type")
        span_dict["metadata"]["pydantic.first_error_loc"] = first_error.get("loc")

    tracing._batcher.enqueue(span_dict)


# ── Pydantic plugin protocol ────────────────────────────────────────────


class _EvalGuardPydanticPlugin:
    """Implements pydantic.plugin.PydanticPluginProtocol.

    Pydantic v2's `new_schema_validator` expects a 3-tuple where each
    element is either an instance with `on_enter` / `on_success` /
    `on_error` / `on_exception` methods, OR `None` to skip that
    validator method. Pydantic discovers this via the
    ``pydantic.plugins`` entry point.
    """

    def new_schema_validator(  # type: ignore[no-untyped-def]
        self,
        schema,
        schema_type,
        schema_type_path,
        schema_kind,
        config,
        plugin_settings,
    ):
        try:
            return (
                _make_handlers(
                    schema, schema_type, schema_type_path, schema_kind,
                    config, plugin_settings, method="validate_python",
                ),
                _make_handlers(
                    schema, schema_type, schema_type_path, schema_kind,
                    config, plugin_settings, method="validate_json",
                ),
                _make_handlers(
                    schema, schema_type, schema_type_path, schema_kind,
                    config, plugin_settings, method="validate_strings",
                ),
            )
        except Exception:
            logger.debug("EvalGuard pydantic plugin failed to build", exc_info=True)
            return (None, None, None)


# Module-level singleton used by the entry point.
plugin = _EvalGuardPydanticPlugin()


def is_disabled() -> bool:
    """Honor the kill-switch env vars without importing pydantic."""
    if os.environ.get("EVALGUARD_PYDANTIC_DISABLED", "").strip() in ("1", "true", "TRUE"):
        return True
    if os.environ.get("PYDANTIC_DISABLE_PLUGINS", "").strip() in ("__all__", "evalguard"):
        return True
    return False


__all__ = [
    "configure_pydantic",
    "get_validation_counters",
    "reset_validation_counters",
    "plugin",
    "is_disabled",
]
