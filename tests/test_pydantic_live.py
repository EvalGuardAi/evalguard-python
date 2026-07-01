"""R9-1 Phase-B1 — live Pydantic plugin verification.

These tests use REAL `pydantic.BaseModel` validation and confirm that
our plugin actually fires through Pydantic's plugin loader path. The
unit-level direct-handler tests live in `test_pydantic_integration.py`;
this file is the end-to-end version that proves the entry-point
registration works.

If Pydantic isn't installed in the test environment, the entire module
is skipped — the unit tests still cover the handler logic.

Activation contract checked here:
    1. `importlib.metadata.entry_points(group="pydantic.plugins")`
       includes EvalGuard.
    2. Defining a `BaseModel` and calling `Model(...)` causes our
       handler to enqueue a span into the EvalGuard batcher.
    3. `ConfigDict(plugin_settings={"evalguard": {"record": "off"}})`
       fully disables span emission for that model.
"""

from __future__ import annotations

import importlib.metadata as importlib_metadata

import pytest

from evalguard import (
    get_validation_counters,
    reset_validation_counters,
)
from evalguard import tracing

pydantic = pytest.importorskip("pydantic", reason="pydantic not installed")
from pydantic import BaseModel, ConfigDict, ValidationError  # noqa: E402


def _drain_batcher() -> list[dict]:
    """Pull whatever the EvalGuard batcher has queued + clear it."""
    with tracing._batcher._lock:
        out = tracing._batcher._queue[:]
        tracing._batcher._queue.clear()
    return out


@pytest.fixture(autouse=True)
def _reset_state(monkeypatch: pytest.MonkeyPatch) -> None:
    reset_validation_counters()
    monkeypatch.setenv("EVALGUARD_API_KEY", "test-key")
    monkeypatch.setenv("EVALGUARD_TRACING_ENABLED", "true")
    monkeypatch.delenv("EVALGUARD_PYDANTIC_DISABLED", raising=False)
    monkeypatch.delenv("PYDANTIC_DISABLE_PLUGINS", raising=False)
    _drain_batcher()


def test_entry_point_is_registered() -> None:
    """Plugin must be discoverable via the standard Python entry-points API.

    NB: Pydantic looks under group "pydantic" (NOT "pydantic.plugins").
    See pydantic.plugin._loader.PYDANTIC_ENTRY_POINT_GROUP.
    """
    eps = importlib_metadata.entry_points(group="pydantic")
    names = [ep.name for ep in eps]
    assert "evalguard" in names, (
        "EvalGuard pydantic plugin entry point not registered. "
        "Did `pip install -e .` run after pyproject.toml was updated?"
    )


def test_real_basemodel_success_emits_span() -> None:
    """Calling Model(valid_data) emits exactly one EvalGuard span on success."""

    class Order(BaseModel):
        item: str
        qty: int

    Order(item="apple", qty=3)

    queued = _drain_batcher()
    matching = [s for s in queued if s["name"] == "pydantic.validate_python"]
    assert len(matching) == 1, f"Expected 1 validate_python span, got {len(matching)}"
    span = matching[0]
    assert span["status"] == "ok"
    assert span["metadata"]["pydantic.outcome"] == "success"
    assert span["metadata"]["pydantic.error_count"] == 0

    counters = get_validation_counters()
    assert counters["success"] >= 1


def test_real_basemodel_failure_emits_error_span() -> None:
    """Validation failure emits a status=error span with first_error metadata."""

    class Order(BaseModel):
        item: str
        qty: int

    with pytest.raises(ValidationError):
        Order(item="apple", qty="not-an-int")  # type: ignore[arg-type]

    queued = _drain_batcher()
    matching = [s for s in queued if s["name"] == "pydantic.validate_python"]
    assert len(matching) == 1
    span = matching[0]
    assert span["status"] == "error"
    assert span["metadata"]["pydantic.outcome"] == "failure"
    assert span["metadata"]["pydantic.error_count"] >= 1
    assert span["metadata"]["pydantic.first_error_type"]

    counters = get_validation_counters()
    assert counters["failure"] >= 1


def test_record_off_in_plugin_settings_disables_emission() -> None:
    """plugin_settings.evalguard.record='off' fully skips spans for that model."""

    class Quiet(BaseModel):
        model_config = ConfigDict(
            plugin_settings={"evalguard": {"record": "off"}},
        )
        x: int

    Quiet(x=1)
    queued = _drain_batcher()
    # Counters might still bump (cheap), but NO span should land for this model.
    matching = [s for s in queued if "Quiet" in str(s.get("inputs", {}).get("model", ""))]
    assert len(matching) == 0


def test_record_failure_only_skips_success_span() -> None:
    """plugin_settings.evalguard.record='failure' emits only on validation error."""

    class OnlyOnFail(BaseModel):
        model_config = ConfigDict(
            plugin_settings={"evalguard": {"record": "failure"}},
        )
        x: int

    OnlyOnFail(x=1)
    queued_success = _drain_batcher()
    own_success = [s for s in queued_success if "OnlyOnFail" in str(s.get("inputs", {}).get("model", ""))]
    assert len(own_success) == 0

    with pytest.raises(ValidationError):
        OnlyOnFail(x="not-int")  # type: ignore[arg-type]
    queued_fail = _drain_batcher()
    own_fail = [s for s in queued_fail if "OnlyOnFail" in str(s.get("inputs", {}).get("model", ""))]
    assert len(own_fail) == 1


def test_per_model_counters_segregate() -> None:
    """Different model classes get independent counter buckets."""

    class A(BaseModel):
        x: int

    class B(BaseModel):
        y: int

    A(x=1)
    A(x=2)
    B(y=1)

    counters = get_validation_counters()["by_model"]
    a_keys = [k for k in counters if k.endswith(".A")]
    b_keys = [k for k in counters if k.endswith(".B")]
    assert len(a_keys) == 1
    assert len(b_keys) == 1
    assert counters[a_keys[0]]["success"] == 2
    assert counters[b_keys[0]]["success"] == 1


def test_kill_switch_env_disables_plugin(monkeypatch: pytest.MonkeyPatch) -> None:
    """EVALGUARD_PYDANTIC_DISABLED=1 — checked at the function level via is_disabled()."""
    from evalguard.pydantic_integration import is_disabled

    monkeypatch.setenv("EVALGUARD_PYDANTIC_DISABLED", "1")
    assert is_disabled() is True
    monkeypatch.delenv("EVALGUARD_PYDANTIC_DISABLED")
    monkeypatch.setenv("PYDANTIC_DISABLE_PLUGINS", "__all__")
    assert is_disabled() is True
