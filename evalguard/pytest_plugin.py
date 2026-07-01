"""EvalGuard pytest plugin -- auto-report test results as eval runs.

Usage::

    # Enable via CLI
    pytest --evalguard --evalguard-project proj_123

    # Or use the decorator for selective reporting
    from evalguard.pytest_plugin import evalguard_test

    @evalguard_test
    def test_model_accuracy():
        assert call_model("2+2") == "4"

    # Use the fixture for manual control
    def test_custom(evalguard_client):
        result = evalguard_client.run_eval({...})
        assert result["score"] > 0.8

Register via ``pytest11`` entry point (automatic when evalguard is installed).
"""

from __future__ import annotations

import functools
import os
import time
import traceback
from typing import Any, Callable, Dict, List, Optional

import pytest

from .client import EvalGuardClient, EvalGuardError

# ── Marker for selective test reporting ──────────────────────────────

MARKER_NAME = "evalguard_test"
_MARKER_ATTR = "_evalguard_test"


def evalguard_test(fn: Callable | None = None, *, tags: List[str] | None = None):
    """Decorator that marks a test for EvalGuard reporting.

    Can be used bare or with keyword arguments::

        @evalguard_test
        def test_basic(): ...

        @evalguard_test(tags=["accuracy", "gpt-4o"])
        def test_advanced(): ...
    """
    def decorator(func: Callable) -> Callable:
        setattr(func, _MARKER_ATTR, {"tags": tags or []})

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return func(*args, **kwargs)

        # Preserve the marker on the wrapper
        setattr(wrapper, _MARKER_ATTR, getattr(func, _MARKER_ATTR))
        return wrapper

    if fn is not None:
        # Bare @evalguard_test without parentheses
        return decorator(fn)
    return decorator


# ── pytest hooks ─────────────────────────────────────────────────────

def pytest_addoption(parser: pytest.Parser) -> None:
    """Register EvalGuard CLI options."""
    group = parser.getgroup("evalguard", "EvalGuard reporting")
    group.addoption(
        "--evalguard",
        action="store_true",
        default=False,
        help="Enable EvalGuard test reporting.",
    )
    group.addoption(
        "--evalguard-project",
        dest="evalguard_project",
        default=None,
        help="EvalGuard project ID for reporting.",
    )
    group.addoption(
        "--evalguard-api-key",
        dest="evalguard_api_key",
        default=None,
        help="EvalGuard API key (defaults to EVALGUARD_API_KEY env var).",
    )
    group.addoption(
        "--evalguard-base-url",
        dest="evalguard_base_url",
        default=None,
        help="EvalGuard API base URL (defaults to https://evalguard.ai/api/v1).",
    )


def pytest_configure(config: pytest.Config) -> None:
    """Register the evalguard_test marker and attach the plugin if enabled."""
    config.addinivalue_line(
        "markers",
        f"{MARKER_NAME}: mark test for EvalGuard eval reporting",
    )

    if not config.getoption("evalguard", default=False):
        return

    api_key = (
        config.getoption("evalguard_api_key")
        or os.environ.get("EVALGUARD_API_KEY")
    )
    if not api_key:
        pytest.exit(
            "EvalGuard reporting enabled but no API key provided. "
            "Set EVALGUARD_API_KEY or use --evalguard-api-key.",
            returncode=1,
        )

    base_url = (
        config.getoption("evalguard_base_url")
        or os.environ.get("EVALGUARD_BASE_URL")
        or "https://evalguard.ai/api"
    )
    project_id = (
        config.getoption("evalguard_project")
        or os.environ.get("EVALGUARD_PROJECT_ID")
    )

    client = EvalGuardClient(api_key=api_key, base_url=base_url)
    plugin = EvalGuardReporter(client=client, project_id=project_id)
    config.pluginmanager.register(plugin, "evalguard-reporter")


# ── Fixture ──────────────────────────────────────────────────────────

@pytest.fixture
def evalguard_client(request: pytest.FixtureRequest) -> EvalGuardClient:
    """Provide an EvalGuardClient instance configured from CLI options / env.

    Available regardless of whether ``--evalguard`` is passed, so tests
    can use the client for manual eval calls.
    """
    api_key = (
        request.config.getoption("evalguard_api_key", default=None)
        or os.environ.get("EVALGUARD_API_KEY", "")
    )
    base_url = (
        request.config.getoption("evalguard_base_url", default=None)
        or os.environ.get("EVALGUARD_BASE_URL", "https://evalguard.ai/api")
    )
    if not api_key:
        pytest.skip("EVALGUARD_API_KEY not set; skipping evalguard_client fixture")

    return EvalGuardClient(api_key=api_key, base_url=base_url)


# ── Reporter plugin ─────────────────────────────────────────────────

class EvalGuardReporter:
    """Collects test results and sends them to EvalGuard as eval runs."""

    def __init__(self, client: EvalGuardClient, project_id: Optional[str] = None):
        self.client = client
        self.project_id = project_id
        self._results: List[Dict[str, Any]] = []
        self._timings: Dict[str, float] = {}

    # ── Per-test hooks ───────────────────────────────────────────────

    def pytest_runtest_setup(self, item: pytest.Item) -> None:
        """Record test start time."""
        self._timings[item.nodeid] = time.time()

    def pytest_runtest_makereport(self, item: pytest.Item, call: pytest.CallInfo) -> None:  # type: ignore[type-arg]
        """Capture test outcome after the call phase."""
        if call.when != "call":
            return

        # Only report tests decorated with @evalguard_test or marked
        marker_meta = getattr(item.obj, _MARKER_ATTR, None) if hasattr(item, "obj") else None
        has_pytest_marker = bool(item.get_closest_marker(MARKER_NAME))
        if marker_meta is None and not has_pytest_marker:
            return

        start = self._timings.pop(item.nodeid, call.start)
        duration_ms = round((call.stop - start) * 1000, 2)

        # Build result record
        result: Dict[str, Any] = {
            "testName": item.nodeid,
            "displayName": item.name,
            "passed": call.excinfo is None,
            "duration": duration_ms,
            "tags": (marker_meta or {}).get("tags", []),
        }

        if call.excinfo is not None:
            result["error"] = {
                "type": call.excinfo.typename,
                "message": str(call.excinfo.value),
                "traceback": "".join(
                    traceback.format_exception(
                        call.excinfo.type, call.excinfo.value, call.excinfo.tb
                    )
                )[:2000],
            }

        # Capture input/output from test docstring or markers
        if hasattr(item, "obj") and item.obj.__doc__:
            result["description"] = item.obj.__doc__.strip()

        self._results.append(result)

    # ── Session-level reporting ──────────────────────────────────────

    def pytest_sessionfinish(self, session: pytest.Session, exitstatus: int) -> None:
        """Send collected results to EvalGuard after the test session."""
        if not self._results:
            return

        total = len(self._results)
        passed = sum(1 for r in self._results if r["passed"])
        failed = total - passed
        total_duration = sum(r["duration"] for r in self._results)

        payload: Dict[str, Any] = {
            "source": "pytest",
            "summary": {
                "total": total,
                "passed": passed,
                "failed": failed,
                "passRate": round(passed / total, 4) if total else 0,
                "totalDuration": round(total_duration, 2),
            },
            "cases": self._results,
        }

        if self.project_id:
            payload["projectId"] = self.project_id

        try:
            self.client._post("/v1/evals/ci", json=payload)
        except EvalGuardError as exc:
            # Don't fail the test suite because of a reporting error
            import warnings
            warnings.warn(
                f"EvalGuard: failed to report results: {exc}",
                RuntimeWarning,
                stacklevel=1,
            )
        except Exception as exc:
            import warnings
            warnings.warn(
                f"EvalGuard: unexpected error reporting results: {exc}",
                RuntimeWarning,
                stacklevel=1,
            )
