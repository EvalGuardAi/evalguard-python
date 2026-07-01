"""ControlFlow integration for EvalGuard.

Hooks into Prefect's ControlFlow agent orchestration framework to observe
task execution, capture workflow traces, and apply guardrail checks on
task outputs.

Usage::

    from evalguard.controlflow import (
        EvalGuardTaskObserver,
        observe_flow,
        guard_task_output,
    )
    import controlflow as cf

    # 1. Task observer -- hooks into ControlFlow's task lifecycle
    observer = EvalGuardTaskObserver(api_key="eg_...", project_id="proj_...")

    @cf.flow
    def my_flow():
        task = cf.Task("Summarize the document", agents=[agent])
        result = task.run()
        observer.on_task_complete(task, result)
        return result

    # 2. Observe an entire flow
    @observe_flow(api_key="eg_...", project_id="proj_...")
    @cf.flow
    def my_flow():
        ...

    # 3. Guard individual task outputs
    result = guard_task_output(
        task_result="Some LLM output",
        api_key="eg_...",
        rules=["toxic_content", "pii_redact"],
    )

Requires ``controlflow`` to be installed (``pip install controlflow``).
"""

from __future__ import annotations

import functools
import inspect
import logging
import time
import uuid
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union, overload

from .guardrails import GuardrailClient, GuardrailViolation

logger = logging.getLogger("evalguard.controlflow")

F = TypeVar("F", bound=Callable[..., Any])


# ── Task Observer ───────────────────────────────────────────────────────


class EvalGuardTaskObserver:
    """Observes ControlFlow task execution and sends traces to EvalGuard.

    Captures task definitions, agent assignments, results, timing, and
    the overall flow execution graph.

    Parameters
    ----------
    api_key:
        EvalGuard API key.
    project_id:
        EvalGuard project ID for trace grouping.
    rules:
        Guardrail rules to check task outputs against.
    block_on_violation:
        Raise :class:`GuardrailViolation` if a task output is blocked.
    auto_trace:
        If *True*, automatically log traces for every observed event.
    """

    def __init__(
        self,
        api_key: str,
        project_id: Optional[str] = None,
        base_url: str = "https://evalguard.ai/api",
        rules: Optional[List[str]] = None,
        block_on_violation: bool = False,
        auto_trace: bool = True,
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
        self._auto_trace = auto_trace

        # Track flow execution state
        self._flow_id: Optional[str] = None
        self._flow_start: Optional[float] = None
        self._task_traces: List[Dict[str, Any]] = []

    # ── Flow lifecycle ──────────────────────────────────────────────────

    def on_flow_start(
        self,
        flow: Any = None,
        flow_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Called when a ControlFlow flow begins execution.

        Parameters
        ----------
        flow:
            The ControlFlow flow object (optional).
        flow_name:
            Human-readable flow name.
        metadata:
            Extra metadata to attach.

        Returns
        -------
        str
            A unique flow trace ID.
        """
        self._flow_id = uuid.uuid4().hex
        self._flow_start = time.monotonic()
        self._task_traces = []

        name = flow_name or _get_flow_name(flow)

        if self._auto_trace:
            self._guard.log_trace({
                "provider": "controlflow",
                "event": "flow_start",
                "flow_id": self._flow_id,
                "flow_name": name,
                "metadata": metadata or {},
            })

        return self._flow_id

    def on_flow_end(
        self,
        flow: Any = None,
        result: Any = None,
        error: Optional[Exception] = None,
    ) -> Dict[str, Any]:
        """Called when a ControlFlow flow completes.

        Returns
        -------
        dict
            Summary of the flow execution including all task traces.
        """
        elapsed_ms = (time.monotonic() - self._flow_start) * 1000 if self._flow_start else 0.0
        name = _get_flow_name(flow)

        summary: Dict[str, Any] = {
            "provider": "controlflow",
            "event": "flow_end",
            "flow_id": self._flow_id,
            "flow_name": name,
            "duration_ms": round(elapsed_ms, 2),
            "num_tasks": len(self._task_traces),
            "tasks": self._task_traces,
            "status": "error" if error else "completed",
        }

        if error:
            summary["error"] = f"{type(error).__name__}: {error}"
        if result is not None:
            summary["result"] = str(result)[:4096]

        if self._auto_trace:
            self._guard.log_trace(summary)

        # Reset state
        self._flow_id = None
        self._flow_start = None

        return summary

    # ── Task lifecycle ──────────────────────────────────────────────────

    def on_task_start(
        self,
        task: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Called when a ControlFlow task begins execution.

        Parameters
        ----------
        task:
            A ControlFlow Task object.
        metadata:
            Extra metadata.

        Returns
        -------
        dict
            Task info dict (pass to ``on_task_complete`` as context).
        """
        task_info = _extract_task_info(task)
        task_info["start_time"] = time.monotonic()
        task_info["flow_id"] = self._flow_id

        if metadata:
            task_info["metadata"] = metadata

        if self._auto_trace:
            self._guard.log_trace({
                "provider": "controlflow",
                "event": "task_start",
                "flow_id": self._flow_id,
                **{k: v for k, v in task_info.items() if k != "start_time"},
            })

        return task_info

    def on_task_complete(
        self,
        task: Any,
        result: Any = None,
        task_info: Optional[Dict[str, Any]] = None,
        error: Optional[Exception] = None,
    ) -> Dict[str, Any]:
        """Called when a ControlFlow task completes.

        Parameters
        ----------
        task:
            The ControlFlow Task object.
        result:
            The task result.
        task_info:
            Optional context from ``on_task_start``.
        error:
            Exception if the task failed.

        Returns
        -------
        dict
            The guardrail check result (if rules are set), or empty dict.
        """
        info = task_info or _extract_task_info(task)
        start = info.pop("start_time", time.monotonic())
        elapsed_ms = (time.monotonic() - start) * 1000

        result_str = str(result)[:4096] if result is not None else ""

        # Guardrail check on output
        check_result: Dict[str, Any] = {}
        violations: List[Dict[str, Any]] = []
        if self._rules and result_str and not error:
            check_result = self._guard.check_output(
                result_str,
                rules=self._rules,
                metadata={
                    "framework": "controlflow",
                    "task_objective": info.get("objective", ""),
                },
            )
            violations = check_result.get("violations", [])
            if not check_result.get("allowed", True) and self._block:
                raise GuardrailViolation(violations)

        # Build trace
        trace_entry: Dict[str, Any] = {
            "provider": "controlflow",
            "event": "task_complete",
            "flow_id": self._flow_id,
            "task_objective": info.get("objective", ""),
            "task_type": info.get("type", ""),
            "agents": info.get("agents", []),
            "result": result_str,
            "duration_ms": round(elapsed_ms, 2),
            "status": "error" if error else "completed",
            "violations": violations,
        }

        if error:
            trace_entry["error"] = f"{type(error).__name__}: {error}"

        self._task_traces.append(trace_entry)

        if self._auto_trace:
            self._guard.log_trace(trace_entry)

        return check_result

    # ── Agent observation ───────────────────────────────────────────────

    def on_agent_action(
        self,
        agent: Any,
        action: str,
        task: Any = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log an individual agent action within a task.

        Parameters
        ----------
        agent:
            The ControlFlow Agent object.
        action:
            Description of the action taken.
        task:
            The task context.
        metadata:
            Extra metadata.
        """
        agent_info = _extract_agent_info(agent)

        if self._auto_trace:
            self._guard.log_trace({
                "provider": "controlflow",
                "event": "agent_action",
                "flow_id": self._flow_id,
                "agent": agent_info,
                "action": action[:2000],
                "task_objective": _get_task_objective(task),
                "metadata": metadata or {},
            })

    # ── Convenience ─────────────────────────────────────────────────────

    def get_execution_graph(self) -> Dict[str, Any]:
        """Return the execution graph of observed tasks.

        Returns
        -------
        dict
            Flow execution summary with task dependency graph.
        """
        return {
            "flow_id": self._flow_id,
            "num_tasks": len(self._task_traces),
            "tasks": self._task_traces,
            "total_duration_ms": sum(t.get("duration_ms", 0) for t in self._task_traces),
        }


# ── Flow decorator ──────────────────────────────────────────────────────


@overload
def observe_flow(fn: F) -> F: ...


@overload
def observe_flow(
    *,
    api_key: Optional[str] = None,
    project_id: Optional[str] = None,
    base_url: str = "https://evalguard.ai/api",
    rules: Optional[List[str]] = None,
    block_on_violation: bool = False,
    timeout: float = 5.0,
) -> Callable[[F], F]: ...


def observe_flow(
    fn: Optional[F] = None,
    *,
    api_key: Optional[str] = None,
    project_id: Optional[str] = None,
    base_url: str = "https://evalguard.ai/api",
    rules: Optional[List[str]] = None,
    block_on_violation: bool = False,
    timeout: float = 5.0,
) -> Union[F, Callable[[F], F]]:
    """Decorator that wraps a ControlFlow flow to auto-trace execution.

    Usage::

        @observe_flow(api_key="eg_...", project_id="proj_...")
        @cf.flow
        def my_flow():
            task = cf.Task("Summarize", agents=[agent])
            return task.run()

    Parameters
    ----------
    api_key:
        EvalGuard API key.  Falls back to ``EVALGUARD_API_KEY`` env var.
    project_id:
        EvalGuard project ID.
    rules:
        Guardrail rules to check flow outputs.
    block_on_violation:
        Raise on guardrail violation.
    """
    import os

    resolved_key = api_key or os.environ.get("EVALGUARD_API_KEY", "")
    resolved_project = project_id or os.environ.get("EVALGUARD_PROJECT_ID")
    resolved_base = base_url or os.environ.get("EVALGUARD_BASE_URL", "https://evalguard.ai/api")

    def decorator(func: F) -> F:
        observer: Optional[EvalGuardTaskObserver] = None
        if resolved_key:
            observer = EvalGuardTaskObserver(
                api_key=resolved_key,
                base_url=resolved_base,
                project_id=resolved_project,
                rules=rules,
                block_on_violation=block_on_violation,
                timeout=timeout,
            )

        if inspect.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                if observer:
                    observer.on_flow_start(flow_name=func.__qualname__)
                error: Optional[Exception] = None
                result = None
                try:
                    result = await func(*args, **kwargs)
                    return result
                except Exception as exc:
                    error = exc
                    raise
                finally:
                    if observer:
                        observer.on_flow_end(result=result, error=error)
            return async_wrapper  # type: ignore[return-value]
        else:
            @functools.wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                if observer:
                    observer.on_flow_start(flow_name=func.__qualname__)
                error: Optional[Exception] = None
                result = None
                try:
                    result = func(*args, **kwargs)
                    return result
                except Exception as exc:
                    error = exc
                    raise
                finally:
                    if observer:
                        observer.on_flow_end(result=result, error=error)
            return sync_wrapper  # type: ignore[return-value]

    if fn is not None:
        return decorator(fn)
    return decorator  # type: ignore[return-value]


# ── Standalone guardrail check ──────────────────────────────────────────


def guard_task_output(
    task_result: Any,
    *,
    api_key: str,
    project_id: Optional[str] = None,
    base_url: str = "https://evalguard.ai/api",
    rules: Optional[List[str]] = None,
    block_on_violation: bool = True,
    timeout: float = 5.0,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Check a ControlFlow task result against EvalGuard guardrails.

    Parameters
    ----------
    task_result:
        The task output to check (string or object with str representation).
    api_key:
        EvalGuard API key.
    rules:
        Guardrail rules.  Defaults to ``["toxic_content", "pii_redact"]``.
    block_on_violation:
        Raise :class:`GuardrailViolation` on blocked output.
    metadata:
        Extra metadata for the check.

    Returns
    -------
    dict
        ``{"allowed": bool, "violations": [...], "sanitized": str | None}``

    Raises
    ------
    GuardrailViolation
        If blocked and ``block_on_violation`` is True.
    """
    guard = GuardrailClient(
        api_key=api_key,
        base_url=base_url,
        project_id=project_id,
        timeout=timeout,
    )
    text = str(task_result)[:8192] if task_result is not None else ""

    meta = {"framework": "controlflow", **(metadata or {})}
    result = guard.check_output(
        text,
        rules=rules or ["toxic_content", "pii_redact"],
        metadata=meta,
    )

    if not result.get("allowed", True) and block_on_violation:
        raise GuardrailViolation(result.get("violations", []))

    return result


# ── Internal helpers ────────────────────────────────────────────────────


def _extract_task_info(task: Any) -> Dict[str, Any]:
    """Extract structured info from a ControlFlow Task object."""
    info: Dict[str, Any] = {}

    # Task objective / description
    for attr in ("objective", "description", "name", "instructions"):
        val = getattr(task, attr, None)
        if val:
            info["objective"] = str(val)[:2000]
            break
    if "objective" not in info:
        info["objective"] = str(task)[:500]

    # Task type / result_type
    result_type = getattr(task, "result_type", None)
    if result_type is not None:
        info["type"] = str(result_type)

    # Agents assigned to the task
    agents = getattr(task, "agents", None)
    if agents:
        info["agents"] = [_extract_agent_info(a) for a in agents]
    else:
        info["agents"] = []

    # Task status
    status = getattr(task, "status", None)
    if status:
        info["status"] = str(status)

    # Dependencies
    depends_on = getattr(task, "depends_on", None) or getattr(task, "dependencies", None)
    if depends_on:
        info["dependencies"] = [
            getattr(d, "objective", getattr(d, "name", str(d)))[:200]
            for d in depends_on
        ]

    # Context / tools
    context = getattr(task, "context", None)
    if context:
        info["context"] = str(context)[:1000]

    tools = getattr(task, "tools", None)
    if tools:
        info["tools"] = [getattr(t, "__name__", str(t))[:100] for t in tools]

    return info


def _extract_agent_info(agent: Any) -> Dict[str, Any]:
    """Extract structured info from a ControlFlow Agent object."""
    info: Dict[str, Any] = {}

    for attr in ("name", "role", "id"):
        val = getattr(agent, attr, None)
        if val:
            info["name"] = str(val)
            break
    if "name" not in info:
        info["name"] = str(agent)[:200]

    model = getattr(agent, "model", None)
    if model:
        info["model"] = str(model)

    instructions = getattr(agent, "instructions", None) or getattr(agent, "description", None)
    if instructions:
        info["instructions"] = str(instructions)[:500]

    tools = getattr(agent, "tools", None)
    if tools:
        info["tools"] = [getattr(t, "__name__", str(t))[:100] for t in tools]

    return info


def _get_flow_name(flow: Any) -> str:
    """Extract a name from a flow object."""
    if flow is None:
        return "unknown"
    for attr in ("name", "__qualname__", "__name__"):
        val = getattr(flow, attr, None)
        if val:
            return str(val)
    return str(flow)[:200]


def _get_task_objective(task: Any) -> str:
    """Get the task objective string."""
    if task is None:
        return ""
    for attr in ("objective", "description", "name"):
        val = getattr(task, attr, None)
        if val:
            return str(val)[:500]
    return str(task)[:200]
