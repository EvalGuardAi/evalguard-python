"""Composio tool execution tracing and guardrails for EvalGuard.

Composio is a tool integration layer for AI agents (200+ tools). This
integration provides middleware that traces all tool executions, captures
inputs/outputs/latency, and applies guardrail checks to prevent sensitive
data from leaking through tool calls.

Usage::

    from composio import ComposioToolSet
    from evalguard.composio import EvalGuardMiddleware, wrap_toolset

    # Option 1: Wrap an entire Composio toolset
    toolset = ComposioToolSet()
    toolset = wrap_toolset(
        toolset,
        api_key="eg_...",
        project_id="proj_...",
    )
    # All tool executions are now traced and guarded

    # Option 2: Use middleware for granular control
    middleware = EvalGuardMiddleware(api_key="eg_...")
    result = middleware.execute_with_guard(
        tool_name="GMAIL_SEND",
        params={"to": "user@example.com", "body": "Hello"},
        execute_fn=lambda p: toolset.execute_action("GMAIL_SEND", p),
    )

    # Option 3: Guard tool inputs against sensitive data leaks
    middleware = EvalGuardMiddleware(
        api_key="eg_...",
        sensitive_patterns=["SSN", "credit_card", "password"],
    )
"""

from __future__ import annotations

import functools
import logging
import re
import time
import uuid
from typing import Any, Callable, Dict, List, Optional, Set

from .guardrails import GuardrailClient, GuardrailViolation

logger = logging.getLogger("evalguard.composio")

# Default patterns that indicate sensitive data in tool inputs
_DEFAULT_SENSITIVE_PATTERNS: List[str] = [
    r"\b\d{3}-\d{2}-\d{4}\b",           # SSN
    r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b",  # Credit card
    r"(?i)\b(?:password|passwd|secret|api_key|token)\s*[:=]\s*\S+",  # Credentials
    r"(?i)\bBearer\s+[A-Za-z0-9\-._~+/]+=*\b",  # Bearer tokens
]


class EvalGuardMiddleware:
    """Middleware that traces and guards Composio tool executions.

    Intercepts tool calls to:
    1. Check inputs against guardrails (PII, prompt injection, sensitive data)
    2. Measure execution latency
    3. Capture inputs, outputs, success/failure status
    4. Send structured traces to EvalGuard

    Parameters
    ----------
    api_key:
        EvalGuard API key.
    project_id:
        Optional project ID for trace grouping.
    base_url:
        API base URL. Override for self-hosted deployments.
    rules:
        Guardrail rules for input checking.
    block_on_violation:
        If *True*, block tool execution when guardrails are violated.
    sensitive_patterns:
        Additional regex patterns for detecting sensitive data in tool
        inputs. Applied on top of the default set (SSN, credit card,
        credentials, bearer tokens).
    redact_sensitive:
        If *True*, redact matched sensitive data in traces instead of
        blocking. If *False* and ``block_on_violation`` is *True*, the
        tool call is blocked entirely.
    timeout:
        HTTP request timeout in seconds.
    """

    def __init__(
        self,
        api_key: str,
        project_id: Optional[str] = None,
        base_url: str = "https://evalguard.ai/api",
        rules: Optional[List[str]] = None,
        block_on_violation: bool = True,
        sensitive_patterns: Optional[List[str]] = None,
        redact_sensitive: bool = False,
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
        self._redact = redact_sensitive
        self._traces: List[Dict[str, Any]] = []

        # Compile sensitive data patterns
        all_patterns = list(_DEFAULT_SENSITIVE_PATTERNS)
        if sensitive_patterns:
            all_patterns.extend(sensitive_patterns)
        self._sensitive_re = [re.compile(p) for p in all_patterns]

        # Track tool execution stats
        self._tool_stats: Dict[str, Dict[str, Any]] = {}

    # ── Sensitive data scanning ──────────────────────────────────────

    def _scan_for_sensitive_data(self, data: Any) -> List[Dict[str, Any]]:
        """Scan tool input data for sensitive patterns.

        Returns a list of findings with pattern and location info.
        """
        findings: List[Dict[str, Any]] = []
        text = _flatten_to_string(data)

        for pattern_re in self._sensitive_re:
            for match in pattern_re.finditer(text):
                findings.append({
                    "pattern": pattern_re.pattern,
                    "match": match.group()[:20] + "..." if len(match.group()) > 20 else match.group(),
                    "position": match.start(),
                })

        return findings

    def _redact_sensitive_data(self, data: Any) -> Any:
        """Redact sensitive patterns from data, returning a sanitized copy."""
        if isinstance(data, str):
            result = data
            for pattern_re in self._sensitive_re:
                result = pattern_re.sub("[REDACTED]", result)
            return result
        if isinstance(data, dict):
            return {k: self._redact_sensitive_data(v) for k, v in data.items()}
        if isinstance(data, (list, tuple)):
            return type(data)(self._redact_sensitive_data(v) for v in data)
        return data

    # ── Core execution with guardrails ───────────────────────────────

    def execute_with_guard(
        self,
        tool_name: str,
        params: Dict[str, Any],
        execute_fn: Callable[[Dict[str, Any]], Any],
        *,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Execute a tool call with guardrail checks and tracing.

        Parameters
        ----------
        tool_name:
            Name of the Composio tool/action (e.g., ``"GMAIL_SEND"``).
        params:
            Tool input parameters.
        execute_fn:
            Callable that performs the actual tool execution.
        metadata:
            Additional metadata to include in the trace.

        Returns
        -------
        The tool's return value.

        Raises
        ------
        GuardrailViolation
            If ``block_on_violation`` is True and input check fails or
            sensitive data is detected (when ``redact_sensitive`` is False).
        """
        span_id = uuid.uuid4().hex[:16]
        trace_id = uuid.uuid4().hex

        # Step 1: Scan for sensitive data
        sensitive_findings = self._scan_for_sensitive_data(params)
        if sensitive_findings:
            if self._redact:
                params = self._redact_sensitive_data(params)
                logger.info(
                    "Redacted %d sensitive data occurrence(s) in %s inputs",
                    len(sensitive_findings), tool_name,
                )
            elif self._block:
                violation = {
                    "rule": "sensitive_data_leak",
                    "tool": tool_name,
                    "findings_count": len(sensitive_findings),
                    "message": f"Sensitive data detected in {tool_name} inputs",
                }
                self._guard.log_trace({
                    "provider": "composio",
                    "span_type": "tool_blocked",
                    "span_id": span_id,
                    "trace_id": trace_id,
                    "tool_name": tool_name,
                    "status": "blocked",
                    "violation": violation,
                })
                raise GuardrailViolation(
                    [violation],
                    message=f"Sensitive data detected in {tool_name} tool inputs",
                )

        # Step 2: Guardrail check on serialized input
        input_text = _flatten_to_string(params)
        check = self._guard.check_input(
            input_text,
            rules=self._rules,
            metadata={
                "framework": "composio",
                "tool": tool_name,
                **(metadata or {}),
            },
        )
        if not check.get("allowed", True) and self._block:
            self._guard.log_trace({
                "provider": "composio",
                "span_type": "tool_blocked",
                "span_id": span_id,
                "trace_id": trace_id,
                "tool_name": tool_name,
                "status": "blocked",
                "violations": check.get("violations", []),
            })
            raise GuardrailViolation(check.get("violations", []))

        # Step 3: Execute with timing
        start = time.monotonic()
        error_msg: Optional[str] = None
        result = None
        try:
            result = execute_fn(params)
            return result
        except GuardrailViolation:
            raise
        except Exception as exc:
            error_msg = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            elapsed_ms = (time.monotonic() - start) * 1000
            status = "error" if error_msg else "ok"

            # Update stats
            stats = self._tool_stats.setdefault(tool_name, {
                "call_count": 0, "error_count": 0, "total_ms": 0.0,
            })
            stats["call_count"] += 1
            stats["total_ms"] += elapsed_ms
            if error_msg:
                stats["error_count"] += 1

            # Log trace
            entry: Dict[str, Any] = {
                "provider": "composio",
                "span_type": "tool_execution",
                "span_id": span_id,
                "trace_id": trace_id,
                "tool_name": tool_name,
                "input": input_text[:2000],
                "output": str(result)[:2000] if result else "",
                "latency_ms": round(elapsed_ms, 2),
                "status": status,
                "violations": check.get("violations", []),
                "sensitive_findings_count": len(sensitive_findings),
            }
            if error_msg:
                entry["error"] = error_msg
            if metadata:
                entry["metadata"] = metadata
            self._traces.append(entry)
            self._guard.log_trace(entry)

    # ── Stats and trace access ───────────────────────────────────────

    def get_tool_stats(self) -> Dict[str, Dict[str, Any]]:
        """Return execution statistics per tool.

        Returns
        -------
        dict
            Keyed by tool name, each value has ``call_count``,
            ``error_count``, ``total_ms``, and ``avg_ms``.
        """
        result = {}
        for name, stats in self._tool_stats.items():
            count = stats["call_count"]
            result[name] = {
                **stats,
                "avg_ms": round(stats["total_ms"] / count, 2) if count > 0 else 0,
            }
        return result

    def get_traces(self) -> List[Dict[str, Any]]:
        """Return a copy of all collected trace entries."""
        return list(self._traces)

    def flush(self) -> None:
        """Clear the local trace buffer."""
        self._traces.clear()


def wrap_toolset(
    toolset: Any,
    *,
    api_key: str,
    project_id: Optional[str] = None,
    base_url: str = "https://evalguard.ai/api",
    rules: Optional[List[str]] = None,
    block_on_violation: bool = True,
    sensitive_patterns: Optional[List[str]] = None,
    redact_sensitive: bool = False,
    timeout: float = 5.0,
) -> Any:
    """Wrap a Composio ``ComposioToolSet`` to trace all tool executions.

    Patches ``execute_action`` so every tool call is automatically guarded
    and traced without changing application code.

    Parameters
    ----------
    toolset:
        A ``composio.ComposioToolSet`` instance.
    api_key:
        EvalGuard API key.
    project_id:
        Optional project ID for trace grouping.
    rules:
        Guardrail rules for input checking.
    block_on_violation:
        If *True*, block tool calls that violate guardrails.
    sensitive_patterns:
        Additional regex patterns for sensitive data detection.
    redact_sensitive:
        If *True*, redact sensitive data instead of blocking.

    Returns
    -------
    The same toolset instance with monitoring applied.
    """
    middleware = EvalGuardMiddleware(
        api_key=api_key,
        project_id=project_id,
        base_url=base_url,
        rules=rules,
        block_on_violation=block_on_violation,
        sensitive_patterns=sensitive_patterns,
        redact_sensitive=redact_sensitive,
        timeout=timeout,
    )

    # Patch execute_action
    original_execute = getattr(toolset, "execute_action", None)
    if original_execute is None:
        logger.warning("Toolset has no 'execute_action' method; skipping wrap")
        return toolset

    @functools.wraps(original_execute)
    def traced_execute(action: Any, params: Any = None, **kwargs: Any) -> Any:
        tool_name = str(action.value) if hasattr(action, "value") else str(action)
        actual_params = params if isinstance(params, dict) else {}

        return middleware.execute_with_guard(
            tool_name=tool_name,
            params=actual_params,
            execute_fn=lambda p: original_execute(action, p, **kwargs),
            metadata={"action_raw": str(action)[:200]},
        )

    toolset.execute_action = traced_execute

    # Also patch get_tools if it exists, to wrap returned tool functions
    original_get_tools = getattr(toolset, "get_tools", None)
    if original_get_tools is not None:

        @functools.wraps(original_get_tools)
        def traced_get_tools(*args: Any, **kwargs: Any) -> Any:
            tools = original_get_tools(*args, **kwargs)
            if isinstance(tools, list):
                for tool in tools:
                    _patch_composio_tool(tool, middleware)
            return tools

        toolset.get_tools = traced_get_tools

    # Expose middleware for stats access
    toolset._evalguard_middleware = middleware
    return toolset


def _patch_composio_tool(tool: Any, middleware: EvalGuardMiddleware) -> None:
    """Patch a single Composio tool function/object for tracing."""
    # Composio tools can be callables or objects with an execute method
    tool_name = getattr(tool, "name", None) or getattr(tool, "__name__", None) or str(type(tool).__name__)

    execute_fn = getattr(tool, "execute", None)
    if execute_fn is not None and callable(execute_fn):
        original = execute_fn

        @functools.wraps(original)
        def traced_execute(params: Any = None, **kwargs: Any) -> Any:
            actual_params = params if isinstance(params, dict) else {}
            return middleware.execute_with_guard(
                tool_name=tool_name,
                params=actual_params,
                execute_fn=lambda p: original(p, **kwargs),
            )

        tool.execute = traced_execute

    elif callable(tool):
        original_call = tool.__call__

        @functools.wraps(original_call)
        def traced_call(*args: Any, **kwargs: Any) -> Any:
            params = {"args": args, "kwargs": kwargs}
            return middleware.execute_with_guard(
                tool_name=tool_name,
                params=params,
                execute_fn=lambda p: original_call(*p.get("args", ()), **p.get("kwargs", {})),
            )

        tool.__call__ = traced_call


def _flatten_to_string(data: Any, max_len: int = 4000) -> str:
    """Convert arbitrary data to a flat string for guardrail checking."""
    if isinstance(data, str):
        return data[:max_len]
    if isinstance(data, dict):
        parts: List[str] = []
        for k, v in data.items():
            parts.append(f"{k}: {v}")
        return "\n".join(parts)[:max_len]
    if isinstance(data, (list, tuple)):
        return "\n".join(str(item) for item in data)[:max_len]
    return str(data)[:max_len]
