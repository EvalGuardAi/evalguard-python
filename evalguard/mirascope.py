"""Mirascope integration for EvalGuard.

Provides a decorator and middleware for Mirascope's Pythonic LLM library,
auto-capturing prompt templates, model info, responses, token usage, and
latency, then sending traces to EvalGuard.

Usage::

    from mirascope.core import openai
    from evalguard.mirascope import evalguard_trace, EvalGuardMirascopeMiddleware

    # Option 1: Decorator that wraps @llm.call
    @evalguard_trace(api_key="eg_...", project_id="proj_...")
    @openai.call("gpt-4o")
    def summarize(text: str) -> str:
        return f"Summarize: {text}"

    response = summarize("Long article...")
    # Trace automatically sent to EvalGuard

    # Option 2: Guardrail middleware
    middleware = EvalGuardMirascopeMiddleware(
        api_key="eg_...",
        rules=["prompt_injection", "pii_redact"],
    )
    # Use with Mirascope's middleware system

Requires ``mirascope`` to be installed (``pip install mirascope``).
"""

from __future__ import annotations

import functools
import inspect
import logging
import time
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union, overload

from .guardrails import GuardrailClient, GuardrailViolation

logger = logging.getLogger("evalguard.mirascope")

F = TypeVar("F", bound=Callable[..., Any])


# ── Decorator ───────────────────────────────────────────────────────────


@overload
def evalguard_trace(fn: F) -> F: ...


@overload
def evalguard_trace(
    *,
    api_key: Optional[str] = None,
    project_id: Optional[str] = None,
    base_url: str = "https://evalguard.ai/api",
    rules: Optional[List[str]] = None,
    block_on_violation: bool = True,
    check_output: bool = False,
    timeout: float = 5.0,
) -> Callable[[F], F]: ...


def evalguard_trace(
    fn: Optional[F] = None,
    *,
    api_key: Optional[str] = None,
    project_id: Optional[str] = None,
    base_url: str = "https://evalguard.ai/api",
    rules: Optional[List[str]] = None,
    block_on_violation: bool = True,
    check_output: bool = False,
    timeout: float = 5.0,
) -> Union[F, Callable[[F], F]]:
    """Decorator that traces a Mirascope ``@llm.call``-decorated function.

    Captures the prompt template, model, response, token usage, and latency,
    then sends a trace to EvalGuard.  Optionally runs guardrail checks on
    inputs and/or outputs.

    Can be used bare (with env vars) or with explicit arguments::

        @evalguard_trace
        @openai.call("gpt-4o")
        def my_fn(text: str) -> str: ...

        @evalguard_trace(api_key="eg_...", rules=["pii_redact"])
        @openai.call("gpt-4o")
        def my_fn(text: str) -> str: ...

    Parameters
    ----------
    api_key:
        EvalGuard API key.  Falls back to ``EVALGUARD_API_KEY`` env var.
    project_id:
        EvalGuard project ID.  Falls back to ``EVALGUARD_PROJECT_ID`` env var.
    base_url:
        EvalGuard API base URL.
    rules:
        Guardrail rules to apply on input before the LLM call.
    block_on_violation:
        Raise :class:`GuardrailViolation` when input is blocked.
    check_output:
        If *True*, also run guardrail checks on the LLM output.
    timeout:
        HTTP timeout for guardrail calls.
    """
    import os

    resolved_key = api_key or os.environ.get("EVALGUARD_API_KEY", "")
    resolved_project = project_id or os.environ.get("EVALGUARD_PROJECT_ID")
    resolved_base = base_url or os.environ.get("EVALGUARD_BASE_URL", "https://evalguard.ai/api")

    def decorator(func: F) -> F:
        guard: Optional[GuardrailClient] = None
        if resolved_key:
            guard = GuardrailClient(
                api_key=resolved_key,
                base_url=resolved_base,
                project_id=resolved_project,
                timeout=timeout,
            )

        if inspect.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                return await _trace_call_async(
                    func, args, kwargs, guard, rules, block_on_violation, check_output
                )
            return async_wrapper  # type: ignore[return-value]
        else:
            @functools.wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                return _trace_call_sync(
                    func, args, kwargs, guard, rules, block_on_violation, check_output
                )
            return sync_wrapper  # type: ignore[return-value]

    if fn is not None:
        return decorator(fn)
    return decorator  # type: ignore[return-value]


def _extract_mirascope_info(func: Callable[..., Any], args: tuple, kwargs: dict) -> Dict[str, Any]:
    """Extract Mirascope-specific metadata from a decorated function."""
    info: Dict[str, Any] = {
        "function_name": getattr(func, "__qualname__", getattr(func, "__name__", "unknown")),
    }

    # Mirascope decorators often attach metadata to the function
    for attr in ("_model", "model", "__mirascope_model__"):
        val = getattr(func, attr, None)
        if val is not None:
            info["model"] = str(val)
            break

    for attr in ("_provider", "provider", "__mirascope_provider__"):
        val = getattr(func, attr, None)
        if val is not None:
            info["provider"] = str(val)
            break

    # Try to capture prompt template from the function source or docstring
    try:
        sig = inspect.signature(func)
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        info["inputs"] = {k: str(v)[:2000] for k, v in bound.arguments.items()}
    except Exception:
        if args:
            info["inputs"] = {"args": [str(a)[:500] for a in args]}

    return info


def _extract_response_data(response: Any) -> Dict[str, Any]:
    """Extract content, token usage, and model info from a Mirascope response."""
    data: Dict[str, Any] = {}

    # Mirascope response objects have .content for the text
    content = getattr(response, "content", None)
    if content is not None:
        data["output"] = str(content)[:4096]
    else:
        data["output"] = str(response)[:4096] if response else ""

    # Model info
    model = getattr(response, "model", None)
    if model:
        data["model"] = str(model)

    # Token usage -- Mirascope wraps provider-specific usage
    usage = getattr(response, "usage", None)
    if usage is not None:
        token_data: Dict[str, Any] = {}
        for attr_name, key in [
            ("prompt_tokens", "prompt_tokens"),
            ("input_tokens", "prompt_tokens"),
            ("completion_tokens", "completion_tokens"),
            ("output_tokens", "completion_tokens"),
            ("total_tokens", "total_tokens"),
        ]:
            val = getattr(usage, attr_name, None)
            if val is not None and key not in token_data:
                token_data[key] = val
        if token_data:
            data["token_usage"] = token_data

    # Cost if available
    cost = getattr(response, "cost", None)
    if cost is not None:
        data["cost"] = cost

    # Tool calls
    tool_calls = getattr(response, "tools", None) or getattr(response, "tool_calls", None)
    if tool_calls:
        data["tool_calls"] = [str(tc)[:500] for tc in tool_calls]

    return data


def _build_prompt_text(func: Callable[..., Any], args: tuple, kwargs: dict) -> str:
    """Build the prompt text sent to the LLM for guardrail checking."""
    # Try calling the function's prompt template method
    prompt_template = getattr(func, "_prompt_template", None) or getattr(func, "prompt_template", None)
    if callable(prompt_template):
        try:
            return str(prompt_template(*args, **kwargs))
        except Exception:
            pass

    # Fall back to joining string arguments
    parts: List[str] = []
    for arg in args:
        if isinstance(arg, str):
            parts.append(arg)
    for v in kwargs.values():
        if isinstance(v, str):
            parts.append(v)
    return "\n".join(parts) if parts else ""


def _trace_call_sync(
    func: Callable[..., Any],
    args: tuple,
    kwargs: dict,
    guard: Optional[GuardrailClient],
    rules: Optional[List[str]],
    block_on_violation: bool,
    check_output: bool,
) -> Any:
    """Synchronous trace wrapper."""
    info = _extract_mirascope_info(func, args, kwargs)
    violations: List[Dict[str, Any]] = []

    # Pre-call guardrail check
    if guard and rules:
        prompt_text = _build_prompt_text(func, args, kwargs)
        if prompt_text:
            check = guard.check_input(
                prompt_text,
                rules=rules,
                metadata={"framework": "mirascope", "function": info.get("function_name", "")},
            )
            violations = check.get("violations", [])
            if not check.get("allowed", True) and block_on_violation:
                raise GuardrailViolation(violations)

    # Execute the Mirascope call
    start = time.monotonic()
    error_str: Optional[str] = None
    response = None
    try:
        response = func(*args, **kwargs)
    except GuardrailViolation:
        raise
    except Exception as exc:
        error_str = f"{type(exc).__name__}: {exc}"
        elapsed_ms = (time.monotonic() - start) * 1000
        if guard:
            guard.log_trace({
                "provider": "mirascope",
                "function": info.get("function_name", ""),
                "model": info.get("model", "unknown"),
                "inputs": info.get("inputs", {}),
                "output": "",
                "error": error_str,
                "llm_latency_ms": round(elapsed_ms, 2),
                "violations": violations,
            })
        raise

    elapsed_ms = (time.monotonic() - start) * 1000

    # Extract response data
    resp_data = _extract_response_data(response)

    # Post-call output guardrail check
    output_violations: List[Dict[str, Any]] = []
    if guard and check_output and resp_data.get("output"):
        out_check = guard.check_output(
            resp_data["output"],
            rules=["toxic_content", "pii_redact"],
            metadata={"framework": "mirascope", "function": info.get("function_name", "")},
        )
        output_violations = out_check.get("violations", [])
        if not out_check.get("allowed", True) and block_on_violation:
            raise GuardrailViolation(output_violations)

    # Log trace
    if guard:
        guard.log_trace({
            "provider": "mirascope",
            "function": info.get("function_name", ""),
            "model": resp_data.get("model", info.get("model", "unknown")),
            "inputs": info.get("inputs", {}),
            "output": resp_data.get("output", ""),
            "llm_latency_ms": round(elapsed_ms, 2),
            "token_usage": resp_data.get("token_usage"),
            "cost": resp_data.get("cost"),
            "tool_calls": resp_data.get("tool_calls"),
            "violations": violations + output_violations,
        })

    return response


async def _trace_call_async(
    func: Callable[..., Any],
    args: tuple,
    kwargs: dict,
    guard: Optional[GuardrailClient],
    rules: Optional[List[str]],
    block_on_violation: bool,
    check_output: bool,
) -> Any:
    """Async trace wrapper."""
    info = _extract_mirascope_info(func, args, kwargs)
    violations: List[Dict[str, Any]] = []

    # Pre-call guardrail check
    if guard and rules:
        prompt_text = _build_prompt_text(func, args, kwargs)
        if prompt_text:
            check = guard.check_input(
                prompt_text,
                rules=rules,
                metadata={"framework": "mirascope", "function": info.get("function_name", "")},
            )
            violations = check.get("violations", [])
            if not check.get("allowed", True) and block_on_violation:
                raise GuardrailViolation(violations)

    # Execute the Mirascope call
    start = time.monotonic()
    error_str: Optional[str] = None
    response = None
    try:
        response = await func(*args, **kwargs)
    except GuardrailViolation:
        raise
    except Exception as exc:
        error_str = f"{type(exc).__name__}: {exc}"
        elapsed_ms = (time.monotonic() - start) * 1000
        if guard:
            guard.log_trace({
                "provider": "mirascope",
                "function": info.get("function_name", ""),
                "model": info.get("model", "unknown"),
                "inputs": info.get("inputs", {}),
                "output": "",
                "error": error_str,
                "llm_latency_ms": round(elapsed_ms, 2),
                "violations": violations,
            })
        raise

    elapsed_ms = (time.monotonic() - start) * 1000

    # Extract response data
    resp_data = _extract_response_data(response)

    # Post-call output guardrail check
    output_violations: List[Dict[str, Any]] = []
    if guard and check_output and resp_data.get("output"):
        out_check = guard.check_output(
            resp_data["output"],
            rules=["toxic_content", "pii_redact"],
            metadata={"framework": "mirascope", "function": info.get("function_name", "")},
        )
        output_violations = out_check.get("violations", [])
        if not out_check.get("allowed", True) and block_on_violation:
            raise GuardrailViolation(output_violations)

    # Log trace
    if guard:
        guard.log_trace({
            "provider": "mirascope",
            "function": info.get("function_name", ""),
            "model": resp_data.get("model", info.get("model", "unknown")),
            "inputs": info.get("inputs", {}),
            "output": resp_data.get("output", ""),
            "llm_latency_ms": round(elapsed_ms, 2),
            "token_usage": resp_data.get("token_usage"),
            "cost": resp_data.get("cost"),
            "tool_calls": resp_data.get("tool_calls"),
            "violations": violations + output_violations,
        })

    return response


# ── Middleware ───────────────────────────────────────────────────────────


class EvalGuardMirascopeMiddleware:
    """Guardrail middleware for Mirascope call pipelines.

    Can be used as a pre/post processor in Mirascope's middleware system
    or manually in application code.

    Usage::

        middleware = EvalGuardMirascopeMiddleware(api_key="eg_...")

        # Manual usage
        middleware.before_call("What is the meaning of life?")
        response = my_mirascope_fn("What is the meaning of life?")
        middleware.after_call(response)

    Parameters
    ----------
    api_key:
        EvalGuard API key.
    project_id:
        EvalGuard project ID.
    rules:
        Guardrail rules for input checking.
    output_rules:
        Guardrail rules for output checking.
    block_on_violation:
        Raise on violation.
    """

    def __init__(
        self,
        api_key: str,
        project_id: Optional[str] = None,
        base_url: str = "https://evalguard.ai/api",
        rules: Optional[List[str]] = None,
        output_rules: Optional[List[str]] = None,
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
        self._output_rules = output_rules or ["toxic_content", "pii_redact"]
        self._block = block_on_violation

    def before_call(
        self,
        prompt: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Check input before the LLM call.

        Returns
        -------
        dict
            ``{"allowed": bool, "violations": [...], "sanitized": str | None}``

        Raises
        ------
        GuardrailViolation
            If blocked and ``block_on_violation`` is True.
        """
        meta = {"framework": "mirascope", **(metadata or {})}
        result = self._guard.check_input(prompt, rules=self._rules, metadata=meta)
        if not result.get("allowed", True) and self._block:
            raise GuardrailViolation(result.get("violations", []))
        return result

    def after_call(
        self,
        response: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Check LLM output after the call.

        Parameters
        ----------
        response:
            Either a string or a Mirascope response object.

        Returns
        -------
        dict
            ``{"allowed": bool, "violations": [...], "sanitized": str | None}``
        """
        output_text = getattr(response, "content", None)
        if output_text is None:
            output_text = str(response) if response else ""

        meta = {"framework": "mirascope", **(metadata or {})}
        result = self._guard.check_output(output_text, rules=self._output_rules, metadata=meta)
        if not result.get("allowed", True) and self._block:
            raise GuardrailViolation(result.get("violations", []))
        return result

    def log(self, data: Dict[str, Any]) -> None:
        """Log a trace entry."""
        self._guard.log_trace(data)
