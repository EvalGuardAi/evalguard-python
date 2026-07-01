"""Strands Agents SDK integration for EvalGuard.

A Strands ``Agent`` is invoked by calling it directly — ``agent("prompt")``
returns an ``AgentResult``. :func:`guard` wraps that call so every invocation
is checked before the agent runs and traced after::

    from strands import Agent
    from evalguard.strands import guard

    agent = guard(Agent(model="..."), api_key="eg_...")
    result = agent("Summarise the attached report")

The wrapper forwards to :class:`~evalguard.guardrails.GuardrailClient` and
raises :class:`~evalguard.guardrails.GuardrailViolation` when
``block_on_violation`` is set and a pre-check fails. It does not import
``strands`` — the agent is duck-typed, so EvalGuard remains an optional
dependency.
"""

from __future__ import annotations

import functools
import time
from typing import Any, Callable, List, Optional

from .guardrails import GuardrailClient, GuardrailViolation


def _extract_prompt(prompt: Any) -> str:
    """Pull the user text from the first argument of an agent call."""
    if isinstance(prompt, str):
        return prompt
    # Strands also accepts a list of content blocks / messages.
    if isinstance(prompt, (list, tuple)):
        parts: List[str] = []
        for item in prompt:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)
    if isinstance(prompt, dict):
        text = prompt.get("text") or prompt.get("content")
        if isinstance(text, str):
            return text
    return str(prompt) if prompt is not None else ""


def _extract_result_text(result: Any) -> str:
    """Stringify a Strands ``AgentResult`` for trace logging."""
    if result is None:
        return ""
    # AgentResult.message is a dict like {"role": "assistant", "content": [...]}.
    message = getattr(result, "message", None)
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, list):
            parts = [c.get("text", "") for c in content if isinstance(c, dict)]
            joined = "".join(p for p in parts if isinstance(p, str))
            if joined.strip():
                return joined[:2000]
        if isinstance(content, str) and content.strip():
            return content[:2000]
    return str(result)[:2000]


def guard(
    agent: Any,
    *,
    api_key: str,
    project_id: Optional[str] = None,
    base_url: str = "https://evalguard.ai/api",
    rules: Optional[List[str]] = None,
    block_on_violation: bool = True,
    timeout: float = 5.0,
) -> Callable[..., Any]:
    """Wrap a Strands ``Agent`` (or any callable agent) with guard + tracing.

    Returns a callable with the same ``(prompt, *args, **kwargs)`` signature as
    the agent; call it in place of the agent. The first argument is treated as
    the user prompt and checked before the agent runs; the ``AgentResult`` is
    traced after.
    """
    guard_client = GuardrailClient(
        api_key=api_key,
        base_url=base_url,
        project_id=project_id,
        timeout=timeout,
    )

    @functools.wraps(agent if callable(agent) else guard)
    def guarded(prompt: Any = "", *args: Any, **kwargs: Any) -> Any:
        text = _extract_prompt(prompt)
        check = guard_client.check_input(
            text,
            rules=rules,
            metadata={"framework": "strands"},
        )
        if not check.get("allowed", True) and block_on_violation:
            raise GuardrailViolation(check.get("violations", []))

        start = time.monotonic()
        result = agent(prompt, *args, **kwargs)
        elapsed_ms = round((time.monotonic() - start) * 1000, 2)

        guard_client.log_trace(
            {
                "provider": "strands",
                "input": text[:2000],
                "output": _extract_result_text(result),
                "llm_latency_ms": elapsed_ms,
                "violations": check.get("violations", []),
            }
        )
        return result

    return guarded


# Descriptive alias mirroring the other framework wrappers.
guard_agent = guard

__all__: List[str] = ["guard", "guard_agent"]
