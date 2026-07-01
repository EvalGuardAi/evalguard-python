"""Drop-in Anthropic wrapper with EvalGuard guardrails.

Usage::

    from evalguard.anthropic import wrap
    from anthropic import Anthropic

    client = wrap(Anthropic(), api_key="eg_...", project_id="proj_...")
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[{"role": "user", "content": "Hello"}],
    )
"""

from __future__ import annotations

import time
from typing import Any, List, Optional

from .guardrails import GuardrailClient, GuardrailViolation


def wrap(
    client: Any,
    *,
    api_key: str,
    project_id: Optional[str] = None,
    base_url: str = "https://evalguard.ai/api",
    rules: Optional[List[str]] = None,
    block_on_violation: bool = True,
    timeout: float = 5.0,
) -> "_AnthropicProxy":
    """Wrap an ``anthropic.Anthropic`` client with guardrails.

    Parameters
    ----------
    client:
        An instantiated ``anthropic.Anthropic`` or ``anthropic.AsyncAnthropic``.
    api_key:
        EvalGuard API key.
    project_id:
        Optional EvalGuard project ID.
    rules:
        Guardrail rules.  Defaults to prompt-injection + PII.
    block_on_violation:
        Raise on violation if *True*, log-only if *False*.
    """
    guard = GuardrailClient(
        api_key=api_key,
        base_url=base_url,
        project_id=project_id,
        timeout=timeout,
    )
    return _AnthropicProxy(client, guard, rules, block_on_violation)


class _AnthropicProxy:
    """Transparent proxy that intercepts ``messages.create``."""

    __slots__ = ("_client", "_guard", "_rules", "_block")

    def __init__(self, client: Any, guard: GuardrailClient, rules: Optional[List[str]], block: bool) -> None:
        self._client = client
        self._guard = guard
        self._rules = rules
        self._block = block

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._client, name)
        if name == "messages":
            return _MessagesProxy(attr, self._guard, self._rules, self._block)
        return attr


class _MessagesProxy:
    __slots__ = ("_messages", "_guard", "_rules", "_block")

    def __init__(self, messages: Any, guard: GuardrailClient, rules: Optional[List[str]], block: bool) -> None:
        self._messages = messages
        self._guard = guard
        self._rules = rules
        self._block = block

    def __getattr__(self, name: str) -> Any:
        if name == "create":
            return self._guarded_create
        return getattr(self._messages, name)

    def _guarded_create(self, **kwargs: Any) -> Any:
        messages = kwargs.get("messages", [])
        system = kwargs.get("system", "")
        prompt_text = _extract_prompt(messages, system)
        model = kwargs.get("model", "unknown")

        # ── Pre-LLM check ────────────────────────────────────────────
        start = time.monotonic()
        check = self._guard.check_input(
            prompt_text,
            rules=self._rules,
            metadata={"model": model, "framework": "anthropic"},
        )
        guard_ms = (time.monotonic() - start) * 1000

        if not check.get("allowed", True):
            if self._block:
                raise GuardrailViolation(check.get("violations", []))

        # ── Call Anthropic ────────────────────────────────────────────
        start = time.monotonic()
        response = self._messages.create(**kwargs)
        llm_ms = (time.monotonic() - start) * 1000

        # ── Post-LLM trace ───────────────────────────────────────────
        output_text = _extract_response(response)
        self._guard.log_trace(
            {
                "provider": "anthropic",
                "model": model,
                "input": prompt_text,
                "output": output_text,
                "guard_latency_ms": round(guard_ms, 2),
                "llm_latency_ms": round(llm_ms, 2),
                "violations": check.get("violations", []),
                "token_usage": _extract_usage(response),
            }
        )

        return response


def _extract_prompt(messages: list, system: Any = "") -> str:
    """Build a single string from Anthropic-style messages + system prompt."""
    parts: list[str] = []
    if isinstance(system, str) and system:
        parts.append(system)
    elif isinstance(system, list):
        for block in system:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))

    for msg in messages:
        content = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block.get("text", ""))
    return "\n".join(parts)


def _extract_response(response: Any) -> str:
    """Extract text from an Anthropic Message response."""
    try:
        content = response.content if hasattr(response, "content") else response.get("content", [])
        parts: list[str] = []
        for block in content:
            if hasattr(block, "text"):
                parts.append(block.text)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "\n".join(parts)
    except Exception:
        return ""


def _extract_usage(response: Any) -> Optional[dict]:
    """Extract token usage from an Anthropic response."""
    try:
        usage = response.usage if hasattr(response, "usage") else response.get("usage")
        if usage:
            input_tokens = getattr(usage, "input_tokens", None) or (usage.get("input_tokens") if isinstance(usage, dict) else None)
            output_tokens = getattr(usage, "output_tokens", None) or (usage.get("output_tokens") if isinstance(usage, dict) else None)
            return {
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
                "total_tokens": (input_tokens or 0) + (output_tokens or 0),
            }
    except Exception:
        pass
    return None
