"""Drop-in OpenAI wrapper with EvalGuard guardrails.

Usage::

    from evalguard.openai import wrap
    from openai import OpenAI

    client = wrap(OpenAI(), api_key="eg_...", project_id="proj_...")
    # Use exactly like normal -- guardrails are automatic
    response = client.chat.completions.create(
        model="gpt-4",
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
) -> "_OpenAIProxy":
    """Wrap an ``openai.OpenAI`` (or ``AsyncOpenAI``) client with guardrails.

    Parameters
    ----------
    client:
        An instantiated ``openai.OpenAI`` or ``openai.AsyncOpenAI`` client.
    api_key:
        EvalGuard API key.
    project_id:
        Optional EvalGuard project ID for trace grouping.
    rules:
        Guardrail rules to apply.  Defaults to prompt-injection + PII.
    block_on_violation:
        If *True*, raise :class:`GuardrailViolation` when input is blocked.
        If *False*, violations are logged but the request proceeds.
    """
    guard = GuardrailClient(
        api_key=api_key,
        base_url=base_url,
        project_id=project_id,
        timeout=timeout,
    )
    return _OpenAIProxy(client, guard, rules, block_on_violation)


class _OpenAIProxy:
    """Transparent proxy that intercepts ``chat.completions.create``."""

    __slots__ = ("_client", "_guard", "_rules", "_block")

    def __init__(
        self,
        client: Any,
        guard: GuardrailClient,
        rules: Optional[List[str]],
        block: bool,
    ) -> None:
        self._client = client
        self._guard = guard
        self._rules = rules
        self._block = block

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._client, name)
        if name == "chat":
            return _ChatProxy(attr, self._guard, self._rules, self._block)
        return attr


class _ChatProxy:
    __slots__ = ("_chat", "_guard", "_rules", "_block")

    def __init__(self, chat: Any, guard: GuardrailClient, rules: Optional[List[str]], block: bool) -> None:
        self._chat = chat
        self._guard = guard
        self._rules = rules
        self._block = block

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._chat, name)
        if name == "completions":
            return _CompletionsProxy(attr, self._guard, self._rules, self._block)
        return attr


class _CompletionsProxy:
    __slots__ = ("_completions", "_guard", "_rules", "_block")

    def __init__(self, completions: Any, guard: GuardrailClient, rules: Optional[List[str]], block: bool) -> None:
        self._completions = completions
        self._guard = guard
        self._rules = rules
        self._block = block

    def __getattr__(self, name: str) -> Any:
        if name == "create":
            return self._guarded_create
        return getattr(self._completions, name)

    def _guarded_create(self, **kwargs: Any) -> Any:
        messages = kwargs.get("messages", [])
        prompt_text = _extract_prompt(messages)
        model = kwargs.get("model", "unknown")

        # ── Pre-LLM check ────────────────────────────────────────────
        start = time.monotonic()
        check = self._guard.check_input(
            prompt_text,
            rules=self._rules,
            metadata={"model": model, "framework": "openai"},
        )
        guard_ms = (time.monotonic() - start) * 1000

        if not check.get("allowed", True):
            if self._block:
                raise GuardrailViolation(check.get("violations", []))
            # Non-blocking: log but continue

        # ── Call OpenAI ───────────────────────────────────────────────
        start = time.monotonic()
        response = self._completions.create(**kwargs)
        llm_ms = (time.monotonic() - start) * 1000

        # ── Post-LLM trace ───────────────────────────────────────────
        output_text = _extract_response(response)
        self._guard.log_trace(
            {
                "provider": "openai",
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


def _extract_prompt(messages: list) -> str:
    """Join all message contents into a single string for guardrail checking."""
    parts: list[str] = []
    for msg in messages:
        content = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            # Multi-modal messages: extract text parts
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    parts.append(part.get("text", ""))
    return "\n".join(parts)


def _extract_response(response: Any) -> str:
    """Extract text from an OpenAI ChatCompletion response."""
    try:
        choices = response.choices if hasattr(response, "choices") else response.get("choices", [])
        if choices:
            msg = choices[0].message if hasattr(choices[0], "message") else choices[0].get("message", {})
            return msg.content if hasattr(msg, "content") else msg.get("content", "")
    except Exception:
        pass
    return ""


def _extract_usage(response: Any) -> Optional[dict]:
    """Extract token usage from response."""
    try:
        usage = response.usage if hasattr(response, "usage") else response.get("usage")
        if usage:
            return {
                "prompt_tokens": getattr(usage, "prompt_tokens", None) or usage.get("prompt_tokens"),
                "completion_tokens": getattr(usage, "completion_tokens", None) or usage.get("completion_tokens"),
                "total_tokens": getattr(usage, "total_tokens", None) or usage.get("total_tokens"),
            }
    except Exception:
        pass
    return None
