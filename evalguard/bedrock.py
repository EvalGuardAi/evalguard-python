"""AWS Bedrock wrapper with EvalGuard guardrails.

Usage::

    from evalguard.bedrock import wrap
    import boto3

    bedrock = boto3.client("bedrock-runtime")
    client = wrap(bedrock, api_key="eg_...", project_id="proj_...")

    response = client.invoke_model(
        modelId="anthropic.claude-3-sonnet-20240229-v1:0",
        body='{"messages":[{"role":"user","content":"Hello"}],"max_tokens":256}',
    )
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional

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
) -> "_BedrockProxy":
    """Wrap a ``boto3.client('bedrock-runtime')`` with guardrails.

    Parameters
    ----------
    client:
        A boto3 Bedrock Runtime client.
    api_key:
        EvalGuard API key.
    project_id:
        Optional project ID.
    rules:
        Guardrail rules.  Defaults to prompt-injection + PII.
    block_on_violation:
        Raise on violation if *True*.
    """
    guard = GuardrailClient(
        api_key=api_key,
        base_url=base_url,
        project_id=project_id,
        timeout=timeout,
    )
    return _BedrockProxy(client, guard, rules, block_on_violation)


class _BedrockProxy:
    """Transparent proxy that intercepts ``invoke_model`` and ``converse``."""

    __slots__ = ("_client", "_guard", "_rules", "_block")

    def __init__(self, client: Any, guard: GuardrailClient, rules: Optional[List[str]], block: bool) -> None:
        self._client = client
        self._guard = guard
        self._rules = rules
        self._block = block

    def __getattr__(self, name: str) -> Any:
        if name == "invoke_model":
            return self._guarded_invoke_model
        if name == "converse":
            return self._guarded_converse
        return getattr(self._client, name)

    def _guarded_invoke_model(self, **kwargs: Any) -> Any:
        model_id = kwargs.get("modelId", kwargs.get("ModelId", "unknown"))
        body_raw = kwargs.get("body", kwargs.get("Body", "{}"))

        # Parse the body to extract prompt text
        body = json.loads(body_raw) if isinstance(body_raw, (str, bytes)) else body_raw
        prompt_text = _extract_invoke_prompt(body, model_id)

        # ── Pre-LLM check ────────────────────────────────────────────
        start = time.monotonic()
        check = self._guard.check_input(
            prompt_text,
            rules=self._rules,
            metadata={"model": model_id, "framework": "bedrock"},
        )
        guard_ms = (time.monotonic() - start) * 1000

        if not check.get("allowed", True):
            if self._block:
                raise GuardrailViolation(check.get("violations", []))

        # ── Call Bedrock ──────────────────────────────────────────────
        start = time.monotonic()
        response = self._client.invoke_model(**kwargs)
        llm_ms = (time.monotonic() - start) * 1000

        # ── Post-LLM trace ───────────────────────────────────────────
        output_text = _extract_invoke_response(response, model_id)
        self._guard.log_trace(
            {
                "provider": "bedrock",
                "model": model_id,
                "input": prompt_text,
                "output": output_text,
                "guard_latency_ms": round(guard_ms, 2),
                "llm_latency_ms": round(llm_ms, 2),
                "violations": check.get("violations", []),
            }
        )

        return response

    def _guarded_converse(self, **kwargs: Any) -> Any:
        model_id = kwargs.get("modelId", kwargs.get("ModelId", "unknown"))
        messages = kwargs.get("messages", [])
        system = kwargs.get("system", [])

        prompt_text = _extract_converse_prompt(messages, system)

        # ── Pre-LLM check ────────────────────────────────────────────
        start = time.monotonic()
        check = self._guard.check_input(
            prompt_text,
            rules=self._rules,
            metadata={"model": model_id, "framework": "bedrock"},
        )
        guard_ms = (time.monotonic() - start) * 1000

        if not check.get("allowed", True):
            if self._block:
                raise GuardrailViolation(check.get("violations", []))

        # ── Call Bedrock ──────────────────────────────────────────────
        start = time.monotonic()
        response = self._client.converse(**kwargs)
        llm_ms = (time.monotonic() - start) * 1000

        # ── Post-LLM trace ───────────────────────────────────────────
        output_text = _extract_converse_response(response)
        usage = response.get("usage", {})
        self._guard.log_trace(
            {
                "provider": "bedrock",
                "model": model_id,
                "input": prompt_text,
                "output": output_text,
                "guard_latency_ms": round(guard_ms, 2),
                "llm_latency_ms": round(llm_ms, 2),
                "violations": check.get("violations", []),
                "token_usage": {
                    "prompt_tokens": usage.get("inputTokens"),
                    "completion_tokens": usage.get("outputTokens"),
                    "total_tokens": usage.get("totalTokens"),
                } if usage else None,
            }
        )

        return response


def _extract_invoke_prompt(body: dict, model_id: str) -> str:
    """Extract prompt from Bedrock invoke_model body (varies by model provider)."""
    # Anthropic Claude via Bedrock
    if "anthropic" in model_id.lower():
        messages = body.get("messages", [])
        system = body.get("system", "")
        parts: list[str] = []
        if system:
            parts.append(system if isinstance(system, str) else str(system))
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                parts.append(content)
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        parts.append(block.get("text", ""))
        return "\n".join(parts)

    # Amazon Titan
    if "titan" in model_id.lower():
        config = body.get("inputText", "")
        return config if isinstance(config, str) else str(config)

    # Meta Llama
    if "meta" in model_id.lower() or "llama" in model_id.lower():
        return body.get("prompt", str(body))

    # Cohere
    if "cohere" in model_id.lower():
        return body.get("prompt", body.get("message", str(body)))

    # AI21
    if "ai21" in model_id.lower():
        return body.get("prompt", str(body))

    # Mistral
    if "mistral" in model_id.lower():
        return body.get("prompt", str(body))

    # Fallback
    return body.get("prompt", body.get("inputText", str(body)))


def _extract_invoke_response(response: dict, model_id: str) -> str:
    """Extract output text from Bedrock invoke_model response."""
    try:
        body = response.get("body")
        if body is None:
            return ""
        # StreamingBody needs to be read
        if hasattr(body, "read"):
            raw = body.read()
            data = json.loads(raw)
        else:
            data = json.loads(body) if isinstance(body, (str, bytes)) else body

        # Anthropic
        if "anthropic" in model_id.lower():
            content = data.get("content", [])
            return "\n".join(
                b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"
            )

        # Titan
        if "titan" in model_id.lower():
            results = data.get("results", [{}])
            return results[0].get("outputText", "") if results else ""

        # Llama
        if "meta" in model_id.lower() or "llama" in model_id.lower():
            return data.get("generation", "")

        # Cohere
        if "cohere" in model_id.lower():
            return data.get("text", data.get("generations", [{}])[0].get("text", ""))

        return data.get("completion", data.get("generation", str(data)))
    except Exception:
        return ""


def _extract_converse_prompt(messages: list, system: Any = None) -> str:
    """Extract prompt from Bedrock Converse API messages."""
    parts: list[str] = []
    if system:
        for block in (system if isinstance(system, list) else [system]):
            if isinstance(block, dict):
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)

    for msg in messages:
        content = msg.get("content", [])
        for block in (content if isinstance(content, list) else [content]):
            if isinstance(block, dict):
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
    return "\n".join(parts)


def _extract_converse_response(response: dict) -> str:
    """Extract output from Bedrock Converse API response."""
    try:
        output = response.get("output", {})
        message = output.get("message", {})
        content = message.get("content", [])
        return "\n".join(
            b.get("text", "") for b in content if isinstance(b, dict)
        )
    except Exception:
        return ""
