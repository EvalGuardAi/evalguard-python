"""Agno (formerly PHIData) integration for EvalGuard.

Usage::

    from evalguard.agno import EvalGuardTool, EvalGuardMonitor

    # Option 1: Give agents an EvalGuard tool for self-checking
    from agno.agent import Agent
    from agno.models.openai import OpenAIChat

    eg_tool = EvalGuardTool(api_key="eg_...", project_id="proj_...")
    agent = Agent(
        model=OpenAIChat(id="gpt-4o"),
        tools=[eg_tool],
        instructions=["Use evalguard_check before returning sensitive answers"],
    )

    # Option 2: Monitor all agent runs with a callback
    monitor = EvalGuardMonitor(api_key="eg_...", project_id="proj_...")
    agent = Agent(
        model=OpenAIChat(id="gpt-4o"),
        callbacks=[monitor],
    )
    agent.run("Summarize these financial documents")

    # Option 3: Standalone guardrail check
    monitor = EvalGuardMonitor(api_key="eg_...")
    result = monitor.check("Some text to validate")

Works with Agno >= 1.0.0.  No hard dependency on the ``agno`` package --
the integration duck-types against the expected interfaces.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Dict, List, Optional

from .guardrails import GuardrailClient, GuardrailViolation

logger = logging.getLogger("evalguard.agno")


class EvalGuardTool:
    """An Agno-compatible tool that agents can invoke to self-check outputs.

    When added to an Agno agent's ``tools`` list, the agent gains access to
    an ``evalguard_check`` function it can call to validate its own responses
    against EvalGuard guardrails before presenting them to the user.

    This class follows Agno's tool protocol: it exposes ``name``,
    ``description``, and a callable interface.  No import of ``agno`` is
    required.

    Parameters
    ----------
    api_key:
        EvalGuard API key.
    project_id:
        Optional project ID for trace grouping.
    rules:
        Guardrail rules applied when the agent calls the check.
    """

    # Agno tool metadata
    name: str = "evalguard_check"
    description: str = (
        "Check text for safety violations (prompt injection, PII, toxic content). "
        "Call this before returning sensitive answers to ensure compliance. "
        "Input: the text to check. Returns: {allowed: bool, violations: [...]}."
    )

    def __init__(
        self,
        api_key: str,
        project_id: Optional[str] = None,
        base_url: str = "https://evalguard.ai/api",
        rules: Optional[List[str]] = None,
        timeout: float = 5.0,
    ) -> None:
        self._guard = GuardrailClient(
            api_key=api_key,
            base_url=base_url,
            project_id=project_id,
            timeout=timeout,
        )
        self._rules = rules

    def __call__(self, text: str) -> Dict[str, Any]:
        """Run the guardrail check.  Called by the agent at inference time."""
        return self.run(text)

    def run(self, text: str) -> Dict[str, Any]:
        """Check *text* against EvalGuard guardrails.

        Returns
        -------
        dict
            ``{"allowed": bool, "violations": [...], "sanitized": str | None}``
        """
        result = self._guard.check_output(
            text,
            rules=self._rules,
            metadata={"framework": "agno", "source": "agent_self_check"},
        )
        return {
            "allowed": result.get("allowed", True),
            "violations": result.get("violations", []),
            "sanitized": result.get("sanitized"),
        }

    def get_definition(self) -> Dict[str, Any]:
        """Return the tool definition in Agno/OpenAI function-calling format.

        Agno agents use this to register the tool with the underlying LLM.
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "The text to check for safety violations.",
                        }
                    },
                    "required": ["text"],
                },
            },
        }


class EvalGuardMonitor:
    """Agno callback/monitor that captures agent runs and applies guardrails.

    Implements the Agno callback protocol by providing lifecycle hooks for
    agent runs, tool use, model calls, and reasoning steps.  Works as both
    a monitoring callback and a standalone guardrail checker.

    Parameters
    ----------
    api_key:
        EvalGuard API key.
    project_id:
        Optional project ID for trace grouping.
    rules:
        Guardrail rules for input/output checking.
    block_on_violation:
        If *True*, raises :class:`GuardrailViolation` when a check fails.
    check_inputs:
        If *True*, check agent inputs before execution.
    check_outputs:
        If *True*, check agent outputs after execution.
    """

    def __init__(
        self,
        api_key: str,
        project_id: Optional[str] = None,
        base_url: str = "https://evalguard.ai/api",
        rules: Optional[List[str]] = None,
        block_on_violation: bool = True,
        check_inputs: bool = True,
        check_outputs: bool = True,
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
        self._check_inputs = check_inputs
        self._check_outputs = check_outputs
        # Per-run state keyed by run_id
        self._runs: Dict[str, Dict[str, Any]] = {}

    # ── Standalone guardrail API ─────────────────────────────────────

    def check(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Check text against guardrails (usable outside the callback flow).

        Returns
        -------
        dict
            ``{"allowed": bool, "violations": [...]}``

        Raises
        ------
        GuardrailViolation
            If ``block_on_violation`` is *True* and the check fails.
        """
        result = self._guard.check_input(text, rules=self._rules, metadata=metadata)
        if not result.get("allowed", True) and self._block:
            raise GuardrailViolation(result.get("violations", []))
        return result

    # ── Agno callback protocol ───────────────────────────────────────

    def on_agent_start(
        self,
        *,
        agent: Any = None,
        run_id: Optional[str] = None,
        message: str = "",
        **kwargs: Any,
    ) -> None:
        """Called when an Agno agent run begins.

        Records the user message, checks it against guardrails, and
        initializes trajectory tracking for the run.
        """
        rid = str(run_id or uuid.uuid4())
        agent_name = getattr(agent, "name", "unknown") if agent else "unknown"
        model_id = ""
        if agent and hasattr(agent, "model"):
            model_id = getattr(agent.model, "id", str(agent.model))

        check: Dict[str, Any] = {"allowed": True, "violations": []}
        guard_ms = 0.0
        if self._check_inputs and message:
            start = time.monotonic()
            check = self._guard.check_input(
                message,
                rules=self._rules,
                metadata={
                    "agent_name": agent_name,
                    "model": model_id,
                    "framework": "agno",
                },
            )
            guard_ms = (time.monotonic() - start) * 1000

        self._runs[rid] = {
            "agent_name": agent_name,
            "model": model_id,
            "message": message,
            "guard_ms": guard_ms,
            "violations": check.get("violations", []),
            "start": time.monotonic(),
            "tool_calls": [],
            "model_calls": [],
            "reasoning_steps": [],
        }

        if not check.get("allowed", True) and self._block:
            raise GuardrailViolation(check.get("violations", []))

    def on_agent_end(
        self,
        *,
        agent: Any = None,
        run_id: Optional[str] = None,
        response: Any = None,
        **kwargs: Any,
    ) -> None:
        """Called when an Agno agent run completes.

        Logs the full run trace including tool calls, model calls, and
        reasoning steps.  Optionally checks the final output.
        """
        rid = str(run_id or "")
        run_data = self._runs.pop(rid, {})
        elapsed_ms = (time.monotonic() - run_data.get("start", time.monotonic())) * 1000

        # Extract response text
        output_text = _extract_agno_response(response)

        # Optional output check
        output_violations: List[Dict[str, Any]] = []
        if self._check_outputs and output_text:
            out_check = self._guard.check_output(
                output_text[:4000],
                metadata={
                    "agent_name": run_data.get("agent_name", "unknown"),
                    "framework": "agno",
                },
            )
            output_violations = out_check.get("violations", [])
            if not out_check.get("allowed", True) and self._block:
                raise GuardrailViolation(output_violations)

        self._guard.log_trace({
            "provider": "agno",
            "agent_name": run_data.get("agent_name", "unknown"),
            "model": run_data.get("model", "unknown"),
            "input": run_data.get("message", ""),
            "output": output_text[:2000],
            "guard_latency_ms": round(run_data.get("guard_ms", 0), 2),
            "agent_latency_ms": round(elapsed_ms, 2),
            "input_violations": run_data.get("violations", []),
            "output_violations": output_violations,
            "tool_calls": run_data.get("tool_calls", []),
            "model_calls": run_data.get("model_calls", []),
            "reasoning_steps": run_data.get("reasoning_steps", []),
        })

    def on_agent_error(
        self,
        *,
        agent: Any = None,
        run_id: Optional[str] = None,
        error: Optional[BaseException] = None,
        **kwargs: Any,
    ) -> None:
        """Called when an Agno agent run fails."""
        rid = str(run_id or "")
        run_data = self._runs.pop(rid, {})
        self._guard.log_trace({
            "provider": "agno",
            "agent_name": run_data.get("agent_name", "unknown"),
            "model": run_data.get("model", "unknown"),
            "input": run_data.get("message", ""),
            "output": "",
            "error": str(error) if error else "unknown_error",
            "tool_calls": run_data.get("tool_calls", []),
            "violations": run_data.get("violations", []),
        })

    def on_tool_call(
        self,
        *,
        run_id: Optional[str] = None,
        tool_name: str = "unknown",
        tool_args: Optional[Dict[str, Any]] = None,
        tool_result: Any = None,
        latency_ms: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        """Called when a tool is invoked during an agent run.

        Captures tool name, arguments, result, and latency for the trace.
        """
        rid = str(run_id or "")
        run_data = self._runs.get(rid)
        if run_data is None:
            return

        run_data["tool_calls"].append({
            "tool_name": tool_name,
            "args": tool_args or {},
            "result": str(tool_result)[:2000] if tool_result else "",
            "latency_ms": latency_ms,
            "timestamp": time.time(),
        })

    def on_model_call(
        self,
        *,
        run_id: Optional[str] = None,
        model: str = "unknown",
        prompt: str = "",
        response: str = "",
        tokens_used: Optional[Dict[str, int]] = None,
        latency_ms: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        """Called when the underlying LLM is invoked during an agent run.

        Captures model name, prompt/response excerpts, token usage, and
        latency.
        """
        rid = str(run_id or "")
        run_data = self._runs.get(rid)
        if run_data is None:
            return

        run_data["model_calls"].append({
            "model": model,
            "prompt": prompt[:1000],
            "response": response[:1000],
            "tokens": tokens_used or {},
            "latency_ms": latency_ms,
            "timestamp": time.time(),
        })

    def on_reasoning_step(
        self,
        *,
        run_id: Optional[str] = None,
        step_type: str = "reasoning",
        content: str = "",
        **kwargs: Any,
    ) -> None:
        """Called for intermediate reasoning or chain-of-thought steps."""
        rid = str(run_id or "")
        run_data = self._runs.get(rid)
        if run_data is None:
            return

        run_data["reasoning_steps"].append({
            "type": step_type,
            "content": content[:2000],
            "timestamp": time.time(),
        })


def _extract_agno_response(response: Any) -> str:
    """Extract text from an Agno RunResponse or similar object."""
    if response is None:
        return ""
    if isinstance(response, str):
        return response

    # Agno RunResponse has .content (str or list of Message)
    content = getattr(response, "content", None)
    if isinstance(content, str):
        return content

    # RunResponse may have .messages list
    messages = getattr(response, "messages", None)
    if messages and isinstance(messages, list):
        parts: list[str] = []
        for msg in messages:
            if isinstance(msg, str):
                parts.append(msg)
            elif hasattr(msg, "content"):
                parts.append(str(msg.content))
            else:
                parts.append(str(msg))
        return "\n".join(parts)

    # Fallback
    return str(response)[:4000]
