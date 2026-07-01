"""NVIDIA NeMo Guardrails / OpenClaw agent integration for EvalGuard.

Usage::

    from evalguard.nemoclaw import EvalGuardAgent

    agent = EvalGuardAgent(api_key="eg_...", agent_name="support-bot")

    # Guard any LLM call regardless of provider
    result = agent.guarded_call(
        provider="openai",
        messages=[{"role": "user", "content": "Hello"}],
        llm_fn=lambda: openai_client.chat.completions.create(
            model="gpt-4", messages=[{"role": "user", "content": "Hello"}]
        ),
    )

    # Or use as a context manager for multi-step agent workflows
    with agent.session("ticket-123") as session:
        session.check("User says: reset my password")
        result = do_llm_call(...)
        session.log_step("password_reset", input="...", output=str(result))
"""

from __future__ import annotations

import time
import uuid
from contextlib import contextmanager
from typing import Any, Callable, Dict, Generator, List, Optional

from .guardrails import GuardrailClient, GuardrailViolation


class EvalGuardAgent:
    """Agent-level guardrail wrapper for NeMo/OpenClaw-style agent systems.

    Parameters
    ----------
    api_key:
        EvalGuard API key.
    agent_name:
        A human-readable name for this agent (used in traces).
    project_id:
        Optional project ID.
    rules:
        Default guardrail rules.
    block_on_violation:
        Raise on violation if *True*.
    """

    def __init__(
        self,
        api_key: str,
        agent_name: str = "default",
        project_id: Optional[str] = None,
        base_url: str = "https://evalguard.ai/api",
        rules: Optional[List[str]] = None,
        block_on_violation: bool = True,
        timeout: float = 5.0,
    ) -> None:
        self._guard = GuardrailClient(
            api_key=api_key,
            base_url=base_url,
            project_id=project_id,
            timeout=timeout,
        )
        self._agent_name = agent_name
        self._rules = rules
        self._block = block_on_violation

    def guarded_call(
        self,
        provider: str,
        messages: List[Dict[str, str]],
        llm_fn: Callable[[], Any],
        *,
        rules: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Execute an LLM call with pre/post guardrail checks.

        Parameters
        ----------
        provider:
            LLM provider name (``"openai"``, ``"anthropic"``, etc.).
        messages:
            The messages being sent (for guardrail input checking).
        llm_fn:
            A zero-argument callable that performs the actual LLM call.
        rules:
            Override default rules for this call.
        metadata:
            Additional metadata for the trace.

        Returns
        -------
        The result of ``llm_fn()``.
        """
        prompt_text = "\n".join(
            msg.get("content", "") for msg in messages if isinstance(msg, dict)
        )

        # ── Pre-LLM check ────────────────────────────────────────────
        start = time.monotonic()
        check = self._guard.check_input(
            prompt_text,
            rules=rules or self._rules,
            metadata={
                "agent": self._agent_name,
                "provider": provider,
                "framework": "nemoclaw",
                **(metadata or {}),
            },
        )
        guard_ms = (time.monotonic() - start) * 1000

        if not check.get("allowed", True) and self._block:
            raise GuardrailViolation(check.get("violations", []))

        # ── Execute LLM call ─────────────────────────────────────────
        start = time.monotonic()
        result = llm_fn()
        llm_ms = (time.monotonic() - start) * 1000

        # ── Post-LLM trace ───────────────────────────────────────────
        output_text = str(result)[:2000] if result else ""
        self._guard.log_trace(
            {
                "provider": provider,
                "agent": self._agent_name,
                "framework": "nemoclaw",
                "input": prompt_text,
                "output": output_text,
                "guard_latency_ms": round(guard_ms, 2),
                "llm_latency_ms": round(llm_ms, 2),
                "violations": check.get("violations", []),
                **(metadata or {}),
            }
        )

        return result

    @contextmanager
    def session(self, session_id: Optional[str] = None) -> Generator["_AgentSession", None, None]:
        """Create a guarded session for multi-step agent workflows.

        Usage::

            with agent.session("ticket-123") as s:
                s.check("user input here")
                result = my_llm_call()
                s.log_step("step_name", input="...", output="...")
        """
        sid = session_id or str(uuid.uuid4())
        sess = _AgentSession(self._guard, self._agent_name, sid, self._rules, self._block)
        try:
            yield sess
        finally:
            sess._finalize()


class _AgentSession:
    """A multi-step guarded session within an agent."""

    __slots__ = ("_guard", "_agent_name", "_session_id", "_rules", "_block", "_steps")

    def __init__(
        self,
        guard: GuardrailClient,
        agent_name: str,
        session_id: str,
        rules: Optional[List[str]],
        block: bool,
    ) -> None:
        self._guard = guard
        self._agent_name = agent_name
        self._session_id = session_id
        self._rules = rules
        self._block = block
        self._steps: List[Dict[str, Any]] = []

    def check(self, text: str, *, rules: Optional[List[str]] = None) -> Dict[str, Any]:
        """Run a guardrail check within this session."""
        result = self._guard.check_input(
            text,
            rules=rules or self._rules,
            metadata={
                "agent": self._agent_name,
                "session_id": self._session_id,
                "framework": "nemoclaw",
            },
        )
        if not result.get("allowed", True) and self._block:
            raise GuardrailViolation(result.get("violations", []))
        return result

    def log_step(
        self,
        step_name: str,
        *,
        input: str = "",
        output: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log a single step in the agent workflow."""
        step = {
            "step": step_name,
            "input": input[:2000],
            "output": output[:2000],
            **(metadata or {}),
        }
        self._steps.append(step)
        self._guard.log_trace(
            {
                "provider": "nemoclaw",
                "agent": self._agent_name,
                "session_id": self._session_id,
                "framework": "nemoclaw",
                **step,
            }
        )

    def _finalize(self) -> None:
        """Log session summary on exit."""
        self._guard.log_trace(
            {
                "provider": "nemoclaw",
                "agent": self._agent_name,
                "session_id": self._session_id,
                "framework": "nemoclaw",
                "event": "session_end",
                "total_steps": len(self._steps),
            }
        )


# Convenience alias
def init(
    api_key: str,
    agent_name: str = "default",
    **kwargs: Any,
) -> EvalGuardAgent:
    """Shorthand for creating an EvalGuardAgent.

    Usage::

        from evalguard.nemoclaw import init
        agent = init(api_key="eg_...", agent_name="support-bot")
    """
    return EvalGuardAgent(api_key=api_key, agent_name=agent_name, **kwargs)
