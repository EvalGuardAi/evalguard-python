"""CrewAI guardrail integration for EvalGuard.

Usage::

    from evalguard.crewai import EvalGuardGuardrail, guard_agent
    from crewai import Crew, Agent, Task

    # Option 1: Guard individual agents
    agent = guard_agent(
        Agent(role="researcher", goal="...", backstory="..."),
        api_key="eg_...",
    )

    # Option 2: Use as a crew-level guardrail
    guardrail = EvalGuardGuardrail(api_key="eg_...", project_id="proj_...")
    crew = Crew(agents=[agent], tasks=[...])
    # Call guardrail.check() before/after crew.kickoff()
"""

from __future__ import annotations

import functools
import time
from typing import Any, Callable, Dict, List, Optional

from .guardrails import GuardrailClient, GuardrailViolation


class EvalGuardGuardrail:
    """Standalone guardrail that can be used with CrewAI workflows.

    Parameters
    ----------
    api_key:
        EvalGuard API key.
    project_id:
        Optional project ID for trace grouping.
    rules:
        Guardrail rules for input checking.
    block_on_violation:
        If *True*, :meth:`check` raises on violation.
    """

    def __init__(
        self,
        api_key: str,
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
        self._rules = rules
        self._block = block_on_violation

    def check(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Check text against guardrails.

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

    def log(self, data: Dict[str, Any]) -> None:
        """Log a trace entry."""
        self._guard.log_trace(data)

    def wrap_function(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        """Decorator that guards a function's first string argument."""
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Find the first string argument to check
            text = ""
            for arg in args:
                if isinstance(arg, str):
                    text = arg
                    break
            if not text:
                for v in kwargs.values():
                    if isinstance(v, str):
                        text = v
                        break

            if text:
                self.check(text, metadata={"function": fn.__name__})

            start = time.monotonic()
            result = fn(*args, **kwargs)
            elapsed_ms = (time.monotonic() - start) * 1000

            self._guard.log_trace({
                "provider": "crewai",
                "function": fn.__name__,
                "input": text,
                "output": str(result)[:2000] if result else "",
                "llm_latency_ms": round(elapsed_ms, 2),
            })
            return result

        return wrapper


def guard_agent(
    agent: Any,
    *,
    api_key: str,
    project_id: Optional[str] = None,
    base_url: str = "https://evalguard.ai/api",
    rules: Optional[List[str]] = None,
    block_on_violation: bool = True,
    timeout: float = 5.0,
) -> Any:
    """Wrap a CrewAI ``Agent`` so that every task execution is guarded.

    This patches the agent's ``execute_task`` method to run guardrail
    checks before and trace logging after each task execution.

    Parameters
    ----------
    agent:
        A ``crewai.Agent`` instance.

    Returns
    -------
    The same agent instance, with guardrails applied.
    """
    guard = GuardrailClient(
        api_key=api_key,
        base_url=base_url,
        project_id=project_id,
        timeout=timeout,
    )
    guardrail_rules = rules
    block = block_on_violation

    original_execute = getattr(agent, "execute_task", None)
    if original_execute is None:
        return agent

    @functools.wraps(original_execute)
    def guarded_execute(task: Any, *args: Any, **kwargs: Any) -> Any:
        task_desc = getattr(task, "description", str(task)) if task else ""

        # Pre-check
        check = guard.check_input(
            task_desc,
            rules=guardrail_rules,
            metadata={
                "agent_role": getattr(agent, "role", "unknown"),
                "framework": "crewai",
            },
        )
        if not check.get("allowed", True) and block:
            raise GuardrailViolation(check.get("violations", []))

        # Execute
        start = time.monotonic()
        result = original_execute(task, *args, **kwargs)
        elapsed_ms = (time.monotonic() - start) * 1000

        # Trace
        guard.log_trace({
            "provider": "crewai",
            "agent_role": getattr(agent, "role", "unknown"),
            "input": task_desc[:2000],
            "output": str(result)[:2000] if result else "",
            "llm_latency_ms": round(elapsed_ms, 2),
            "violations": check.get("violations", []),
        })
        return result

    agent.execute_task = guarded_execute
    return agent
