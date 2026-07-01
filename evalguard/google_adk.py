"""Google ADK (Agent Development Kit) integration for EvalGuard.

Usage::

    from evalguard.google_adk import EvalGuardAgentCallback, guard_adk_agent

    # Option 1: Use as a callback on any ADK agent
    from google.adk.agents import Agent

    callback = EvalGuardAgentCallback(api_key="eg_...", project_id="proj_...")
    agent = Agent(name="my_agent", model="gemini-2.0-flash", callbacks=[callback])

    # Option 2: Wrap an existing agent for automatic guardrails
    agent = Agent(name="my_agent", model="gemini-2.0-flash")
    guarded = guard_adk_agent(agent, api_key="eg_...")

    # Option 3: Use as an ADK before_tool_callback / after_tool_callback
    from evalguard.google_adk import evalguard_before_tool, evalguard_after_tool

    agent = Agent(
        name="my_agent",
        model="gemini-2.0-flash",
        before_tool_callback=evalguard_before_tool(api_key="eg_..."),
        after_tool_callback=evalguard_after_tool(api_key="eg_..."),
    )

Works with Google ADK >= 0.1.0.  No hard dependency on the ``google-adk``
package -- the integration duck-types against the callback protocol.
"""

from __future__ import annotations

import functools
import logging
import time
import uuid
from typing import Any, Callable, Dict, List, Optional

from .guardrails import GuardrailClient, GuardrailViolation

logger = logging.getLogger("evalguard.google_adk")


class EvalGuardAgentCallback:
    """Google ADK callback that traces agent execution and applies guardrails.

    Implements the ADK callback protocol by providing lifecycle hooks for
    agent invocations, tool calls, model responses, and thinking steps.
    No import of ``google.adk`` is required -- the class duck-types against
    the expected interface.

    Parameters
    ----------
    api_key:
        EvalGuard API key.
    project_id:
        Optional project ID for trace grouping.
    rules:
        Guardrail rules for input/output checking.
    block_on_violation:
        Raise :class:`GuardrailViolation` when a check fails.
    check_tool_inputs:
        If *True*, guardrail-check tool call arguments before execution.
    check_model_outputs:
        If *True*, guardrail-check model responses after generation.
    """

    def __init__(
        self,
        api_key: str,
        project_id: Optional[str] = None,
        base_url: str = "https://evalguard.ai/api",
        rules: Optional[List[str]] = None,
        block_on_violation: bool = True,
        check_tool_inputs: bool = True,
        check_model_outputs: bool = True,
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
        self._check_tool_inputs = check_tool_inputs
        self._check_model_outputs = check_model_outputs
        # Per-invocation state keyed by invocation_id
        self._invocations: Dict[str, Dict[str, Any]] = {}

    # ── Agent lifecycle callbacks ────────────────────────────────────

    def on_agent_start(
        self,
        *,
        agent_name: str = "unknown",
        invocation_id: Optional[str] = None,
        user_message: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Called when an ADK agent invocation begins.

        Checks the user message against guardrails and records the start
        of the trajectory trace.
        """
        inv_id = str(invocation_id or uuid.uuid4())
        start = time.monotonic()
        check = self._guard.check_input(
            user_message,
            rules=self._rules,
            metadata={
                "agent_name": agent_name,
                "framework": "google_adk",
                **(metadata or {}),
            },
        )
        guard_ms = (time.monotonic() - start) * 1000

        self._invocations[inv_id] = {
            "agent_name": agent_name,
            "user_message": user_message,
            "guard_ms": guard_ms,
            "violations": check.get("violations", []),
            "start": time.monotonic(),
            "steps": [],
            "tool_calls": [],
        }

        if not check.get("allowed", True) and self._block:
            raise GuardrailViolation(check.get("violations", []))

    def on_agent_end(
        self,
        *,
        invocation_id: Optional[str] = None,
        agent_name: str = "unknown",
        final_response: str = "",
        **kwargs: Any,
    ) -> None:
        """Called when an ADK agent invocation completes.

        Logs the full trajectory trace to EvalGuard including all
        intermediate steps, tool calls, and the final response.
        """
        inv_id = str(invocation_id or "")
        inv_data = self._invocations.pop(inv_id, {})
        elapsed_ms = (time.monotonic() - inv_data.get("start", time.monotonic())) * 1000

        # Optional output guardrail check
        output_violations: List[Dict[str, Any]] = []
        if self._check_model_outputs and final_response:
            out_check = self._guard.check_output(
                final_response,
                metadata={"agent_name": agent_name, "framework": "google_adk"},
            )
            output_violations = out_check.get("violations", [])
            if not out_check.get("allowed", True) and self._block:
                raise GuardrailViolation(output_violations)

        self._guard.log_trace({
            "provider": "google_adk",
            "agent_name": inv_data.get("agent_name", agent_name),
            "input": inv_data.get("user_message", ""),
            "output": final_response[:4000],
            "guard_latency_ms": round(inv_data.get("guard_ms", 0), 2),
            "agent_latency_ms": round(elapsed_ms, 2),
            "input_violations": inv_data.get("violations", []),
            "output_violations": output_violations,
            "steps": inv_data.get("steps", []),
            "tool_calls": inv_data.get("tool_calls", []),
        })

    def on_agent_error(
        self,
        *,
        invocation_id: Optional[str] = None,
        agent_name: str = "unknown",
        error: Optional[BaseException] = None,
        **kwargs: Any,
    ) -> None:
        """Called when an ADK agent invocation fails."""
        inv_id = str(invocation_id or "")
        inv_data = self._invocations.pop(inv_id, {})
        self._guard.log_trace({
            "provider": "google_adk",
            "agent_name": inv_data.get("agent_name", agent_name),
            "input": inv_data.get("user_message", ""),
            "output": "",
            "error": str(error) if error else "unknown_error",
            "steps": inv_data.get("steps", []),
            "tool_calls": inv_data.get("tool_calls", []),
            "violations": inv_data.get("violations", []),
        })

    # ── Tool call callbacks ──────────────────────────────────────────

    def on_tool_call_start(
        self,
        *,
        invocation_id: Optional[str] = None,
        tool_name: str = "unknown",
        tool_args: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Optional[Dict[str, Any]]:
        """Called before a tool is executed.

        If ``check_tool_inputs`` is enabled, the stringified arguments are
        checked for prompt injection or PII leakage.

        Returns
        -------
        dict or None
            If a violation is detected and blocking is on, raises.
            Otherwise returns *None* to allow normal execution.
        """
        inv_id = str(invocation_id or "")
        inv_data = self._invocations.get(inv_id, {})
        args_str = str(tool_args) if tool_args else ""

        tool_record: Dict[str, Any] = {
            "tool_name": tool_name,
            "args": tool_args or {},
            "start": time.monotonic(),
        }

        if self._check_tool_inputs and args_str:
            check = self._guard.check_input(
                args_str,
                rules=self._rules,
                metadata={
                    "tool_name": tool_name,
                    "framework": "google_adk",
                },
            )
            tool_record["input_violations"] = check.get("violations", [])
            if not check.get("allowed", True) and self._block:
                raise GuardrailViolation(check.get("violations", []))

        inv_data.setdefault("tool_calls", []).append(tool_record)
        return None

    def on_tool_call_end(
        self,
        *,
        invocation_id: Optional[str] = None,
        tool_name: str = "unknown",
        tool_result: Any = None,
        **kwargs: Any,
    ) -> None:
        """Called after a tool execution completes."""
        inv_id = str(invocation_id or "")
        inv_data = self._invocations.get(inv_id, {})
        tool_calls = inv_data.get("tool_calls", [])
        # Update the most recent matching tool call
        for tc in reversed(tool_calls):
            if tc["tool_name"] == tool_name and "result" not in tc:
                tc["result"] = str(tool_result)[:2000] if tool_result else ""
                tc["latency_ms"] = round(
                    (time.monotonic() - tc.get("start", time.monotonic())) * 1000, 2
                )
                break

    # ── Model / thinking step callbacks ──────────────────────────────

    def on_model_response(
        self,
        *,
        invocation_id: Optional[str] = None,
        model_name: str = "unknown",
        response_text: str = "",
        thinking: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Called when the LLM generates a response within the agent loop.

        Captures intermediate model responses and thinking/reasoning steps
        as part of the agent trajectory.
        """
        inv_id = str(invocation_id or "")
        inv_data = self._invocations.get(inv_id, {})
        step: Dict[str, Any] = {
            "type": "model_response",
            "model": model_name,
            "response": response_text[:2000],
            "timestamp": time.time(),
        }
        if thinking:
            step["thinking"] = thinking[:2000]

        inv_data.setdefault("steps", []).append(step)

        # Optional output check on intermediate responses
        if self._check_model_outputs and response_text:
            check = self._guard.check_output(
                response_text,
                metadata={"model": model_name, "framework": "google_adk"},
            )
            if not check.get("allowed", True) and self._block:
                raise GuardrailViolation(check.get("violations", []))

    def on_thinking_step(
        self,
        *,
        invocation_id: Optional[str] = None,
        thought: str = "",
        **kwargs: Any,
    ) -> None:
        """Called for explicit thinking/reasoning steps (e.g. Gemini thinking)."""
        inv_id = str(invocation_id or "")
        inv_data = self._invocations.get(inv_id, {})
        inv_data.setdefault("steps", []).append({
            "type": "thinking",
            "thought": thought[:2000],
            "timestamp": time.time(),
        })


def evalguard_before_tool(
    api_key: str,
    project_id: Optional[str] = None,
    base_url: str = "https://evalguard.ai/api",
    rules: Optional[List[str]] = None,
    block_on_violation: bool = True,
) -> Callable[..., Any]:
    """Return a ``before_tool_callback`` function compatible with ADK agents.

    Usage::

        from evalguard.google_adk import evalguard_before_tool

        agent = Agent(
            name="my_agent",
            model="gemini-2.0-flash",
            before_tool_callback=evalguard_before_tool(api_key="eg_..."),
        )

    The returned callback checks tool arguments against EvalGuard guardrails
    before the tool is executed.
    """
    guard = GuardrailClient(
        api_key=api_key, base_url=base_url, project_id=project_id,
    )

    def _before_tool(
        tool: Any,
        args: Dict[str, Any],
        tool_context: Any = None,
        **kwargs: Any,
    ) -> Optional[Dict[str, Any]]:
        tool_name = getattr(tool, "name", getattr(tool, "__name__", str(tool)))
        args_str = str(args) if args else ""
        if not args_str:
            return None

        check = guard.check_input(
            args_str,
            rules=rules,
            metadata={
                "tool_name": tool_name,
                "framework": "google_adk",
            },
        )
        if not check.get("allowed", True) and block_on_violation:
            raise GuardrailViolation(check.get("violations", []))
        return None

    return _before_tool


def evalguard_after_tool(
    api_key: str,
    project_id: Optional[str] = None,
    base_url: str = "https://evalguard.ai/api",
    rules: Optional[List[str]] = None,
    block_on_violation: bool = True,
) -> Callable[..., Any]:
    """Return an ``after_tool_callback`` function compatible with ADK agents.

    Usage::

        from evalguard.google_adk import evalguard_after_tool

        agent = Agent(
            name="my_agent",
            model="gemini-2.0-flash",
            after_tool_callback=evalguard_after_tool(api_key="eg_..."),
        )

    The returned callback checks tool outputs against EvalGuard guardrails
    after execution.
    """
    guard = GuardrailClient(
        api_key=api_key, base_url=base_url, project_id=project_id,
    )

    def _after_tool(
        tool: Any,
        args: Dict[str, Any],
        tool_context: Any = None,
        tool_response: Any = None,
        **kwargs: Any,
    ) -> Optional[Any]:
        tool_name = getattr(tool, "name", getattr(tool, "__name__", str(tool)))
        result_str = str(tool_response)[:4000] if tool_response else ""
        if not result_str:
            return None

        check = guard.check_output(
            result_str,
            metadata={
                "tool_name": tool_name,
                "framework": "google_adk",
            },
        )
        if not check.get("allowed", True) and block_on_violation:
            raise GuardrailViolation(check.get("violations", []))

        guard.log_trace({
            "provider": "google_adk",
            "tool_name": tool_name,
            "tool_args": str(args)[:2000] if args else "",
            "tool_result": result_str[:2000],
        })
        return None

    return _after_tool


def guard_adk_agent(
    agent: Any,
    *,
    api_key: str,
    project_id: Optional[str] = None,
    base_url: str = "https://evalguard.ai/api",
    rules: Optional[List[str]] = None,
    block_on_violation: bool = True,
    timeout: float = 5.0,
) -> Any:
    """Wrap a Google ADK ``Agent`` with EvalGuard guardrails.

    Patches the agent's ``_run_async_impl`` / ``_run_live_impl`` methods
    (the internal execution entrypoints in ADK) to automatically apply
    input/output guardrail checks and trajectory tracing.

    Parameters
    ----------
    agent:
        A ``google.adk.agents.Agent`` instance.
    api_key:
        EvalGuard API key.
    project_id:
        Optional project ID for trace grouping.
    rules:
        Guardrail rules.
    block_on_violation:
        If *True*, raises on guardrail violations.

    Returns
    -------
    The same agent, with guardrails applied.
    """
    guard = GuardrailClient(
        api_key=api_key,
        base_url=base_url,
        project_id=project_id,
        timeout=timeout,
    )
    agent_name = getattr(agent, "name", "unknown")

    # Patch the generate_content / run method
    original_run = getattr(agent, "run_async", None) or getattr(agent, "run", None)
    if original_run is None:
        logger.warning("Could not find run method on agent %s; skipping guardrail patching", agent_name)
        return agent

    @functools.wraps(original_run)
    async def guarded_run(*args: Any, **kwargs: Any) -> Any:
        # Extract user message from args/kwargs
        user_message = ""
        for arg in args:
            if isinstance(arg, str):
                user_message = arg
                break
        if not user_message:
            user_message = str(kwargs.get("user_message", kwargs.get("message", "")))

        # Pre-check
        check = guard.check_input(
            user_message,
            rules=rules,
            metadata={"agent_name": agent_name, "framework": "google_adk"},
        )
        if not check.get("allowed", True) and block_on_violation:
            raise GuardrailViolation(check.get("violations", []))

        start = time.monotonic()
        result = await original_run(*args, **kwargs)
        elapsed_ms = (time.monotonic() - start) * 1000

        # Extract output text
        output_text = ""
        if isinstance(result, str):
            output_text = result
        elif hasattr(result, "text"):
            output_text = result.text
        elif hasattr(result, "content"):
            output_text = str(result.content)
        else:
            output_text = str(result)[:4000]

        # Post-check
        output_violations: List[Dict[str, Any]] = []
        if output_text:
            out_check = guard.check_output(
                output_text[:4000],
                metadata={"agent_name": agent_name, "framework": "google_adk"},
            )
            output_violations = out_check.get("violations", [])
            if not out_check.get("allowed", True) and block_on_violation:
                raise GuardrailViolation(output_violations)

        guard.log_trace({
            "provider": "google_adk",
            "agent_name": agent_name,
            "input": user_message[:2000],
            "output": output_text[:2000],
            "agent_latency_ms": round(elapsed_ms, 2),
            "input_violations": check.get("violations", []),
            "output_violations": output_violations,
        })
        return result

    # Patch the appropriate method
    if hasattr(agent, "run_async"):
        agent.run_async = guarded_run
    else:
        agent.run = guarded_run

    return agent
