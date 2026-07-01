"""Smolagents monitoring integration for EvalGuard.

HuggingFace's lightweight agent framework integration that captures tool calls,
code execution steps, model reasoning, and final outputs as structured traces.

Usage::

    from smolagents import CodeAgent, HfApiModel
    from evalguard.smolagents import EvalGuardMonitor, EvalGuardTool

    # Option 1: Monitor an agent's execution loop
    monitor = EvalGuardMonitor(api_key="eg_...", project_id="proj_...")
    agent = CodeAgent(tools=[], model=HfApiModel())

    # Wrap the agent to capture all steps
    agent = monitor.wrap_agent(agent)
    result = agent.run("Summarize the latest news")

    # Option 2: Give agents a self-evaluation tool
    eval_tool = EvalGuardTool(api_key="eg_...")
    agent = CodeAgent(tools=[eval_tool], model=HfApiModel())
    result = agent.run("Generate a report and evaluate its quality")

    # Option 3: Manual step logging
    monitor = EvalGuardMonitor(api_key="eg_...")
    monitor.on_tool_call("web_search", {"query": "latest news"}, "results...", 150.0)
    monitor.on_step_complete("reasoning", "The user wants...", step_number=1)
    monitor.flush()
"""

from __future__ import annotations

import functools
import logging
import time
import uuid
from typing import Any, Callable, Dict, List, Optional

from .guardrails import GuardrailClient, GuardrailViolation

logger = logging.getLogger("evalguard.smolagents")


class EvalGuardMonitor:
    """Monitoring callback that hooks into smolagents' agent loop.

    Captures tool calls, code execution steps, model reasoning, and final
    outputs, sending structured traces to the EvalGuard API with
    parent-child span relationships.

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
        If *True*, raises on guardrail violation before agent runs.
    timeout:
        HTTP request timeout in seconds.
    capture_code:
        If *True*, captures generated code from CodeAgent steps.
    """

    def __init__(
        self,
        api_key: str,
        project_id: Optional[str] = None,
        base_url: str = "https://evalguard.ai/api",
        rules: Optional[List[str]] = None,
        block_on_violation: bool = True,
        timeout: float = 5.0,
        capture_code: bool = True,
    ) -> None:
        self._guard = GuardrailClient(
            api_key=api_key,
            base_url=base_url,
            project_id=project_id,
            timeout=timeout,
        )
        self._rules = rules
        self._block = block_on_violation
        self._capture_code = capture_code
        self._traces: List[Dict[str, Any]] = []

    # ── Manual logging API ───────────────────────────────────────────

    def on_tool_call(
        self,
        tool_name: str,
        tool_input: Any,
        tool_output: Any = None,
        latency_ms: float = 0.0,
        *,
        error: Optional[str] = None,
        trace_id: Optional[str] = None,
        parent_span_id: Optional[str] = None,
    ) -> str:
        """Log a tool call executed by the agent.

        Returns the span ID for parent-child linking.
        """
        span_id = uuid.uuid4().hex[:16]
        entry: Dict[str, Any] = {
            "provider": "smolagents",
            "span_type": "tool_call",
            "span_id": span_id,
            "tool_name": tool_name,
            "input": str(tool_input)[:2000] if tool_input else "",
            "output": str(tool_output)[:2000] if tool_output else "",
            "latency_ms": round(latency_ms, 2),
        }
        if error:
            entry["error"] = error
            entry["status"] = "error"
        else:
            entry["status"] = "ok"
        if trace_id:
            entry["trace_id"] = trace_id
        if parent_span_id:
            entry["parent_span_id"] = parent_span_id
        self._traces.append(entry)
        self._guard.log_trace(entry)
        return span_id

    def on_step_complete(
        self,
        step_type: str,
        content: str,
        *,
        step_number: int = 0,
        latency_ms: float = 0.0,
        code: Optional[str] = None,
        trace_id: Optional[str] = None,
        parent_span_id: Optional[str] = None,
    ) -> str:
        """Log a completed agent step (reasoning, code execution, etc.).

        Parameters
        ----------
        step_type:
            One of ``"reasoning"``, ``"code_execution"``, ``"action"``,
            ``"observation"``, ``"planning"``.
        content:
            The textual content of the step.
        step_number:
            Sequential step index within the agent run.
        code:
            Generated code, if this is a code execution step.

        Returns the span ID.
        """
        span_id = uuid.uuid4().hex[:16]
        entry: Dict[str, Any] = {
            "provider": "smolagents",
            "span_type": "agent_step",
            "span_id": span_id,
            "step_type": step_type,
            "step_number": step_number,
            "content": content[:2000],
            "latency_ms": round(latency_ms, 2),
            "status": "ok",
        }
        if code and self._capture_code:
            entry["code"] = code[:4000]
        if trace_id:
            entry["trace_id"] = trace_id
        if parent_span_id:
            entry["parent_span_id"] = parent_span_id
        self._traces.append(entry)
        self._guard.log_trace(entry)
        return span_id

    def on_agent_run(
        self,
        task: str,
        output: Any = None,
        *,
        latency_ms: float = 0.0,
        total_steps: int = 0,
        error: Optional[str] = None,
    ) -> str:
        """Log a complete agent run as a top-level trace span.

        Returns the trace ID.
        """
        trace_id = uuid.uuid4().hex
        entry: Dict[str, Any] = {
            "provider": "smolagents",
            "span_type": "agent_run",
            "trace_id": trace_id,
            "input": task[:2000],
            "output": str(output)[:2000] if output else "",
            "latency_ms": round(latency_ms, 2),
            "total_steps": total_steps,
            "status": "error" if error else "ok",
        }
        if error:
            entry["error"] = error
        self._traces.append(entry)
        self._guard.log_trace(entry)
        return trace_id

    def flush(self) -> None:
        """Flush any pending trace entries."""
        # Individual entries are sent immediately via log_trace, but
        # clear the local buffer.
        self._traces.clear()

    def get_traces(self) -> List[Dict[str, Any]]:
        """Return a copy of all collected trace entries."""
        return list(self._traces)

    # ── Agent wrapping ───────────────────────────────────────────────

    def wrap_agent(self, agent: Any) -> Any:
        """Wrap a smolagents Agent to automatically trace all executions.

        Patches the agent's ``run`` method to capture:
        - Input task guardrail checks
        - Each step in the agent loop (via ``step`` method)
        - Tool calls (via tool ``__call__`` or ``forward``)
        - Final output and total latency

        Parameters
        ----------
        agent:
            A ``smolagents.CodeAgent``, ``smolagents.ToolCallingAgent``,
            or any agent with a ``run`` method.

        Returns
        -------
        The same agent instance with monitoring applied.
        """
        monitor = self

        # Patch tools to capture calls
        tools = getattr(agent, "tools", None) or getattr(agent, "toolbox", None)
        if tools:
            tool_dict = tools if isinstance(tools, dict) else {}
            if isinstance(tools, list):
                for t in tools:
                    name = getattr(t, "name", None) or type(t).__name__
                    tool_dict[name] = t
            elif hasattr(tools, "tools"):
                # smolagents Toolbox object
                tool_dict = tools.tools if isinstance(tools.tools, dict) else {}

            for tool_name, tool in tool_dict.items():
                _patch_tool(tool, tool_name, monitor)

        # Patch the run method
        original_run = getattr(agent, "run", None)
        if original_run is None:
            logger.warning("Agent has no 'run' method; skipping wrap")
            return agent

        @functools.wraps(original_run)
        def traced_run(task: str, *args: Any, **kwargs: Any) -> Any:
            # Pre-check guardrails on the task input
            check = monitor._guard.check_input(
                task,
                rules=monitor._rules,
                metadata={"framework": "smolagents", "agent_type": type(agent).__name__},
            )
            if not check.get("allowed", True) and monitor._block:
                raise GuardrailViolation(check.get("violations", []))

            trace_id = uuid.uuid4().hex
            start = time.monotonic()
            error_msg: Optional[str] = None
            result = None
            try:
                result = original_run(task, *args, **kwargs)
                return result
            except Exception as exc:
                error_msg = f"{type(exc).__name__}: {exc}"
                raise
            finally:
                elapsed_ms = (time.monotonic() - start) * 1000
                # Count steps from agent's internal log if available
                steps = 0
                agent_logs = getattr(agent, "logs", None) or getattr(agent, "write_inner_memory_from_logs", None)
                if isinstance(agent_logs, list):
                    steps = len(agent_logs)

                monitor.on_agent_run(
                    task=task,
                    output=result,
                    latency_ms=elapsed_ms,
                    total_steps=steps,
                    error=error_msg,
                )

        agent.run = traced_run
        return agent


def _patch_tool(tool: Any, tool_name: str, monitor: EvalGuardMonitor) -> None:
    """Patch a smolagents tool's forward/call method to capture invocations."""
    # smolagents tools implement either forward() or __call__
    original_fn = getattr(tool, "forward", None) or getattr(tool, "__call__", None)
    if original_fn is None:
        return

    method_name = "forward" if hasattr(tool, "forward") else "__call__"

    @functools.wraps(original_fn)
    def traced_tool_call(*args: Any, **kwargs: Any) -> Any:
        tool_input = {"args": args, "kwargs": kwargs} if args or kwargs else {}
        start = time.monotonic()
        error_msg: Optional[str] = None
        result = None
        try:
            result = original_fn(*args, **kwargs)
            return result
        except Exception as exc:
            error_msg = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            elapsed_ms = (time.monotonic() - start) * 1000
            monitor.on_tool_call(
                tool_name=tool_name,
                tool_input=tool_input,
                tool_output=result,
                latency_ms=elapsed_ms,
                error=error_msg,
            )

    setattr(tool, method_name, traced_tool_call)


class EvalGuardTool:
    """A smolagents-compatible tool that lets agents self-evaluate outputs.

    When added to a smolagents agent's toolbox, the agent can call this tool
    to check its own outputs against EvalGuard guardrails before returning
    them to the user.

    Usage::

        from evalguard.smolagents import EvalGuardTool
        from smolagents import CodeAgent, HfApiModel

        eval_tool = EvalGuardTool(api_key="eg_...")
        agent = CodeAgent(tools=[eval_tool], model=HfApiModel())
        result = agent.run("Write a summary and evaluate it for quality")

    The agent can call ``evalguard_check(text="...")`` in its generated code.
    """

    # smolagents tool interface attributes
    name: str = "evalguard_check"
    description: str = (
        "Check text against safety and quality guardrails. "
        "Returns a dict with 'allowed' (bool), 'violations' (list), "
        "and 'sanitized' (cleaned text or None). Use this to validate "
        "outputs before returning them to the user."
    )
    inputs: Dict[str, Any] = {
        "text": {
            "type": "string",
            "description": "The text to check against guardrails.",
        },
        "check_type": {
            "type": "string",
            "description": "Type of check: 'input' or 'output'. Defaults to 'output'.",
            "nullable": True,
        },
    }
    output_type: str = "string"

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

    def forward(self, text: str, check_type: Optional[str] = None) -> str:
        """Execute the guardrail check.

        Parameters
        ----------
        text:
            Text to evaluate.
        check_type:
            ``"input"`` for pre-LLM checks, ``"output"`` for post-LLM.
            Defaults to ``"output"``.

        Returns
        -------
        str
            JSON-formatted result with allowed, violations, and sanitized fields.
        """
        import json

        if check_type == "input":
            result = self._guard.check_input(
                text,
                rules=self._rules,
                metadata={"framework": "smolagents", "source": "self_eval"},
            )
        else:
            result = self._guard.check_output(
                text,
                rules=self._rules,
                metadata={"framework": "smolagents", "source": "self_eval"},
            )

        self._guard.log_trace({
            "provider": "smolagents",
            "span_type": "self_evaluation",
            "input": text[:2000],
            "check_type": check_type or "output",
            "allowed": result.get("allowed", True),
            "violations": result.get("violations", []),
        })

        return json.dumps(result, default=str)

    def __call__(self, text: str, check_type: Optional[str] = None) -> str:
        """Alias for forward() for direct invocation compatibility."""
        return self.forward(text, check_type=check_type)
