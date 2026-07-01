"""Camel-AI multi-agent monitoring integration for EvalGuard.

Camel-AI is a multi-agent communication framework. This integration captures
agent-to-agent messages, role assignments, task delegation, and final outputs
as structured traces with parent-child span relationships.

Usage::

    from camel.agents import ChatAgent
    from camel.societies import RolePlaying
    from evalguard.camel_ai import EvalGuardCamelMonitor, guard_society

    # Option 1: Monitor a role-playing society
    monitor = EvalGuardCamelMonitor(api_key="eg_...", project_id="proj_...")
    society = RolePlaying(
        assistant_role_name="Python Programmer",
        user_role_name="Stock Trader",
        task_prompt="Develop a trading bot",
    )
    society = monitor.wrap_society(society)
    # All inter-agent messages are now traced

    # Option 2: Monitor individual agents
    agent = ChatAgent(system_message="You are a helpful assistant.")
    agent = monitor.wrap_agent(agent, role="assistant")
    response = agent.step("Write me a poem")

    # Option 3: Convenience function
    society = guard_society(
        RolePlaying(...),
        api_key="eg_...",
        check_messages=True,
    )

    # Option 4: Manual message logging
    monitor = EvalGuardCamelMonitor(api_key="eg_...")
    session_id = monitor.start_session(task="Build a trading bot")
    monitor.on_message(
        session_id=session_id,
        from_role="user_agent",
        to_role="assistant_agent",
        content="Please implement the strategy module",
    )
    monitor.end_session(session_id, output="Trading bot completed")
"""

from __future__ import annotations

import functools
import logging
import time
import uuid
from typing import Any, Callable, Dict, List, Optional

from .guardrails import GuardrailClient, GuardrailViolation

logger = logging.getLogger("evalguard.camel_ai")


class EvalGuardCamelMonitor:
    """Monitoring callback for Camel-AI multi-agent conversations.

    Captures agent-to-agent messages, role assignments, task delegation,
    and final outputs as structured traces with parent-child span
    relationships.

    Parameters
    ----------
    api_key:
        EvalGuard API key.
    project_id:
        Optional project ID for trace grouping.
    base_url:
        API base URL. Override for self-hosted deployments.
    rules:
        Guardrail rules for message checking.
    block_on_violation:
        If *True*, block messages that violate guardrails.
    check_messages:
        If *True*, run guardrail checks on every inter-agent message.
        If *False*, only trace without checking.
    timeout:
        HTTP request timeout in seconds.
    """

    def __init__(
        self,
        api_key: str,
        project_id: Optional[str] = None,
        base_url: str = "https://evalguard.ai/api",
        rules: Optional[List[str]] = None,
        block_on_violation: bool = True,
        check_messages: bool = True,
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
        self._check_messages = check_messages
        self._sessions: Dict[str, Dict[str, Any]] = {}
        self._traces: List[Dict[str, Any]] = []

    # ── Session management ───────────────────────────────────────────

    def start_session(
        self,
        task: str,
        *,
        roles: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Start a new multi-agent session.

        Returns a session/trace ID that should be passed to subsequent
        ``on_message`` and ``end_session`` calls.
        """
        session_id = uuid.uuid4().hex
        self._sessions[session_id] = {
            "task": task,
            "roles": roles or [],
            "start_time": time.monotonic(),
            "message_count": 0,
            "messages": [],
            "metadata": metadata or {},
        }

        entry: Dict[str, Any] = {
            "provider": "camel_ai",
            "span_type": "session_start",
            "trace_id": session_id,
            "span_id": uuid.uuid4().hex[:16],
            "task": task[:2000],
            "roles": roles or [],
            "status": "ok",
        }
        if metadata:
            entry["metadata"] = metadata
        self._traces.append(entry)
        self._guard.log_trace(entry)
        return session_id

    def end_session(
        self,
        session_id: str,
        *,
        output: Any = None,
        error: Optional[str] = None,
    ) -> Dict[str, Any]:
        """End a multi-agent session and log the final summary trace.

        Returns
        -------
        dict
            Session summary with message_count, latency_ms, and status.
        """
        session = self._sessions.pop(session_id, {})
        elapsed_ms = (time.monotonic() - session.get("start_time", time.monotonic())) * 1000

        summary: Dict[str, Any] = {
            "session_id": session_id,
            "task": session.get("task", ""),
            "message_count": session.get("message_count", 0),
            "latency_ms": round(elapsed_ms, 2),
            "status": "error" if error else "ok",
        }

        entry: Dict[str, Any] = {
            "provider": "camel_ai",
            "span_type": "session_end",
            "trace_id": session_id,
            "span_id": uuid.uuid4().hex[:16],
            "task": session.get("task", "")[:2000],
            "output": str(output)[:2000] if output else "",
            "message_count": session.get("message_count", 0),
            "latency_ms": round(elapsed_ms, 2),
            "roles": session.get("roles", []),
            "status": "error" if error else "ok",
        }
        if error:
            entry["error"] = error
        self._traces.append(entry)
        self._guard.log_trace(entry)
        return summary

    # ── Message-level callbacks ──────────────────────────────────────

    def on_message(
        self,
        session_id: str,
        from_role: str,
        to_role: str,
        content: str,
        *,
        message_type: str = "chat",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Log an inter-agent message.

        Parameters
        ----------
        session_id:
            The session trace ID from :meth:`start_session`.
        from_role:
            Sender agent's role name.
        to_role:
            Receiver agent's role name.
        content:
            Message content.
        message_type:
            One of ``"chat"``, ``"task_delegation"``, ``"feedback"``,
            ``"system"``.

        Returns
        -------
        str
            The span ID for this message.

        Raises
        ------
        GuardrailViolation
            If ``check_messages`` and ``block_on_violation`` are both True
            and the message content violates guardrails.
        """
        span_id = uuid.uuid4().hex[:16]

        # Guardrail check on message content
        violations: List[Dict[str, Any]] = []
        if self._check_messages and content:
            check = self._guard.check_input(
                content,
                rules=self._rules,
                metadata={
                    "framework": "camel_ai",
                    "from_role": from_role,
                    "to_role": to_role,
                    "message_type": message_type,
                },
            )
            violations = check.get("violations", [])
            if not check.get("allowed", True) and self._block:
                self._guard.log_trace({
                    "provider": "camel_ai",
                    "span_type": "message_blocked",
                    "trace_id": session_id,
                    "span_id": span_id,
                    "from_role": from_role,
                    "to_role": to_role,
                    "status": "blocked",
                    "violations": violations,
                })
                raise GuardrailViolation(
                    violations,
                    message=f"Inter-agent message from {from_role} to {to_role} blocked by guardrail",
                )

        # Update session
        session = self._sessions.get(session_id)
        if session:
            session["message_count"] += 1

        # Log trace
        entry: Dict[str, Any] = {
            "provider": "camel_ai",
            "span_type": "agent_message",
            "trace_id": session_id,
            "span_id": span_id,
            "from_role": from_role,
            "to_role": to_role,
            "message_type": message_type,
            "content": content[:2000],
            "status": "ok",
            "violations": violations,
        }
        if metadata:
            entry["metadata"] = metadata
        self._traces.append(entry)
        self._guard.log_trace(entry)
        return span_id

    def on_role_assignment(
        self,
        session_id: str,
        role_name: str,
        system_message: str,
        *,
        agent_type: str = "chat",
    ) -> str:
        """Log an agent role assignment within a session.

        Returns the span ID.
        """
        span_id = uuid.uuid4().hex[:16]
        session = self._sessions.get(session_id)
        if session and role_name not in session.get("roles", []):
            session.setdefault("roles", []).append(role_name)

        entry: Dict[str, Any] = {
            "provider": "camel_ai",
            "span_type": "role_assignment",
            "trace_id": session_id,
            "span_id": span_id,
            "role_name": role_name,
            "agent_type": agent_type,
            "system_message": system_message[:2000],
            "status": "ok",
        }
        self._traces.append(entry)
        self._guard.log_trace(entry)
        return span_id

    def on_task_delegation(
        self,
        session_id: str,
        from_role: str,
        to_role: str,
        task_description: str,
        *,
        parent_span_id: Optional[str] = None,
    ) -> str:
        """Log a task delegation between agents.

        Returns the span ID.
        """
        return self.on_message(
            session_id=session_id,
            from_role=from_role,
            to_role=to_role,
            content=task_description,
            message_type="task_delegation",
            metadata={"parent_span_id": parent_span_id} if parent_span_id else None,
        )

    # ── Agent wrapping ───────────────────────────────────────────────

    def wrap_agent(
        self,
        agent: Any,
        *,
        role: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> Any:
        """Wrap a Camel-AI ChatAgent to trace its step() calls.

        Patches the agent's ``step`` method to capture each input/output
        exchange as a traced message span.

        Parameters
        ----------
        agent:
            A ``camel.agents.ChatAgent`` instance.
        role:
            Role name to identify this agent in traces. Defaults to the
            agent's role name or class name.
        session_id:
            Optional session ID for grouping traces. If not provided,
            a new trace ID is generated per call.

        Returns
        -------
        The same agent instance with tracing applied.
        """
        monitor = self
        agent_role = role or _get_agent_role(agent)

        original_step = getattr(agent, "step", None)
        if original_step is None:
            logger.warning("Agent has no 'step' method; skipping wrap")
            return agent

        @functools.wraps(original_step)
        def traced_step(input_message: Any = None, *args: Any, **kwargs: Any) -> Any:
            sid = session_id or uuid.uuid4().hex
            content = _extract_message_content(input_message)

            # Guardrail check
            if monitor._check_messages and content:
                check = monitor._guard.check_input(
                    content,
                    rules=monitor._rules,
                    metadata={
                        "framework": "camel_ai",
                        "agent_role": agent_role,
                    },
                )
                if not check.get("allowed", True) and monitor._block:
                    raise GuardrailViolation(check.get("violations", []))

            start = time.monotonic()
            error_msg: Optional[str] = None
            result = None
            try:
                result = original_step(input_message, *args, **kwargs)
                return result
            except GuardrailViolation:
                raise
            except Exception as exc:
                error_msg = f"{type(exc).__name__}: {exc}"
                raise
            finally:
                elapsed_ms = (time.monotonic() - start) * 1000
                output_content = _extract_response_content(result)

                entry: Dict[str, Any] = {
                    "provider": "camel_ai",
                    "span_type": "agent_step",
                    "trace_id": sid,
                    "span_id": uuid.uuid4().hex[:16],
                    "agent_role": agent_role,
                    "input": content[:2000],
                    "output": output_content[:2000],
                    "latency_ms": round(elapsed_ms, 2),
                    "status": "error" if error_msg else "ok",
                }
                if error_msg:
                    entry["error"] = error_msg
                monitor._traces.append(entry)
                monitor._guard.log_trace(entry)

        agent.step = traced_step
        return agent

    def wrap_society(self, society: Any) -> Any:
        """Wrap a Camel-AI RolePlaying society to trace the full conversation.

        Patches ``init_chat`` and ``step`` to capture:
        - Role assignments and system messages
        - Each conversation turn between assistant and user agents
        - Task description and final output

        Parameters
        ----------
        society:
            A ``camel.societies.RolePlaying`` instance.

        Returns
        -------
        The same society instance with tracing applied.
        """
        monitor = self

        # Start a persistent session for this society
        task = getattr(society, "task_prompt", "") or getattr(society, "task", "") or ""
        assistant_role = getattr(society, "assistant_role_name", "assistant")
        user_role = getattr(society, "user_role_name", "user")
        session_id = monitor.start_session(
            task=str(task),
            roles=[assistant_role, user_role],
            metadata={"society_type": type(society).__name__},
        )
        society._evalguard_session_id = session_id

        # Patch init_chat
        original_init_chat = getattr(society, "init_chat", None)
        if original_init_chat is not None:

            @functools.wraps(original_init_chat)
            def traced_init_chat(*args: Any, **kwargs: Any) -> Any:
                result = original_init_chat(*args, **kwargs)

                # Log role assignments
                for attr, role_name in [
                    ("assistant_agent", assistant_role),
                    ("user_agent", user_role),
                ]:
                    ag = getattr(society, attr, None)
                    if ag:
                        sys_msg = _get_system_message(ag)
                        monitor.on_role_assignment(
                            session_id=session_id,
                            role_name=role_name,
                            system_message=sys_msg,
                            agent_type=type(ag).__name__,
                        )

                return result

            society.init_chat = traced_init_chat

        # Patch step
        original_step = getattr(society, "step", None)
        if original_step is not None:

            @functools.wraps(original_step)
            def traced_society_step(*args: Any, **kwargs: Any) -> Any:
                start = time.monotonic()
                error_msg: Optional[str] = None
                result = None
                try:
                    result = original_step(*args, **kwargs)
                    return result
                except Exception as exc:
                    error_msg = f"{type(exc).__name__}: {exc}"
                    raise
                finally:
                    elapsed_ms = (time.monotonic() - start) * 1000

                    # Extract messages from result
                    # RolePlaying.step() returns (assistant_msg, user_msg) or
                    # ChatAgentResponse objects
                    if result is not None:
                        if isinstance(result, tuple) and len(result) >= 2:
                            asst_content = _extract_response_content(result[0])
                            user_content = _extract_response_content(result[1])
                        else:
                            asst_content = _extract_response_content(result)
                            user_content = ""

                        if asst_content:
                            monitor.on_message(
                                session_id=session_id,
                                from_role=assistant_role,
                                to_role=user_role,
                                content=asst_content,
                            )
                        if user_content:
                            monitor.on_message(
                                session_id=session_id,
                                from_role=user_role,
                                to_role=assistant_role,
                                content=user_content,
                            )
                    elif error_msg:
                        monitor._guard.log_trace({
                            "provider": "camel_ai",
                            "span_type": "society_step_error",
                            "trace_id": session_id,
                            "span_id": uuid.uuid4().hex[:16],
                            "error": error_msg,
                            "latency_ms": round(elapsed_ms, 2),
                            "status": "error",
                        })

            society.step = traced_society_step

        return society

    # ── Access and cleanup ───────────────────────────────────────────

    def get_traces(self) -> List[Dict[str, Any]]:
        """Return a copy of all collected trace entries."""
        return list(self._traces)

    def flush(self) -> None:
        """Clear the local trace buffer."""
        self._traces.clear()

    def get_active_sessions(self) -> Dict[str, Dict[str, Any]]:
        """Return info about active (not yet ended) sessions."""
        return {
            sid: {
                "task": s.get("task", ""),
                "roles": s.get("roles", []),
                "message_count": s.get("message_count", 0),
                "elapsed_ms": round(
                    (time.monotonic() - s.get("start_time", time.monotonic())) * 1000, 2
                ),
            }
            for sid, s in self._sessions.items()
        }


def guard_society(
    society: Any,
    *,
    api_key: str,
    project_id: Optional[str] = None,
    base_url: str = "https://evalguard.ai/api",
    rules: Optional[List[str]] = None,
    block_on_violation: bool = True,
    check_messages: bool = True,
    timeout: float = 5.0,
) -> Any:
    """Convenience function to wrap a Camel-AI society with full monitoring.

    Parameters
    ----------
    society:
        A ``camel.societies.RolePlaying`` or similar society instance.
    api_key:
        EvalGuard API key.
    check_messages:
        If *True*, check every inter-agent message against guardrails.

    Returns
    -------
    The same society instance with monitoring and guardrails applied.
    """
    monitor = EvalGuardCamelMonitor(
        api_key=api_key,
        project_id=project_id,
        base_url=base_url,
        rules=rules,
        block_on_violation=block_on_violation,
        check_messages=check_messages,
        timeout=timeout,
    )
    return monitor.wrap_society(society)


# ── Internal helpers ─────────────────────────────────────────────────


def _get_agent_role(agent: Any) -> str:
    """Extract a role name from a Camel-AI agent."""
    # ChatAgent stores role info in system_message or role_name
    role = getattr(agent, "role_name", None)
    if role:
        return str(role)
    sys_msg = getattr(agent, "system_message", None)
    if sys_msg:
        role_name = getattr(sys_msg, "role_name", None)
        if role_name:
            return str(role_name)
    return type(agent).__name__


def _get_system_message(agent: Any) -> str:
    """Extract system message text from a Camel-AI agent."""
    sys_msg = getattr(agent, "system_message", None)
    if sys_msg is None:
        return ""
    # BaseMessage has .content
    content = getattr(sys_msg, "content", None)
    if isinstance(content, str):
        return content[:2000]
    return str(sys_msg)[:2000]


def _extract_message_content(message: Any) -> str:
    """Extract text content from a Camel-AI message object or string."""
    if message is None:
        return ""
    if isinstance(message, str):
        return message
    # BaseMessage / ChatMessage has .content
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content
    # May be a dict
    if isinstance(message, dict):
        return str(message.get("content", message.get("text", "")))
    return str(message)


def _extract_response_content(response: Any) -> str:
    """Extract text from a Camel-AI response (ChatAgentResponse or message)."""
    if response is None:
        return ""
    if isinstance(response, str):
        return response

    # ChatAgentResponse has .msg or .msgs
    msg = getattr(response, "msg", None)
    if msg is not None:
        return _extract_message_content(msg)

    msgs = getattr(response, "msgs", None)
    if msgs and isinstance(msgs, list):
        parts = [_extract_message_content(m) for m in msgs]
        return "\n".join(p for p in parts if p)

    # Fallback: try .content directly
    content = getattr(response, "content", None)
    if isinstance(content, str):
        return content

    return str(response)[:2000]
