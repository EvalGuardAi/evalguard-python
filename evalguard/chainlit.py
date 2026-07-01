"""Chainlit integration for EvalGuard.

Chainlit is an LLM app UI framework for building chat interfaces. This
module provides:

1. A ``@chainlit_trace`` decorator that wraps Chainlit step handlers to
   auto-capture user messages, assistant responses, tool calls, and durations.
2. A ``ChainlitTracer`` class for manual tracing within Chainlit apps.
3. Feedback collection integration that connects Chainlit's thumbs up/down
   to the EvalGuard feedback API.

Usage -- Decorator with @cl.on_message::

    import chainlit as cl
    from evalguard.chainlit import chainlit_trace

    @cl.on_message
    @chainlit_trace(api_key="eg_...", project_id="proj_...")
    async def on_message(message: cl.Message):
        response = await my_llm_call(message.content)
        await cl.Message(content=response).send()

Usage -- Manual tracer::

    import chainlit as cl
    from evalguard.chainlit import ChainlitTracer

    tracer = ChainlitTracer(api_key="eg_...", project_id="proj_...")

    @cl.on_message
    async def on_message(message: cl.Message):
        trace = tracer.start_trace(user_message=message.content)
        try:
            response = await my_llm_call(message.content)
            tracer.end_trace(trace, assistant_message=response)
        except Exception as e:
            tracer.end_trace(trace, error=e)
            raise
        await cl.Message(content=response).send()

Usage -- Feedback integration::

    import chainlit as cl
    from evalguard.chainlit import ChainlitFeedback

    feedback = ChainlitFeedback(api_key="eg_...", project_id="proj_...")

    @cl.on_message
    async def on_message(message: cl.Message):
        response = "Hello!"
        msg = cl.Message(content=response)
        await msg.send()
        # Store message ID for feedback correlation
        feedback.track_message(msg.id, message.content, response)

    @cl.action_callback("thumbs_up")
    async def on_thumbs_up(action: cl.Action):
        feedback.record(action.value, score=1, comment="Thumbs up")

    @cl.action_callback("thumbs_down")
    async def on_thumbs_down(action: cl.Action):
        feedback.record(action.value, score=0, comment="Thumbs down")

Usage -- Step-level tracing::

    import chainlit as cl
    from evalguard.chainlit import chainlit_step_trace

    @cl.on_message
    async def on_message(message: cl.Message):
        @chainlit_step_trace(api_key="eg_...", step_type="tool")
        async def search_docs(query: str) -> str:
            return await vector_search(query)

        results = await search_docs(message.content)
        await cl.Message(content=results).send()
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import logging
import time
import uuid
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union

from .guardrails import GuardrailClient, GuardrailViolation

logger = logging.getLogger("evalguard.chainlit")

F = TypeVar("F", bound=Callable[..., Any])


class ChainlitTracer:
    """Manual tracing for Chainlit applications.

    Provides explicit start/end trace calls for full control over what
    is captured and sent to EvalGuard.

    Parameters
    ----------
    api_key:
        EvalGuard API key.
    project_id:
        Optional project ID for trace grouping.
    base_url:
        API base URL for self-hosted deployments.
    rules:
        Optional guardrail rules to check inputs against.
    block_on_violation:
        If *True*, :meth:`start_trace` raises when input is blocked.
    timeout:
        HTTP request timeout in seconds.
    """

    def __init__(
        self,
        api_key: str,
        project_id: Optional[str] = None,
        base_url: str = "https://evalguard.ai/api",
        rules: Optional[List[str]] = None,
        block_on_violation: bool = False,
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
        self._active_traces: Dict[str, Dict[str, Any]] = {}

    def start_trace(
        self,
        *,
        user_message: str = "",
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Begin a new trace for a conversation turn.

        Parameters
        ----------
        user_message:
            The user's input message.
        session_id:
            Chainlit session ID for conversation grouping.
        metadata:
            Extra context to attach to the trace.

        Returns
        -------
        dict
            Trace context with ``trace_id``, ``start_time``, and
            ``guard_result`` (if rules are configured).

        Raises
        ------
        GuardrailViolation
            If ``block_on_violation`` is *True* and the input check fails.
        """
        trace_id = uuid.uuid4().hex
        trace_ctx: Dict[str, Any] = {
            "trace_id": trace_id,
            "start_time": time.monotonic(),
            "start_timestamp": time.time(),
            "user_message": user_message,
            "session_id": session_id or "",
            "metadata": metadata or {},
            "steps": [],
            "guard_result": None,
        }

        # Pre-check input if rules are configured
        if user_message and self._rules:
            guard_result = self._guard.check_input(
                user_message,
                rules=self._rules,
                metadata={"framework": "chainlit", "session_id": session_id or ""},
            )
            trace_ctx["guard_result"] = guard_result
            if not guard_result.get("allowed", True) and self._block:
                raise GuardrailViolation(guard_result.get("violations", []))

        self._active_traces[trace_id] = trace_ctx
        return trace_ctx

    def add_step(
        self,
        trace_ctx: Dict[str, Any],
        *,
        step_type: str = "assistant",
        name: str = "",
        input_text: str = "",
        output_text: str = "",
        duration_ms: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add an intermediate step to an active trace.

        Parameters
        ----------
        trace_ctx:
            The trace context returned by :meth:`start_trace`.
        step_type:
            Type of step: ``"tool"``, ``"retrieval"``, ``"llm"``, ``"assistant"``.
        name:
            Step name or identifier.
        input_text:
            Input to this step.
        output_text:
            Output from this step.
        duration_ms:
            Step duration in milliseconds.
        metadata:
            Extra context for this step.
        """
        step = {
            "step_type": step_type,
            "name": name,
            "input": input_text[:2000] if input_text else "",
            "output": output_text[:2000] if output_text else "",
            "duration_ms": round(duration_ms, 2),
            "timestamp": time.time(),
        }
        if metadata:
            step["metadata"] = metadata
        trace_ctx.setdefault("steps", []).append(step)

    def end_trace(
        self,
        trace_ctx: Dict[str, Any],
        *,
        assistant_message: str = "",
        tool_calls: Optional[List[Dict[str, Any]]] = None,
        token_usage: Optional[Dict[str, int]] = None,
        error: Optional[BaseException] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Complete a trace and send it to EvalGuard.

        Parameters
        ----------
        trace_ctx:
            The trace context returned by :meth:`start_trace`.
        assistant_message:
            The final assistant response.
        tool_calls:
            List of tool calls made during the turn.
        token_usage:
            Token usage dict with ``prompt_tokens``, ``completion_tokens``,
            ``total_tokens``.
        error:
            If the turn failed, the exception.
        metadata:
            Extra context to merge into the trace.
        """
        trace_id = trace_ctx.get("trace_id", "")
        self._active_traces.pop(trace_id, None)

        start = trace_ctx.get("start_time", time.monotonic())
        duration_ms = (time.monotonic() - start) * 1000

        trace_data: Dict[str, Any] = {
            "provider": "chainlit",
            "trace_id": trace_id,
            "session_id": trace_ctx.get("session_id", ""),
            "input": trace_ctx.get("user_message", "")[:2000],
            "output": assistant_message[:2000] if assistant_message else "",
            "duration_ms": round(duration_ms, 2),
            "status": "error" if error else "ok",
            "steps": trace_ctx.get("steps", []),
        }

        if tool_calls:
            trace_data["tool_calls"] = [
                {
                    "name": tc.get("name", ""),
                    "input": str(tc.get("input", ""))[:500],
                    "output": str(tc.get("output", ""))[:500],
                }
                for tc in tool_calls[:20]
            ]

        if token_usage:
            trace_data["token_usage"] = token_usage

        if error:
            trace_data["error"] = f"{type(error).__name__}: {error}"

        guard_result = trace_ctx.get("guard_result")
        if guard_result:
            trace_data["guard_violations"] = guard_result.get("violations", [])

        # Merge caller metadata
        extra = {**trace_ctx.get("metadata", {})}
        if metadata:
            extra.update(metadata)
        if extra:
            trace_data["metadata"] = extra

        self._guard.log_trace(trace_data)


class ChainlitFeedback:
    """Connects Chainlit user feedback (thumbs up/down) to EvalGuard.

    Tracks message-response pairs and records user feedback scores
    against EvalGuard traces for RLHF and quality monitoring.

    Parameters
    ----------
    api_key:
        EvalGuard API key.
    project_id:
        Optional project ID.
    base_url:
        API base URL for self-hosted deployments.
    timeout:
        HTTP request timeout in seconds.
    """

    def __init__(
        self,
        api_key: str,
        project_id: Optional[str] = None,
        base_url: str = "https://evalguard.ai/api",
        timeout: float = 5.0,
    ) -> None:
        self._guard = GuardrailClient(
            api_key=api_key,
            base_url=base_url,
            project_id=project_id,
            timeout=timeout,
        )
        # message_id -> {input, output, trace_id, timestamp}
        self._tracked: Dict[str, Dict[str, Any]] = {}

    def track_message(
        self,
        message_id: str,
        user_input: str,
        assistant_output: str,
        trace_id: Optional[str] = None,
    ) -> None:
        """Associate a Chainlit message with its input/output for feedback.

        Parameters
        ----------
        message_id:
            The Chainlit ``Message.id`` value.
        user_input:
            The user's message content.
        assistant_output:
            The assistant's response content.
        trace_id:
            Optional EvalGuard trace ID to correlate with.
        """
        self._tracked[message_id] = {
            "input": user_input[:2000],
            "output": assistant_output[:2000],
            "trace_id": trace_id or uuid.uuid4().hex,
            "timestamp": time.time(),
        }
        # Keep a bounded cache (last 1000 messages)
        if len(self._tracked) > 1000:
            oldest_key = next(iter(self._tracked))
            del self._tracked[oldest_key]

    def record(
        self,
        message_id: str,
        *,
        score: float,
        comment: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record user feedback for a tracked message.

        Parameters
        ----------
        message_id:
            The Chainlit message ID that received feedback.
        score:
            Feedback score (0 = negative, 1 = positive; or any float).
        comment:
            Optional text comment from the user.
        metadata:
            Additional context.
        """
        tracked = self._tracked.get(message_id, {})

        feedback_data: Dict[str, Any] = {
            "provider": "chainlit",
            "event": "feedback",
            "message_id": message_id,
            "trace_id": tracked.get("trace_id", ""),
            "input": tracked.get("input", ""),
            "output": tracked.get("output", ""),
            "feedback_score": score,
            "feedback_comment": comment,
            "timestamp": time.time(),
        }
        if metadata:
            feedback_data["metadata"] = metadata

        self._guard.log_trace(feedback_data)


def chainlit_trace(
    *,
    api_key: str,
    project_id: Optional[str] = None,
    base_url: str = "https://evalguard.ai/api",
    rules: Optional[List[str]] = None,
    block_on_violation: bool = False,
    timeout: float = 5.0,
) -> Callable[[F], F]:
    """Decorator that wraps a Chainlit ``@cl.on_message`` handler with tracing.

    Auto-captures user messages, assistant responses (sent via ``cl.Message``),
    step durations, and errors, then sends conversation traces to EvalGuard.

    Parameters
    ----------
    api_key:
        EvalGuard API key.
    project_id:
        Optional project ID.
    rules:
        Guardrail rules for input checking.
    block_on_violation:
        If *True*, block messages that fail guardrail checks.
    timeout:
        HTTP request timeout in seconds.

    Usage::

        import chainlit as cl
        from evalguard.chainlit import chainlit_trace

        @cl.on_message
        @chainlit_trace(api_key="eg_...", project_id="proj_...")
        async def on_message(message: cl.Message):
            response = await my_llm(message.content)
            await cl.Message(content=response).send()
    """
    tracer = ChainlitTracer(
        api_key=api_key,
        project_id=project_id,
        base_url=base_url,
        rules=rules,
        block_on_violation=block_on_violation,
        timeout=timeout,
    )

    def decorator(fn: F) -> F:
        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Extract the Chainlit Message from args
            message = None
            for arg in args:
                if hasattr(arg, "content") and hasattr(arg, "id"):
                    message = arg
                    break
            for v in kwargs.values():
                if hasattr(v, "content") and hasattr(v, "id"):
                    message = v
                    break

            user_content = getattr(message, "content", "") if message else ""
            session_id = ""
            # Try to get Chainlit session context
            try:
                import chainlit as cl
                ctx = cl.context
                if hasattr(ctx, "session"):
                    session_id = getattr(ctx.session, "id", "")
            except (ImportError, AttributeError, RuntimeError):
                pass

            trace_ctx = tracer.start_trace(
                user_message=user_content,
                session_id=session_id,
            )

            # Check if input was blocked
            guard_result = trace_ctx.get("guard_result")
            if guard_result and not guard_result.get("allowed", True) and block_on_violation:
                tracer.end_trace(
                    trace_ctx,
                    assistant_message="[BLOCKED BY GUARDRAIL]",
                )
                # Try to send a Chainlit message about the block
                try:
                    import chainlit as cl
                    await cl.Message(
                        content="Your message was blocked by a content guardrail."
                    ).send()
                except (ImportError, Exception):
                    pass
                return None

            # Patch cl.Message.send to capture assistant responses
            sent_messages: list[str] = []
            _original_send = None
            try:
                import chainlit as cl
                _original_send = cl.Message.send

                async def _patched_send(self_msg: Any, *a: Any, **kw: Any) -> Any:
                    content = getattr(self_msg, "content", "")
                    if content:
                        sent_messages.append(content)
                    return await _original_send(self_msg, *a, **kw)

                cl.Message.send = _patched_send
            except (ImportError, AttributeError):
                pass

            error_caught = None
            try:
                result = await fn(*args, **kwargs)
                return result
            except Exception as exc:
                error_caught = exc
                raise
            finally:
                # Restore original send
                if _original_send is not None:
                    try:
                        import chainlit as cl
                        cl.Message.send = _original_send
                    except (ImportError, AttributeError):
                        pass

                assistant_output = "\n".join(sent_messages) if sent_messages else ""
                tracer.end_trace(
                    trace_ctx,
                    assistant_message=assistant_output,
                    error=error_caught,
                )

        return wrapper  # type: ignore[return-value]

    return decorator


def chainlit_step_trace(
    *,
    api_key: str,
    project_id: Optional[str] = None,
    base_url: str = "https://evalguard.ai/api",
    step_type: str = "tool",
    timeout: float = 5.0,
) -> Callable[[F], F]:
    """Decorator that traces individual Chainlit step functions.

    Wraps async functions used within Chainlit handlers to capture
    per-step timing, inputs, and outputs.

    Parameters
    ----------
    api_key:
        EvalGuard API key.
    project_id:
        Optional project ID.
    step_type:
        Type of step: ``"tool"``, ``"retrieval"``, ``"llm"``.
    timeout:
        HTTP request timeout in seconds.

    Usage::

        @chainlit_step_trace(api_key="eg_...", step_type="retrieval")
        async def search(query: str) -> str:
            return await vector_db.search(query)
    """
    guard = GuardrailClient(
        api_key=api_key,
        base_url=base_url,
        project_id=project_id,
        timeout=timeout,
    )

    def decorator(fn: F) -> F:
        if inspect.iscoroutinefunction(fn):
            @functools.wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                input_text = _extract_first_string(args, kwargs)
                start = time.monotonic()
                error_caught = None
                result = None
                try:
                    result = await fn(*args, **kwargs)
                    return result
                except Exception as exc:
                    error_caught = exc
                    raise
                finally:
                    duration_ms = (time.monotonic() - start) * 1000
                    guard.log_trace({
                        "provider": "chainlit",
                        "event": "step",
                        "step_type": step_type,
                        "name": fn.__qualname__,
                        "input": str(input_text)[:2000] if input_text else "",
                        "output": str(result)[:2000] if result and not error_caught else "",
                        "duration_ms": round(duration_ms, 2),
                        "status": "error" if error_caught else "ok",
                        "error": f"{type(error_caught).__name__}: {error_caught}" if error_caught else None,
                    })

            return async_wrapper  # type: ignore[return-value]
        else:
            @functools.wraps(fn)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                input_text = _extract_first_string(args, kwargs)
                start = time.monotonic()
                error_caught = None
                result = None
                try:
                    result = fn(*args, **kwargs)
                    return result
                except Exception as exc:
                    error_caught = exc
                    raise
                finally:
                    duration_ms = (time.monotonic() - start) * 1000
                    guard.log_trace({
                        "provider": "chainlit",
                        "event": "step",
                        "step_type": step_type,
                        "name": fn.__qualname__,
                        "input": str(input_text)[:2000] if input_text else "",
                        "output": str(result)[:2000] if result and not error_caught else "",
                        "duration_ms": round(duration_ms, 2),
                        "status": "error" if error_caught else "ok",
                        "error": f"{type(error_caught).__name__}: {error_caught}" if error_caught else None,
                    })

            return sync_wrapper  # type: ignore[return-value]

    return decorator


def _extract_first_string(args: tuple, kwargs: dict) -> str:
    """Extract the first string argument from a function call."""
    for arg in args:
        if isinstance(arg, str):
            return arg
    for v in kwargs.values():
        if isinstance(v, str):
            return v
    return ""
