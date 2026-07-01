"""Gradio integration for EvalGuard.

Gradio is Hugging Face's UI framework for ML demos. This module provides:

1. A wrapper for ``gr.ChatInterface`` that auto-traces conversations
   (user input, model output, latency, errors).
2. A guardrail wrapper that checks inputs before they reach the model.
3. A generic function wrapper for any Gradio predict function.

Named ``gradio_integration.py`` (not ``gradio.py``) to avoid import
conflicts with the ``gradio`` package itself.

Usage -- Wrap a ChatInterface predict function::

    import gradio as gr
    from evalguard.gradio_integration import traced_chat, guarded_chat

    def predict(message, history):
        return my_llm(message)

    # Option 1: Tracing only (captures input, output, latency)
    demo = gr.ChatInterface(fn=traced_chat(predict, api_key="eg_..."))

    # Option 2: Tracing + guardrail (checks input, blocks violations)
    demo = gr.ChatInterface(fn=guarded_chat(
        predict,
        api_key="eg_...",
        rules=["prompt_injection"],
    ))

    demo.launch()

Usage -- GradioGuard class for full control::

    import gradio as gr
    from evalguard.gradio_integration import GradioGuard

    guard = GradioGuard(api_key="eg_...", project_id="proj_...")

    def predict(message, history):
        return my_llm(message)

    demo = gr.ChatInterface(fn=guard.wrap_chat(predict))
    demo.launch()

Usage -- Wrap any Gradio predict function (non-chat)::

    import gradio as gr
    from evalguard.gradio_integration import traced_predict

    def classify(text):
        return model.predict(text)

    demo = gr.Interface(
        fn=traced_predict(classify, api_key="eg_..."),
        inputs="text",
        outputs="label",
    )
    demo.launch()

Usage -- Streaming chat support::

    import gradio as gr
    from evalguard.gradio_integration import GradioGuard

    guard = GradioGuard(api_key="eg_...")

    def predict(message, history):
        for chunk in my_streaming_llm(message):
            yield chunk

    demo = gr.ChatInterface(fn=guard.wrap_streaming_chat(predict))
    demo.launch()
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import logging
import time
import uuid
from typing import (
    Any,
    AsyncGenerator,
    Callable,
    Dict,
    Generator,
    List,
    Optional,
    TypeVar,
    Union,
)

from .guardrails import GuardrailClient, GuardrailViolation

logger = logging.getLogger("evalguard.gradio")

F = TypeVar("F", bound=Callable[..., Any])


class GradioGuard:
    """Full-featured Gradio integration with tracing and guardrails.

    Provides wrappers for both regular and streaming chat functions,
    as well as non-chat predict functions. All interactions are traced
    to EvalGuard for monitoring.

    Parameters
    ----------
    api_key:
        EvalGuard API key.
    project_id:
        Optional project ID for trace grouping.
    base_url:
        API base URL for self-hosted deployments.
    input_rules:
        Guardrail rules for input checking.
    output_rules:
        Guardrail rules for output checking.
    block_on_violation:
        If *True*, return an error message instead of calling the model
        when input is blocked.
    block_message:
        Message returned to the user when input is blocked.
    timeout:
        HTTP request timeout in seconds.
    """

    def __init__(
        self,
        api_key: str,
        project_id: Optional[str] = None,
        base_url: str = "https://evalguard.ai/api",
        input_rules: Optional[List[str]] = None,
        output_rules: Optional[List[str]] = None,
        block_on_violation: bool = True,
        block_message: str = "Your message was blocked by a content safety guardrail.",
        timeout: float = 5.0,
    ) -> None:
        self._guard = GuardrailClient(
            api_key=api_key,
            base_url=base_url,
            project_id=project_id,
            timeout=timeout,
        )
        self._input_rules = input_rules
        self._output_rules = output_rules
        self._block = block_on_violation
        self._block_message = block_message

    def wrap_chat(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        """Wrap a ``gr.ChatInterface`` predict function with tracing and guardrails.

        The wrapped function has the same signature as the original:
        ``fn(message: str, history: list) -> str``.

        Parameters
        ----------
        fn:
            The chat predict function to wrap.

        Returns
        -------
        Callable
            Wrapped function compatible with ``gr.ChatInterface(fn=...)``.
        """
        guard = self._guard
        input_rules = self._input_rules
        output_rules = self._output_rules
        block = self._block
        block_message = self._block_message

        if inspect.iscoroutinefunction(fn):
            @functools.wraps(fn)
            async def async_wrapper(message: str, history: Any = None, *args: Any, **kwargs: Any) -> Any:
                trace_id = uuid.uuid4().hex
                violations: list = []

                # Input guardrail check
                if input_rules and message:
                    check = guard.check_input(
                        message,
                        rules=input_rules,
                        metadata={"framework": "gradio", "trace_id": trace_id},
                    )
                    violations = check.get("violations", [])
                    if not check.get("allowed", True) and block:
                        guard.log_trace({
                            "provider": "gradio",
                            "trace_id": trace_id,
                            "input": message[:2000],
                            "output": block_message,
                            "status": "blocked",
                            "violations": violations,
                            "duration_ms": 0,
                        })
                        return block_message

                start = time.monotonic()
                error_caught = None
                result = None
                try:
                    result = await fn(message, history, *args, **kwargs)
                except Exception as exc:
                    error_caught = exc
                    raise
                finally:
                    duration_ms = (time.monotonic() - start) * 1000
                    output_text = str(result)[:2000] if result and not error_caught else ""

                    # Output guardrail check
                    output_violations: list = []
                    if output_rules and output_text and not error_caught:
                        out_check = guard.check_output(
                            output_text,
                            rules=output_rules,
                            metadata={"framework": "gradio", "trace_id": trace_id},
                        )
                        output_violations = out_check.get("violations", [])

                    guard.log_trace({
                        "provider": "gradio",
                        "trace_id": trace_id,
                        "input": message[:2000] if message else "",
                        "output": output_text,
                        "duration_ms": round(duration_ms, 2),
                        "status": "error" if error_caught else "ok",
                        "error": f"{type(error_caught).__name__}: {error_caught}" if error_caught else None,
                        "input_violations": violations,
                        "output_violations": output_violations,
                        "history_length": len(history) if isinstance(history, list) else 0,
                    })

                return result

            return async_wrapper
        else:
            @functools.wraps(fn)
            def sync_wrapper(message: str, history: Any = None, *args: Any, **kwargs: Any) -> Any:
                trace_id = uuid.uuid4().hex
                violations: list = []

                # Input guardrail check
                if input_rules and message:
                    check = guard.check_input(
                        message,
                        rules=input_rules,
                        metadata={"framework": "gradio", "trace_id": trace_id},
                    )
                    violations = check.get("violations", [])
                    if not check.get("allowed", True) and block:
                        guard.log_trace({
                            "provider": "gradio",
                            "trace_id": trace_id,
                            "input": message[:2000],
                            "output": block_message,
                            "status": "blocked",
                            "violations": violations,
                            "duration_ms": 0,
                        })
                        return block_message

                start = time.monotonic()
                error_caught = None
                result = None
                try:
                    result = fn(message, history, *args, **kwargs)
                except Exception as exc:
                    error_caught = exc
                    raise
                finally:
                    duration_ms = (time.monotonic() - start) * 1000
                    output_text = str(result)[:2000] if result and not error_caught else ""

                    # Output guardrail check
                    output_violations: list = []
                    if output_rules and output_text and not error_caught:
                        out_check = guard.check_output(
                            output_text,
                            rules=output_rules,
                            metadata={"framework": "gradio", "trace_id": trace_id},
                        )
                        output_violations = out_check.get("violations", [])

                    guard.log_trace({
                        "provider": "gradio",
                        "trace_id": trace_id,
                        "input": message[:2000] if message else "",
                        "output": output_text,
                        "duration_ms": round(duration_ms, 2),
                        "status": "error" if error_caught else "ok",
                        "error": f"{type(error_caught).__name__}: {error_caught}" if error_caught else None,
                        "input_violations": violations,
                        "output_violations": output_violations,
                        "history_length": len(history) if isinstance(history, list) else 0,
                    })

                return result

            return sync_wrapper

    def wrap_streaming_chat(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        """Wrap a streaming ``gr.ChatInterface`` predict function.

        Works with generators that yield chunks. Traces the complete
        concatenated output after the stream finishes.

        Parameters
        ----------
        fn:
            A generator function: ``fn(message, history) -> Generator[str]``.

        Returns
        -------
        Callable
            Wrapped generator compatible with ``gr.ChatInterface(fn=...)``.
        """
        guard = self._guard
        input_rules = self._input_rules
        block = self._block
        block_message = self._block_message

        if inspect.isasyncgenfunction(fn):
            @functools.wraps(fn)
            async def async_gen_wrapper(message: str, history: Any = None, *args: Any, **kwargs: Any) -> AsyncGenerator[str, None]:
                trace_id = uuid.uuid4().hex

                # Input guardrail check
                if input_rules and message:
                    check = guard.check_input(
                        message,
                        rules=input_rules,
                        metadata={"framework": "gradio", "trace_id": trace_id},
                    )
                    if not check.get("allowed", True) and block:
                        guard.log_trace({
                            "provider": "gradio",
                            "trace_id": trace_id,
                            "input": message[:2000],
                            "output": block_message,
                            "status": "blocked",
                            "violations": check.get("violations", []),
                            "duration_ms": 0,
                        })
                        yield block_message
                        return

                start = time.monotonic()
                chunks: list[str] = []
                error_caught = None
                try:
                    async for chunk in fn(message, history, *args, **kwargs):
                        chunks.append(str(chunk))
                        yield chunk
                except Exception as exc:
                    error_caught = exc
                    raise
                finally:
                    duration_ms = (time.monotonic() - start) * 1000
                    full_output = "".join(chunks)
                    guard.log_trace({
                        "provider": "gradio",
                        "trace_id": trace_id,
                        "input": message[:2000] if message else "",
                        "output": full_output[:2000],
                        "duration_ms": round(duration_ms, 2),
                        "status": "error" if error_caught else "ok",
                        "error": f"{type(error_caught).__name__}: {error_caught}" if error_caught else None,
                        "streaming": True,
                        "chunks": len(chunks),
                        "history_length": len(history) if isinstance(history, list) else 0,
                    })

            return async_gen_wrapper
        else:
            @functools.wraps(fn)
            def sync_gen_wrapper(message: str, history: Any = None, *args: Any, **kwargs: Any) -> Generator[str, None, None]:
                trace_id = uuid.uuid4().hex

                # Input guardrail check
                if input_rules and message:
                    check = guard.check_input(
                        message,
                        rules=input_rules,
                        metadata={"framework": "gradio", "trace_id": trace_id},
                    )
                    if not check.get("allowed", True) and block:
                        guard.log_trace({
                            "provider": "gradio",
                            "trace_id": trace_id,
                            "input": message[:2000],
                            "output": block_message,
                            "status": "blocked",
                            "violations": check.get("violations", []),
                            "duration_ms": 0,
                        })
                        yield block_message
                        return

                start = time.monotonic()
                chunks: list[str] = []
                error_caught = None
                try:
                    for chunk in fn(message, history, *args, **kwargs):
                        chunks.append(str(chunk))
                        yield chunk
                except Exception as exc:
                    error_caught = exc
                    raise
                finally:
                    duration_ms = (time.monotonic() - start) * 1000
                    full_output = "".join(chunks)
                    guard.log_trace({
                        "provider": "gradio",
                        "trace_id": trace_id,
                        "input": message[:2000] if message else "",
                        "output": full_output[:2000],
                        "duration_ms": round(duration_ms, 2),
                        "status": "error" if error_caught else "ok",
                        "error": f"{type(error_caught).__name__}: {error_caught}" if error_caught else None,
                        "streaming": True,
                        "chunks": len(chunks),
                        "history_length": len(history) if isinstance(history, list) else 0,
                    })

            return sync_gen_wrapper

    def wrap_predict(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        """Wrap any Gradio predict function (non-chat) with tracing.

        Parameters
        ----------
        fn:
            Any function used as a Gradio Interface ``fn``.

        Returns
        -------
        Callable
            Wrapped function with tracing.
        """
        guard = self._guard
        input_rules = self._input_rules
        block = self._block
        block_message = self._block_message

        if inspect.iscoroutinefunction(fn):
            @functools.wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                trace_id = uuid.uuid4().hex
                input_text = _extract_text_input(args, kwargs)

                # Input guardrail
                if input_rules and input_text:
                    check = guard.check_input(
                        input_text,
                        rules=input_rules,
                        metadata={"framework": "gradio", "function": fn.__qualname__, "trace_id": trace_id},
                    )
                    if not check.get("allowed", True) and block:
                        guard.log_trace({
                            "provider": "gradio",
                            "trace_id": trace_id,
                            "function": fn.__qualname__,
                            "input": input_text[:2000],
                            "status": "blocked",
                            "violations": check.get("violations", []),
                        })
                        return block_message

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
                        "provider": "gradio",
                        "trace_id": trace_id,
                        "function": fn.__qualname__,
                        "input": input_text[:2000] if input_text else "",
                        "output": str(result)[:2000] if result and not error_caught else "",
                        "duration_ms": round(duration_ms, 2),
                        "status": "error" if error_caught else "ok",
                        "error": f"{type(error_caught).__name__}: {error_caught}" if error_caught else None,
                    })

            return async_wrapper
        else:
            @functools.wraps(fn)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                trace_id = uuid.uuid4().hex
                input_text = _extract_text_input(args, kwargs)

                # Input guardrail
                if input_rules and input_text:
                    check = guard.check_input(
                        input_text,
                        rules=input_rules,
                        metadata={"framework": "gradio", "function": fn.__qualname__, "trace_id": trace_id},
                    )
                    if not check.get("allowed", True) and block:
                        guard.log_trace({
                            "provider": "gradio",
                            "trace_id": trace_id,
                            "function": fn.__qualname__,
                            "input": input_text[:2000],
                            "status": "blocked",
                            "violations": check.get("violations", []),
                        })
                        return block_message

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
                        "provider": "gradio",
                        "trace_id": trace_id,
                        "function": fn.__qualname__,
                        "input": input_text[:2000] if input_text else "",
                        "output": str(result)[:2000] if result and not error_caught else "",
                        "duration_ms": round(duration_ms, 2),
                        "status": "error" if error_caught else "ok",
                        "error": f"{type(error_caught).__name__}: {error_caught}" if error_caught else None,
                    })

            return sync_wrapper


# ── Convenience functions ────────────────────────────────────────────────


def traced_chat(
    fn: Callable[..., Any],
    *,
    api_key: str,
    project_id: Optional[str] = None,
    base_url: str = "https://evalguard.ai/api",
    timeout: float = 5.0,
) -> Callable[..., Any]:
    """Wrap a ``gr.ChatInterface`` predict function with tracing.

    Convenience function that creates a :class:`GradioGuard` with
    tracing only (no guardrail rules).

    Parameters
    ----------
    fn:
        The chat predict function ``fn(message, history) -> str``.
    api_key:
        EvalGuard API key.
    project_id:
        Optional project ID.
    base_url:
        API base URL.
    timeout:
        HTTP timeout in seconds.

    Returns
    -------
    Callable
        Wrapped function for ``gr.ChatInterface(fn=...)``.

    Example::

        import gradio as gr
        from evalguard.gradio_integration import traced_chat

        def predict(message, history):
            return openai.chat(message)

        demo = gr.ChatInterface(fn=traced_chat(predict, api_key="eg_..."))
        demo.launch()
    """
    guard = GradioGuard(
        api_key=api_key,
        project_id=project_id,
        base_url=base_url,
        timeout=timeout,
        block_on_violation=False,
    )
    return guard.wrap_chat(fn)


def guarded_chat(
    fn: Callable[..., Any],
    *,
    api_key: str,
    project_id: Optional[str] = None,
    base_url: str = "https://evalguard.ai/api",
    rules: Optional[List[str]] = None,
    output_rules: Optional[List[str]] = None,
    block_on_violation: bool = True,
    block_message: str = "Your message was blocked by a content safety guardrail.",
    timeout: float = 5.0,
) -> Callable[..., Any]:
    """Wrap a ``gr.ChatInterface`` predict function with guardrails and tracing.

    Parameters
    ----------
    fn:
        The chat predict function ``fn(message, history) -> str``.
    api_key:
        EvalGuard API key.
    project_id:
        Optional project ID.
    rules:
        Input guardrail rules (e.g., ``["prompt_injection", "pii_redact"]``).
    output_rules:
        Output guardrail rules (e.g., ``["toxic_content"]``).
    block_on_violation:
        If *True*, return block_message instead of calling the model.
    block_message:
        Message shown when input is blocked.
    timeout:
        HTTP timeout in seconds.

    Returns
    -------
    Callable
        Wrapped function for ``gr.ChatInterface(fn=...)``.

    Example::

        import gradio as gr
        from evalguard.gradio_integration import guarded_chat

        def predict(message, history):
            return openai.chat(message)

        demo = gr.ChatInterface(fn=guarded_chat(
            predict,
            api_key="eg_...",
            rules=["prompt_injection"],
        ))
        demo.launch()
    """
    guard = GradioGuard(
        api_key=api_key,
        project_id=project_id,
        base_url=base_url,
        input_rules=rules,
        output_rules=output_rules,
        block_on_violation=block_on_violation,
        block_message=block_message,
        timeout=timeout,
    )
    return guard.wrap_chat(fn)


def traced_predict(
    fn: Callable[..., Any],
    *,
    api_key: str,
    project_id: Optional[str] = None,
    base_url: str = "https://evalguard.ai/api",
    rules: Optional[List[str]] = None,
    timeout: float = 5.0,
) -> Callable[..., Any]:
    """Wrap any Gradio predict function with tracing and optional guardrails.

    Parameters
    ----------
    fn:
        Any function used as ``gr.Interface(fn=...)``.
    api_key:
        EvalGuard API key.
    project_id:
        Optional project ID.
    rules:
        Optional input guardrail rules.
    timeout:
        HTTP timeout in seconds.

    Returns
    -------
    Callable
        Wrapped function.

    Example::

        import gradio as gr
        from evalguard.gradio_integration import traced_predict

        def summarize(text):
            return model.summarize(text)

        demo = gr.Interface(
            fn=traced_predict(summarize, api_key="eg_..."),
            inputs="text",
            outputs="text",
        )
        demo.launch()
    """
    guard = GradioGuard(
        api_key=api_key,
        project_id=project_id,
        base_url=base_url,
        input_rules=rules,
        timeout=timeout,
    )
    return guard.wrap_predict(fn)


# ── Helpers ──────────────────────────────────────────────────────────────


def _extract_text_input(args: tuple, kwargs: dict) -> str:
    """Extract the first string argument as the primary text input."""
    for arg in args:
        if isinstance(arg, str):
            return arg
    for v in kwargs.values():
        if isinstance(v, str):
            return v
    return ""
