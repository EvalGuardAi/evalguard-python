"""EvalGuard Python SDK -- @traceable decorator and trace() context manager.

Zero-config function tracing that automatically captures function name, args,
return values, duration, and errors, then sends trace spans to the EvalGuard API.

Usage::

    from evalguard import traceable, trace

    @traceable
    def my_llm_call(prompt: str) -> str:
        return openai.chat(prompt)

    @traceable(name="custom-name", metadata={"model": "gpt-4o"})
    async def my_async_call(prompt: str) -> str:
        return await openai.achat(prompt)

    with trace("data-preprocessing") as span:
        data = load_data()
        span.metadata["rows"] = len(data)

Environment variables:
    EVALGUARD_API_KEY   -- API key for authentication
    EVALGUARD_BASE_URL  -- API base URL (default: https://evalguard.ai/api/v1)
    EVALGUARD_PROJECT_ID -- Default project ID for traces
    EVALGUARD_TRACING_ENABLED -- Set to "false" to disable (default: "true")
"""

from __future__ import annotations

import asyncio
import atexit
import functools
import inspect
import logging
import os
import re
import threading
import time
import traceback as tb_module
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    List,
    Optional,
    TypeVar,
    Union,
    overload,
)

import requests

logger = logging.getLogger("evalguard.tracing")

# ── Secret redaction ─────────────────────────────────────────────────────
#
# The @traceable decorator captures every function argument (and any metadata
# the caller attaches) via inspect.signature().bind — including an api_key,
# token, password, or Authorization header passed as a kwarg. Sending those to
# the ingest endpoint in clear is a credential leak, so we redact BEFORE the
# span leaves the process:
#   1. by KEY name — any dict key matching a secret pattern → "[REDACTED]"
#   2. by VALUE shape — any string that looks like a known secret token
#      (eg_*, sk-*, Bearer …, JWT, GitHub/AWS/Slack/Google tokens) → "[REDACTED]"
# Redaction is deep (dicts + lists/tuples) and runs inside _safe_serialize's
# recursion so it covers nested inputs/outputs/metadata.

_REDACTED = "[REDACTED]"

_SECRET_KEY_RE = re.compile(
    r"(api[_-]?key|secret|token|password|passwd|authorization|auth[_-]?token"
    r"|access[_-]?key|private[_-]?key|client[_-]?secret|bearer|credential"
    r"|session[_-]?id|cookie)",
    re.IGNORECASE,
)

# Value-shape patterns for common secret tokens; each must match the WHOLE
# string (after strip) so ordinary prose is never masked.
_SECRET_VALUE_RES = [
    re.compile(r"^eg_[A-Za-z0-9_-]{8,}$"),               # EvalGuard API keys
    re.compile(r"^sk-ant-[A-Za-z0-9_-]{16,}$"),          # Anthropic keys
    re.compile(r"^sk-[A-Za-z0-9_-]{16,}$"),              # OpenAI-style keys
    re.compile(r"^xox[baprs]-[A-Za-z0-9-]{10,}$"),       # Slack tokens
    re.compile(r"^gh[posru]_[A-Za-z0-9]{20,}$"),         # GitHub tokens
    re.compile(r"^AKIA[0-9A-Z]{16}$"),                   # AWS access key id
    re.compile(r"^ya29\.[A-Za-z0-9_-]{20,}$"),           # Google OAuth tokens
    re.compile(r"^Bearer\s+[A-Za-z0-9._-]{12,}$", re.IGNORECASE),  # Authorization
    re.compile(r"^eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}$"),  # JWT
]


def _is_secret_key(key: str) -> bool:
    """True when a dict key name implies its value is a secret."""
    return bool(_SECRET_KEY_RE.search(key))


def _looks_secret_value(s: str) -> bool:
    t = s.strip()
    if len(t) < 8:
        return False
    return any(rx.match(t) for rx in _SECRET_VALUE_RES)

F = TypeVar("F", bound=Callable[..., Any])

# ── Context propagation ─────────────────────────────────────────────────

_current_span: ContextVar[Optional["Span"]] = ContextVar("_current_span", default=None)
_current_trace_id: ContextVar[Optional[str]] = ContextVar("_current_trace_id", default=None)

# Distributed-tracing identity (observability-tracing-3). Set via set_session();
# auto-attached to every span created afterwards in this/child context(s) and
# inherited by child spans (ContextVars propagate into child contexts).
_current_session_id: ContextVar[Optional[str]] = ContextVar("_current_session_id", default=None)
_current_user_id: ContextVar[Optional[str]] = ContextVar("_current_user_id", default=None)
_current_conversation_id: ContextVar[Optional[str]] = ContextVar("_current_conversation_id", default=None)


def set_session(
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    conversation_id: Optional[str] = None,
) -> None:
    """Set trace identity for the current context.

    Each provided id is stored in a ContextVar and AUTO-ATTACHED to every span
    created afterwards in this (and child) context(s) as the dotted span
    attributes ``session.id`` / ``user.id`` / ``conversation.id``. Never
    auto-populated from PII — pass the ids explicitly. (observability-tracing-3)
    """
    if session_id is not None:
        _current_session_id.set(session_id)
    if user_id is not None:
        _current_user_id.set(user_id)
    if conversation_id is not None:
        _current_conversation_id.set(conversation_id)


def get_session() -> Dict[str, Optional[str]]:
    """Return the current trace identity (session/user/conversation; values may be None)."""
    return {
        "session_id": _current_session_id.get(),
        "user_id": _current_user_id.get(),
        "conversation_id": _current_conversation_id.get(),
    }


def _attach_identity(metadata: Dict[str, Any]) -> None:
    """Merge the current trace identity into ``metadata`` as dotted attributes.

    Caller-supplied metadata wins (we never overwrite an explicit key). The dotted
    form is redaction-safe: the secret-key matcher catches ``session_id`` but not
    ``session.id``.
    """
    sid = _current_session_id.get()
    uid = _current_user_id.get()
    cid = _current_conversation_id.get()
    if sid is not None and "session.id" not in metadata:
        metadata["session.id"] = sid
    if uid is not None and "user.id" not in metadata:
        metadata["user.id"] = uid
    if cid is not None and "conversation.id" not in metadata:
        metadata["conversation.id"] = cid


# ── Span dataclass ──────────────────────────────────────────────────────

@dataclass
class Span:
    """Represents a single trace span."""

    span_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    trace_id: str = ""
    parent_span_id: Optional[str] = None
    name: str = ""
    start_time: float = 0.0
    end_time: float = 0.0
    duration_ms: float = 0.0
    status: str = "ok"  # "ok" | "error"
    inputs: Dict[str, Any] = field(default_factory=dict)
    outputs: Any = None
    error: Optional[str] = None
    error_traceback: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "spanId": self.span_id,
            "traceId": self.trace_id,
            "name": self.name,
            "startTime": self.start_time,
            "endTime": self.end_time,
            "durationMs": self.duration_ms,
            "status": self.status,
        }
        if self.parent_span_id:
            d["parentSpanId"] = self.parent_span_id
        if self.inputs:
            d["inputs"] = _safe_serialize(self.inputs)
        if self.outputs is not None:
            d["outputs"] = _safe_serialize(self.outputs)
        if self.error:
            d["error"] = self.error
        if self.error_traceback:
            d["errorTraceback"] = self.error_traceback
        if self.metadata:
            d["metadata"] = _safe_serialize(self.metadata)
        return d


# ── Serialization helper ────────────────────────────────────────────────

def _safe_serialize(
    obj: Any,
    max_depth: int = 4,
    max_str_len: int = 4096,
    key_hint: Optional[str] = None,
) -> Any:
    """Best-effort JSON-safe serialization with depth/size limits + secret redaction."""
    if max_depth <= 0:
        return "<truncated>"
    if obj is None:
        return obj
    # Redact a whole value when its key name implies it's a secret, regardless
    # of the value's type/shape (e.g. password=1234, token={...}).
    if key_hint is not None and _is_secret_key(key_hint):
        return _REDACTED
    if isinstance(obj, (bool, int, float)):
        return obj
    if isinstance(obj, str):
        if _looks_secret_value(obj):
            return _REDACTED
        return obj[:max_str_len] if len(obj) > max_str_len else obj
    if isinstance(obj, bytes):
        return f"<bytes len={len(obj)}>"
    if isinstance(obj, dict):
        return {str(k): _safe_serialize(v, max_depth - 1, max_str_len, str(k)) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        items = [_safe_serialize(v, max_depth - 1, max_str_len) for v in obj[:100]]
        if len(obj) > 100:
            items.append(f"... +{len(obj) - 100} more")
        return items
    # Fallback: repr
    try:
        s = repr(obj)
        return s[:max_str_len] if len(s) > max_str_len else s
    except Exception:
        return f"<{type(obj).__name__}>"


# ── Background batch sender ────────────────────────────────────────────

class _TraceBatcher:
    """Thread-safe background batcher that flushes spans to the API."""

    def __init__(self) -> None:
        self._queue: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        self._flush_interval = 2.0  # seconds
        self._max_batch_size = 50
        self._timer: Optional[threading.Timer] = None
        self._session: Optional[requests.Session] = None
        self._stopped = False

    @property
    def _api_key(self) -> str:
        return os.environ.get("EVALGUARD_API_KEY", "")

    @property
    def _base_url(self) -> str:
        return os.environ.get("EVALGUARD_BASE_URL", "https://evalguard.ai/api").rstrip("/")

    @property
    def _project_id(self) -> str:
        return os.environ.get("EVALGUARD_PROJECT_ID", "")

    @property
    def _enabled(self) -> bool:
        return os.environ.get("EVALGUARD_TRACING_ENABLED", "true").lower() != "false"

    def _get_session(self) -> requests.Session:
        if self._session is None:
            self._session = requests.Session()
            self._session.headers.update({
                "Content-Type": "application/json",
                "User-Agent": "evalguard-sdk-python/2.0.1-tracing",
            })
        return self._session

    def enqueue(self, span_dict: Dict[str, Any]) -> None:
        if not self._enabled or not self._api_key:
            return
        with self._lock:
            self._queue.append(span_dict)
            if len(self._queue) >= self._max_batch_size:
                self._flush_locked()
            elif self._timer is None:
                self._timer = threading.Timer(self._flush_interval, self._flush)
                self._timer.daemon = True
                self._timer.start()

    def _flush(self) -> None:
        with self._lock:
            self._flush_locked()

    def _flush_locked(self) -> None:
        """Must be called under self._lock."""
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None

        if not self._queue:
            return

        batch = self._queue[:]
        self._queue.clear()

        # Send in a background thread so we never block
        t = threading.Thread(target=self._send, args=(batch,), daemon=True)
        t.start()

    def _send(self, batch: List[Dict[str, Any]]) -> None:
        try:
            session = self._get_session()
            url = f"{self._base_url}/v1/traces/ingest"
            payload = {
                "projectId": self._project_id,
                "spans": batch,
            }
            session.post(
                url,
                json=payload,
                headers={"Authorization": f"Bearer {self._api_key}"},
                timeout=10,
            )
        except Exception:
            logger.debug("Failed to send trace batch", exc_info=True)

    def shutdown(self) -> None:
        """Flush remaining spans synchronously (called at exit)."""
        if self._stopped:
            return
        self._stopped = True
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None
            if self._queue:
                batch = self._queue[:]
                self._queue.clear()
                # Send synchronously on shutdown
                self._send(batch)


_batcher = _TraceBatcher()
atexit.register(_batcher.shutdown)


# ── Core tracing logic ──────────────────────────────────────────────────

def _capture_inputs(fn: Callable[..., Any], args: tuple, kwargs: dict) -> Dict[str, Any]:
    """Extract function arguments as a serializable dict."""
    try:
        sig = inspect.signature(fn)
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        return dict(bound.arguments)
    except Exception:
        # Fallback: positional + keyword
        inputs: Dict[str, Any] = {}
        if args:
            inputs["args"] = list(args)
        if kwargs:
            inputs["kwargs"] = kwargs
        return inputs


def _start_span(name: str, inputs: Dict[str, Any], metadata: Dict[str, Any]) -> Span:
    parent = _current_span.get()
    trace_id = _current_trace_id.get() or uuid.uuid4().hex
    span_metadata = dict(metadata)
    _attach_identity(span_metadata)
    span = Span(
        trace_id=trace_id,
        parent_span_id=parent.span_id if parent else None,
        name=name,
        start_time=time.time(),
        inputs=inputs,
        metadata=span_metadata,
    )
    return span


def _finish_span(span: Span, output: Any = None, error: Optional[BaseException] = None) -> None:
    span.end_time = time.time()
    span.duration_ms = (span.end_time - span.start_time) * 1000

    if error is not None:
        span.status = "error"
        span.error = f"{type(error).__name__}: {error}"
        span.error_traceback = tb_module.format_exc()
    else:
        span.status = "ok"
        span.outputs = output

    _batcher.enqueue(span.to_dict())


# ── @traceable decorator ────────────────────────────────────────────────

@overload
def traceable(fn: F) -> F: ...


@overload
def traceable(
    *,
    name: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Callable[[F], F]: ...


def traceable(
    fn: Optional[F] = None,
    *,
    name: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Union[F, Callable[[F], F]]:
    """Decorator that traces a sync or async function.

    Can be used bare or with arguments::

        @traceable
        def foo(): ...

        @traceable(name="custom", metadata={"tier": "prod"})
        async def bar(): ...
    """

    def decorator(func: F) -> F:
        span_name = name or func.__qualname__
        extra_meta = metadata or {}

        if inspect.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                inputs = _capture_inputs(func, args, kwargs)
                span = _start_span(span_name, inputs, extra_meta)

                token_span = _current_span.set(span)
                token_trace = _current_trace_id.set(span.trace_id)
                try:
                    result = await func(*args, **kwargs)
                    _finish_span(span, output=result)
                    return result
                except BaseException as exc:
                    _finish_span(span, error=exc)
                    raise
                finally:
                    _current_span.reset(token_span)
                    _current_trace_id.reset(token_trace)

            return async_wrapper  # type: ignore[return-value]
        else:
            @functools.wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                inputs = _capture_inputs(func, args, kwargs)
                span = _start_span(span_name, inputs, extra_meta)

                token_span = _current_span.set(span)
                token_trace = _current_trace_id.set(span.trace_id)
                try:
                    result = func(*args, **kwargs)
                    _finish_span(span, output=result)
                    return result
                except BaseException as exc:
                    _finish_span(span, error=exc)
                    raise
                finally:
                    _current_span.reset(token_span)
                    _current_trace_id.reset(token_trace)

            return sync_wrapper  # type: ignore[return-value]

    if fn is not None:
        return decorator(fn)
    return decorator  # type: ignore[return-value]


# ── trace() context manager ─────────────────────────────────────────────

@contextmanager
def trace(
    name: str,
    *,
    metadata: Optional[Dict[str, Any]] = None,
) -> Generator[Span, None, None]:
    """Context manager for manual tracing::

        with trace("my-step", metadata={"key": "val"}) as span:
            result = do_something()
            span.outputs = result
    """
    span = _start_span(name, {}, metadata or {})
    token_span = _current_span.set(span)
    token_trace = _current_trace_id.set(span.trace_id)
    try:
        yield span
        if span.status != "error":
            _finish_span(span, output=span.outputs)
    except BaseException as exc:
        _finish_span(span, error=exc)
        raise
    finally:
        _current_span.reset(token_span)
        _current_trace_id.reset(token_trace)


# ── Utilities ───────────────────────────────────────────────────────────

def get_current_span() -> Optional[Span]:
    """Return the current active span, or None."""
    return _current_span.get()


def get_current_trace_id() -> Optional[str]:
    """Return the current trace ID, or None."""
    return _current_trace_id.get()


def flush() -> None:
    """Force-flush all pending spans. Useful in tests or before process exit."""
    _batcher._flush()


def configure(
    *,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    project_id: Optional[str] = None,
    enabled: Optional[bool] = None,
) -> None:
    """Programmatic configuration (alternative to env vars).

    Sets the corresponding environment variables so the batcher picks them up.
    """
    if api_key is not None:
        os.environ["EVALGUARD_API_KEY"] = api_key
    if base_url is not None:
        os.environ["EVALGUARD_BASE_URL"] = base_url
    if project_id is not None:
        os.environ["EVALGUARD_PROJECT_ID"] = project_id
    if enabled is not None:
        os.environ["EVALGUARD_TRACING_ENABLED"] = str(enabled).lower()
