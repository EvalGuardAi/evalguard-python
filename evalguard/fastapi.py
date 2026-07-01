"""FastAPI middleware for EvalGuard.

Usage::

    from evalguard.fastapi import EvalGuardMiddleware
    from fastapi import FastAPI

    app = FastAPI()
    app.add_middleware(
        EvalGuardMiddleware,
        api_key="eg_...",
        project_id="proj_...",
    )
    # All matching endpoints are now guarded

You can also use the route decorator for fine-grained control::

    from evalguard.fastapi import guard_route

    @app.post("/api/chat")
    @guard_route(api_key="eg_...", rules=["prompt_injection"])
    async def chat(request: Request):
        ...
"""

from __future__ import annotations

import functools
import json
import time
from typing import Any, Callable, Dict, List, Optional, Set

from .guardrails import GuardrailClient, GuardrailViolation


class EvalGuardMiddleware:
    """ASGI middleware that guards incoming requests.

    By default, guards POST requests to paths containing ``/chat``,
    ``/completions``, ``/generate``, or ``/invoke``.  Customize via
    ``guarded_paths``.

    Parameters
    ----------
    app:
        The ASGI application.
    api_key:
        EvalGuard API key.
    project_id:
        Optional project ID.
    rules:
        Guardrail rules for input checking.
    guarded_paths:
        URL path substrings that trigger guardrail checks.
    block_on_violation:
        If *True*, return 403 when input is blocked.
    """

    _DEFAULT_PATHS = {"/chat", "/completions", "/generate", "/invoke", "/messages"}

    def __init__(
        self,
        app: Any,
        api_key: str,
        project_id: Optional[str] = None,
        base_url: str = "https://evalguard.ai/api",
        rules: Optional[List[str]] = None,
        guarded_paths: Optional[Set[str]] = None,
        block_on_violation: bool = True,
        timeout: float = 5.0,
    ) -> None:
        self.app = app
        self._guard = GuardrailClient(
            api_key=api_key,
            base_url=base_url,
            project_id=project_id,
            timeout=timeout,
        )
        self._rules = rules
        self._paths = guarded_paths or self._DEFAULT_PATHS
        self._block = block_on_violation

    async def __call__(self, scope: Dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path: str = scope.get("path", "")
        method: str = scope.get("method", "GET")

        # Only guard POST/PUT requests to matching paths
        if method not in ("POST", "PUT") or not self._should_guard(path):
            await self.app(scope, receive, send)
            return

        # Read the request body
        body_chunks: list[bytes] = []
        request_complete = False

        async def receive_wrapper() -> Dict[str, Any]:
            nonlocal request_complete
            if body_chunks and request_complete:
                # Replay the body for the downstream app
                return {"type": "http.request", "body": b"".join(body_chunks), "more_body": False}
            message = await receive()
            if message["type"] == "http.request":
                body_chunks.append(message.get("body", b""))
                if not message.get("more_body", False):
                    request_complete = True
            return message

        # Consume the body first
        while not request_complete:
            await receive_wrapper()

        body_bytes = b"".join(body_chunks)
        prompt_text = _extract_body_text(body_bytes)

        if prompt_text:
            start = time.monotonic()
            check = self._guard.check_input(
                prompt_text,
                rules=self._rules,
                metadata={"path": path, "method": method, "framework": "fastapi"},
            )
            guard_ms = (time.monotonic() - start) * 1000

            if not check.get("allowed", True) and self._block:
                # Return 403 Forbidden
                response_body = json.dumps({
                    "error": "Blocked by EvalGuard guardrail",
                    "violations": check.get("violations", []),
                }).encode()
                await send({
                    "type": "http.response.start",
                    "status": 403,
                    "headers": [
                        [b"content-type", b"application/json"],
                        [b"content-length", str(len(response_body)).encode()],
                    ],
                })
                await send({
                    "type": "http.response.body",
                    "body": response_body,
                })
                return

        # Pass through to the app with replayed body
        request_complete = True  # ensure replay mode
        start = time.monotonic()
        await self.app(scope, receive_wrapper, send)
        llm_ms = (time.monotonic() - start) * 1000

        # Best-effort trace log
        self._guard.log_trace({
            "provider": "fastapi",
            "path": path,
            "input": prompt_text[:2000] if prompt_text else "",
            "guard_latency_ms": round(guard_ms, 2) if prompt_text else 0,
            "request_latency_ms": round(llm_ms, 2),
        })

    def _should_guard(self, path: str) -> bool:
        return any(p in path for p in self._paths)


def guard_route(
    *,
    api_key: str,
    project_id: Optional[str] = None,
    base_url: str = "https://evalguard.ai/api",
    rules: Optional[List[str]] = None,
    block_on_violation: bool = True,
    timeout: float = 5.0,
) -> Callable:
    """Decorator for guarding individual FastAPI route handlers.

    Usage::

        @app.post("/api/chat")
        @guard_route(api_key="eg_...")
        async def chat(request: Request):
            body = await request.json()
            ...
    """
    guard = GuardrailClient(
        api_key=api_key,
        base_url=base_url,
        project_id=project_id,
        timeout=timeout,
    )

    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Try to find the Request object
            request = None
            for arg in args:
                if hasattr(arg, "json") and hasattr(arg, "method"):
                    request = arg
                    break
            for v in kwargs.values():
                if hasattr(v, "json") and hasattr(v, "method"):
                    request = v
                    break

            if request:
                try:
                    body = await request.json()
                    prompt_text = _extract_dict_text(body)
                    if prompt_text:
                        check = guard.check_input(prompt_text, rules=rules)
                        if not check.get("allowed", True) and block_on_violation:
                            # Import here to avoid hard dependency
                            try:
                                from fastapi.responses import JSONResponse
                                return JSONResponse(
                                    status_code=403,
                                    content={
                                        "error": "Blocked by EvalGuard guardrail",
                                        "violations": check.get("violations", []),
                                    },
                                )
                            except ImportError:
                                raise GuardrailViolation(check.get("violations", []))
                except Exception:
                    pass  # fail-open

            return await fn(*args, **kwargs)

        return wrapper

    return decorator


def _extract_body_text(body: bytes) -> str:
    """Extract prompt text from a JSON request body."""
    try:
        data = json.loads(body)
        return _extract_dict_text(data)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return ""


def _extract_dict_text(data: Any) -> str:
    """Extract prompt/message text from a parsed JSON body."""
    if not isinstance(data, dict):
        return ""

    # Direct prompt field
    if "prompt" in data:
        return data["prompt"] if isinstance(data["prompt"], str) else str(data["prompt"])

    # OpenAI-style messages
    messages = data.get("messages", [])
    if messages and isinstance(messages, list):
        parts: list[str] = []
        for msg in messages:
            if isinstance(msg, dict):
                content = msg.get("content", "")
                if isinstance(content, str):
                    parts.append(content)
        return "\n".join(parts)

    # Input field (common in many APIs)
    if "input" in data:
        return data["input"] if isinstance(data["input"], str) else str(data["input"])

    # Query field
    if "query" in data:
        return data["query"] if isinstance(data["query"], str) else str(data["query"])

    return ""
