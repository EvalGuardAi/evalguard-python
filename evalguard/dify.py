"""Dify integration for EvalGuard.

Dify is a visual LLM workflow platform. This module provides:

1. A webhook handler that receives Dify workflow execution callbacks and
   sends traces to EvalGuard for monitoring.
2. A ``DifyGuardrail`` class usable as a Dify tool node to check
   inputs/outputs against EvalGuard guardrails before/after LLM calls.

Usage -- Webhook handler with FastAPI::

    from fastapi import FastAPI, Request
    from evalguard.dify import DifyWebhookHandler

    app = FastAPI()
    handler = DifyWebhookHandler(api_key="eg_...", project_id="proj_...")

    @app.post("/dify/webhook")
    async def dify_webhook(request: Request):
        payload = await request.json()
        result = handler.handle(payload)
        return {"status": "ok", "trace_id": result.get("trace_id")}

Usage -- Guardrail tool node in Dify workflow::

    from evalguard.dify import DifyGuardrail

    guardrail = DifyGuardrail(api_key="eg_...", project_id="proj_...")

    # Pre-LLM check (use as input to a Dify "Code" or "Tool" node)
    result = guardrail.check_input(user_query)
    if not result["allowed"]:
        return {"error": "Blocked", "violations": result["violations"]}

    # Post-LLM check
    result = guardrail.check_output(llm_response)
    if result.get("sanitized"):
        llm_response = result["sanitized"]

Usage -- ASGI middleware for Dify webhook endpoint::

    from evalguard.dify import DifyWebhookMiddleware
    from fastapi import FastAPI

    app = FastAPI()
    app.add_middleware(
        DifyWebhookMiddleware,
        api_key="eg_...",
        project_id="proj_...",
        webhook_path="/dify/webhook",
    )
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import time
import uuid
from typing import Any, Callable, Dict, List, Optional

from .guardrails import GuardrailClient, GuardrailViolation

logger = logging.getLogger("evalguard.dify")


class DifyWebhookHandler:
    """Parses Dify workflow execution callbacks and sends traces to EvalGuard.

    Dify sends webhook payloads when workflows execute, containing:
    - ``workflow_run_id``: unique ID for the workflow execution
    - ``task_id``: the task that triggered the run
    - ``event``: event type (``workflow_started``, ``node_started``,
      ``node_finished``, ``workflow_finished``)
    - ``data``: event-specific payload with inputs, outputs, node details,
      token usage, elapsed time, etc.

    This handler normalizes the data and sends structured traces to
    EvalGuard for monitoring, cost tracking, and compliance.

    Parameters
    ----------
    api_key:
        EvalGuard API key.
    project_id:
        Optional project ID for trace grouping.
    base_url:
        API base URL for self-hosted deployments.
    signing_secret:
        Optional Dify webhook signing secret for payload verification.
    on_violation:
        Optional callback invoked when a guardrail violation is detected
        during trace processing.
    timeout:
        HTTP request timeout in seconds.
    """

    def __init__(
        self,
        api_key: str,
        project_id: Optional[str] = None,
        base_url: str = "https://evalguard.ai/api",
        signing_secret: Optional[str] = None,
        on_violation: Optional[Callable[[Dict[str, Any]], None]] = None,
        timeout: float = 5.0,
    ) -> None:
        self._guard = GuardrailClient(
            api_key=api_key,
            base_url=base_url,
            project_id=project_id,
            timeout=timeout,
        )
        self._signing_secret = signing_secret
        self._on_violation = on_violation
        # Track in-progress workflow runs for span correlation
        self._active_runs: Dict[str, Dict[str, Any]] = {}

    def verify_signature(self, payload_body: bytes, signature: str) -> bool:
        """Verify Dify webhook HMAC-SHA256 signature.

        Parameters
        ----------
        payload_body:
            Raw request body bytes.
        signature:
            The ``X-Dify-Signature`` header value.

        Returns
        -------
        bool
            True if the signature is valid or no signing secret is configured.
        """
        if not self._signing_secret:
            return True
        expected = hmac.new(
            self._signing_secret.encode(),
            payload_body,
            hashlib.sha256,
        ).hexdigest()
        return hmac.compare_digest(expected, signature)

    def handle(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Process a Dify webhook payload and send traces to EvalGuard.

        Parameters
        ----------
        payload:
            The parsed JSON body from Dify's webhook callback.

        Returns
        -------
        dict
            ``{"trace_id": str, "event": str, "status": str}``
        """
        event = payload.get("event", "unknown")
        workflow_run_id = payload.get("workflow_run_id", "") or payload.get("data", {}).get("id", "")
        task_id = payload.get("task_id", "")
        data = payload.get("data", {})

        trace_id = workflow_run_id or uuid.uuid4().hex

        if event == "workflow_started":
            return self._handle_workflow_started(trace_id, task_id, data)
        elif event == "node_started":
            return self._handle_node_started(trace_id, data)
        elif event == "node_finished":
            return self._handle_node_finished(trace_id, data)
        elif event == "workflow_finished":
            return self._handle_workflow_finished(trace_id, task_id, data)
        else:
            # Unknown event -- log it as-is for observability
            self._guard.log_trace({
                "provider": "dify",
                "event": event,
                "trace_id": trace_id,
                "task_id": task_id,
                "data": _truncate(data),
            })
            return {"trace_id": trace_id, "event": event, "status": "logged"}

    def _handle_workflow_started(
        self, trace_id: str, task_id: str, data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Track workflow start for duration calculation."""
        inputs = data.get("inputs", {})
        self._active_runs[trace_id] = {
            "start_time": time.time(),
            "inputs": inputs,
            "task_id": task_id,
            "nodes": [],
        }
        self._guard.log_trace({
            "provider": "dify",
            "event": "workflow_started",
            "trace_id": trace_id,
            "task_id": task_id,
            "inputs": _truncate(inputs),
            "workflow_id": data.get("workflow_id", ""),
        })
        return {"trace_id": trace_id, "event": "workflow_started", "status": "tracking"}

    def _handle_node_started(
        self, trace_id: str, data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Record node start for per-node tracing."""
        node_id = data.get("node_id", "")
        node_type = data.get("node_type", "")
        run_data = self._active_runs.get(trace_id)
        if run_data is not None:
            run_data["nodes"].append({
                "node_id": node_id,
                "node_type": node_type,
                "title": data.get("title", ""),
                "start_time": time.time(),
            })
        return {"trace_id": trace_id, "event": "node_started", "status": "tracking"}

    def _handle_node_finished(
        self, trace_id: str, data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Record node completion with outputs and token usage."""
        node_id = data.get("node_id", "")
        node_type = data.get("node_type", "")
        outputs = data.get("outputs", {})
        execution_metadata = data.get("execution_metadata", {})

        # Extract token usage from Dify's execution metadata
        token_usage = _extract_token_usage(execution_metadata)

        # Update the active run's node data
        run_data = self._active_runs.get(trace_id)
        node_duration_ms = 0.0
        if run_data is not None:
            for node in reversed(run_data["nodes"]):
                if node["node_id"] == node_id:
                    node["end_time"] = time.time()
                    node["outputs"] = outputs
                    node["token_usage"] = token_usage
                    node["status"] = data.get("status", "succeeded")
                    node_duration_ms = (node["end_time"] - node["start_time"]) * 1000
                    break

        self._guard.log_trace({
            "provider": "dify",
            "event": "node_finished",
            "trace_id": trace_id,
            "node_id": node_id,
            "node_type": node_type,
            "title": data.get("title", ""),
            "status": data.get("status", "succeeded"),
            "outputs": _truncate(outputs),
            "token_usage": token_usage,
            "duration_ms": round(node_duration_ms, 2),
            "error": data.get("error", None),
        })
        return {"trace_id": trace_id, "event": "node_finished", "status": "logged"}

    def _handle_workflow_finished(
        self, trace_id: str, task_id: str, data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process completed workflow: aggregate metrics and send full trace."""
        outputs = data.get("outputs", {})
        status = data.get("status", "succeeded")
        error = data.get("error", None)

        run_data = self._active_runs.pop(trace_id, {})
        start_time = run_data.get("start_time", time.time())
        total_duration_ms = (time.time() - start_time) * 1000

        # Aggregate token usage across all nodes
        total_tokens = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        node_summaries = []
        for node in run_data.get("nodes", []):
            usage = node.get("token_usage", {})
            total_tokens["prompt_tokens"] += usage.get("prompt_tokens", 0)
            total_tokens["completion_tokens"] += usage.get("completion_tokens", 0)
            total_tokens["total_tokens"] += usage.get("total_tokens", 0)
            node_summaries.append({
                "node_id": node.get("node_id"),
                "node_type": node.get("node_type"),
                "title": node.get("title"),
                "status": node.get("status", "unknown"),
            })

        # Use Dify-reported metrics if available
        elapsed_time = data.get("elapsed_time", total_duration_ms / 1000)
        dify_tokens = data.get("total_tokens", total_tokens["total_tokens"])
        dify_steps = data.get("total_steps", len(node_summaries))

        self._guard.log_trace({
            "provider": "dify",
            "event": "workflow_finished",
            "trace_id": trace_id,
            "task_id": task_id,
            "status": status,
            "inputs": _truncate(run_data.get("inputs", {})),
            "outputs": _truncate(outputs),
            "duration_ms": round(elapsed_time * 1000, 2),
            "total_tokens": dify_tokens,
            "token_usage": total_tokens,
            "total_steps": dify_steps,
            "nodes": node_summaries,
            "error": error,
        })
        return {"trace_id": trace_id, "event": "workflow_finished", "status": "logged"}


class DifyGuardrail:
    """Guardrail class for use within Dify workflow tool/code nodes.

    Use this to check user inputs before they reach an LLM node, or to
    validate LLM outputs before returning them to the user.

    Parameters
    ----------
    api_key:
        EvalGuard API key.
    project_id:
        Optional project ID for trace grouping.
    base_url:
        API base URL for self-hosted deployments.
    input_rules:
        Rules applied to input checks (default: prompt_injection, pii_redact).
    output_rules:
        Rules applied to output checks (default: toxic_content, pii_redact).
    block_on_violation:
        If *True*, :meth:`check_input` and :meth:`check_output` raise
        :class:`GuardrailViolation` when blocked.
    timeout:
        HTTP request timeout in seconds.

    Example -- Dify Code node::

        # In a Dify "Code" node placed BEFORE the LLM node:
        from evalguard.dify import DifyGuardrail

        def main(args: dict) -> dict:
            guardrail = DifyGuardrail(api_key="eg_...", project_id="proj_...")
            user_input = args["query"]

            result = guardrail.check_input(user_input)
            if not result["allowed"]:
                return {
                    "blocked": True,
                    "message": "Input blocked by guardrail",
                    "violations": result["violations"],
                }
            return {
                "blocked": False,
                "sanitized_input": result.get("sanitized") or user_input,
            }

    Example -- Dify Code node (output check)::

        # In a Dify "Code" node placed AFTER the LLM node:
        from evalguard.dify import DifyGuardrail

        def main(args: dict) -> dict:
            guardrail = DifyGuardrail(api_key="eg_...", project_id="proj_...")
            llm_output = args["llm_response"]

            result = guardrail.check_output(llm_output)
            if not result["allowed"]:
                return {"response": "I cannot provide that information."}
            return {"response": result.get("sanitized") or llm_output}
    """

    def __init__(
        self,
        api_key: str,
        project_id: Optional[str] = None,
        base_url: str = "https://evalguard.ai/api",
        input_rules: Optional[List[str]] = None,
        output_rules: Optional[List[str]] = None,
        block_on_violation: bool = False,
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

    def check_input(
        self,
        text: str,
        *,
        rules: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Check input text against guardrails before an LLM call.

        Parameters
        ----------
        text:
            The user input or prompt to validate.
        rules:
            Override rules for this specific check.
        metadata:
            Additional context (workflow_id, node_id, etc.).

        Returns
        -------
        dict
            ``{"allowed": bool, "violations": [...], "sanitized": str | None}``

        Raises
        ------
        GuardrailViolation
            If ``block_on_violation`` is *True* and the check fails.
        """
        meta = {"framework": "dify", "check_type": "input"}
        if metadata:
            meta.update(metadata)

        result = self._guard.check_input(
            text,
            rules=rules or self._input_rules,
            metadata=meta,
        )
        if not result.get("allowed", True) and self._block:
            raise GuardrailViolation(result.get("violations", []))
        return result

    def check_output(
        self,
        text: str,
        *,
        rules: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Check LLM output against guardrails before returning to user.

        Parameters
        ----------
        text:
            The LLM response to validate.
        rules:
            Override rules for this specific check.
        metadata:
            Additional context.

        Returns
        -------
        dict
            ``{"allowed": bool, "violations": [...], "sanitized": str | None}``

        Raises
        ------
        GuardrailViolation
            If ``block_on_violation`` is *True* and the check fails.
        """
        meta = {"framework": "dify", "check_type": "output"}
        if metadata:
            meta.update(metadata)

        result = self._guard.check_output(
            text,
            rules=rules or self._output_rules,
            metadata=meta,
        )
        if not result.get("allowed", True) and self._block:
            raise GuardrailViolation(result.get("violations", []))
        return result

    def log_trace(self, data: Dict[str, Any]) -> None:
        """Log a custom trace entry for Dify workflow monitoring."""
        data.setdefault("provider", "dify")
        self._guard.log_trace(data)


class DifyWebhookMiddleware:
    """ASGI middleware that intercepts Dify webhook callbacks.

    Automatically parses incoming Dify webhooks at the configured path,
    sends traces to EvalGuard, and forwards the request to the app.

    Parameters
    ----------
    app:
        The ASGI application.
    api_key:
        EvalGuard API key.
    project_id:
        Optional project ID.
    webhook_path:
        URL path to intercept (default: ``/dify/webhook``).
    signing_secret:
        Optional Dify webhook signing secret for HMAC verification.
    base_url:
        API base URL for self-hosted deployments.
    timeout:
        HTTP request timeout in seconds.
    """

    def __init__(
        self,
        app: Any,
        api_key: str,
        project_id: Optional[str] = None,
        webhook_path: str = "/dify/webhook",
        signing_secret: Optional[str] = None,
        base_url: str = "https://evalguard.ai/api",
        timeout: float = 5.0,
    ) -> None:
        self.app = app
        self._webhook_path = webhook_path
        self._handler = DifyWebhookHandler(
            api_key=api_key,
            project_id=project_id,
            base_url=base_url,
            signing_secret=signing_secret,
            timeout=timeout,
        )

    async def __call__(self, scope: Dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path: str = scope.get("path", "")
        method: str = scope.get("method", "GET")

        if method != "POST" or path != self._webhook_path:
            await self.app(scope, receive, send)
            return

        # Read the request body
        body_chunks: list[bytes] = []
        request_complete = False

        while not request_complete:
            message = await receive()
            if message["type"] == "http.request":
                body_chunks.append(message.get("body", b""))
                if not message.get("more_body", False):
                    request_complete = True

        body_bytes = b"".join(body_chunks)

        # Verify signature if configured
        signature = ""
        for header_name, header_value in scope.get("headers", []):
            if header_name == b"x-dify-signature":
                signature = header_value.decode()
                break

        if not self._handler.verify_signature(body_bytes, signature):
            error_body = json.dumps({"error": "Invalid signature"}).encode()
            await send({
                "type": "http.response.start",
                "status": 401,
                "headers": [
                    [b"content-type", b"application/json"],
                    [b"content-length", str(len(error_body)).encode()],
                ],
            })
            await send({"type": "http.response.body", "body": error_body})
            return

        # Process the webhook
        try:
            payload = json.loads(body_bytes)
            result = self._handler.handle(payload)
            response_body = json.dumps(result).encode()
            status = 200
        except (json.JSONDecodeError, UnicodeDecodeError):
            response_body = json.dumps({"error": "Invalid JSON"}).encode()
            status = 400
        except Exception as exc:
            logger.error("Dify webhook processing failed: %s", exc, exc_info=True)
            response_body = json.dumps({"error": "Internal error"}).encode()
            status = 500

        await send({
            "type": "http.response.start",
            "status": status,
            "headers": [
                [b"content-type", b"application/json"],
                [b"content-length", str(len(response_body)).encode()],
            ],
        })
        await send({"type": "http.response.body", "body": response_body})


# ── Helpers ──────────────────────────────────────────────────────────────


def _extract_token_usage(execution_metadata: Dict[str, Any]) -> Dict[str, int]:
    """Extract token usage from Dify node execution metadata."""
    usage = execution_metadata.get("usage", {})
    if not usage:
        # Some Dify versions nest under total_tokens directly
        return {
            "prompt_tokens": execution_metadata.get("prompt_tokens", 0),
            "completion_tokens": execution_metadata.get("completion_tokens", 0),
            "total_tokens": execution_metadata.get("total_tokens", 0),
        }
    return {
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "total_tokens": usage.get("total_tokens", 0),
    }


def _truncate(obj: Any, max_str_len: int = 2000) -> Any:
    """Truncate large values for trace payloads."""
    if isinstance(obj, str):
        return obj[:max_str_len] if len(obj) > max_str_len else obj
    if isinstance(obj, dict):
        return {k: _truncate(v, max_str_len) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        items = [_truncate(v, max_str_len) for v in obj[:50]]
        if len(obj) > 50:
            items.append(f"... +{len(obj) - 50} more")
        return items
    return obj
