"""Core guardrail client shared by all framework integrations.

Provides pre-LLM input checking (prompt injection, PII redaction) and
post-LLM trace logging.  Every framework wrapper delegates to a single
:class:`GuardrailClient` instance so configuration is consistent.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger("evalguard.guardrails")

_DEFAULT_RULES: List[str] = ["prompt_injection", "pii_redact"]


class GuardrailClient:
    """Lightweight HTTP client for the EvalGuard guardrail & trace APIs.

    Parameters
    ----------
    api_key:
        EvalGuard API key (``eg_live_...`` or ``eg_test_...``).
    base_url:
        API base URL.  Override for self-hosted deployments.
    project_id:
        Optional project ID attached to every trace.
    timeout:
        HTTP request timeout in seconds.  Keep low (default 5 s) so the
        guardrail check never dominates end-to-end latency.
    fail_open:
        If *True*, network / server errors allow the request through
        instead of raising.  Defaults to *False*.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://evalguard.ai/api",
        project_id: Optional[str] = None,
        timeout: float = 5.0,
        fail_open: bool = False,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._project_id = project_id
        self._timeout = timeout
        self._fail_open = fail_open
        self._session = requests.Session()
        self._session.headers.update(
            {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "User-Agent": "evalguard-sdk-python/2.0.1-guardrails",
            }
        )

    # ── Public API ────────────────────────────────────────────────────

    @staticmethod
    def _translate(body: Any) -> Dict[str, Any]:
        """Map the guardrail API response → ``{allowed, violations, sanitized}``.

        The endpoint replies with the standard envelope ``{success, data}`` where
        ``data`` is the firewall result ``{action, reasons, latencyMs}`` (action
        ∈ allow|flag|block). Older/non-enveloped payloads are tolerated by
        reading the top level directly.
        """
        result = body.get("data", body) if isinstance(body, dict) else {}
        if not isinstance(result, dict):
            result = {}
        action = result.get("action", "allow")
        # Firewall is 3-state: only an outright BLOCK disallows the call; a flag
        # is allowed-with-warning (violations still carry the detail). Honor an
        # explicit `allowed` if a future endpoint returns one.
        allowed = result.get("allowed", action != "block")
        violations = result.get("violations")
        if violations is None:
            violations = result.get("reasons", [])
        return {
            "allowed": bool(allowed),
            "violations": violations,
            "sanitized": result.get("sanitized"),
            "action": action,
        }

    def _safe_default(self) -> Dict[str, Any]:
        return {"allowed": True, "violations": [], "sanitized": None, "action": "allow"}

    def check_input(
        self,
        text: str,
        rules: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Pre-LLM guard check.

        Returns
        -------
        dict
            ``{"allowed": bool, "violations": [...], "sanitized": str | None}``
        """
        # The API contract is { text, projectId } (camelCase) — NOT { input,
        # project_id }. Sending the old keys made `text` (required) absent → a
        # 400 on every call, and dropped project scoping (custom rules never
        # loaded). `rules`/`metadata` ride along as harmless passthrough extras.
        payload: Dict[str, Any] = {
            "text": text,
            "rules": rules or _DEFAULT_RULES,
        }
        if self._project_id:
            payload["projectId"] = self._project_id
        if metadata:
            payload["metadata"] = metadata
        try:
            resp = self._session.post(
                f"{self._base_url}/v1/guardrails",
                json=payload,
                timeout=self._timeout,
            )
            resp.raise_for_status()
            return self._translate(resp.json())
        except Exception:
            if self._fail_open:
                logger.debug("Guardrail check failed; fail-open allowing request", exc_info=True)
                return self._safe_default()
            raise

    def check_output(
        self,
        text: str,
        rules: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Post-LLM output check.

        Returns
        -------
        dict
            ``{"allowed": bool, "violations": [...], "sanitized": str | None}``
        """
        # There is no separate `/guardrails/output` endpoint (it 404'd); the
        # `/api/v1/guardrails` route runs the firewall on whatever `text` it is
        # given. Send the output text under the same `text` key + translate the
        # firewall result identically.
        payload: Dict[str, Any] = {
            "text": text,
            "rules": rules or ["toxic_content", "pii_redact"],
        }
        if self._project_id:
            payload["projectId"] = self._project_id
        if metadata:
            payload["metadata"] = metadata
        try:
            resp = self._session.post(
                f"{self._base_url}/v1/guardrails",
                json=payload,
                timeout=self._timeout,
            )
            resp.raise_for_status()
            return self._translate(resp.json())
        except Exception:
            if self._fail_open:
                logger.debug("Output check failed; fail-open allowing response", exc_info=True)
                return self._safe_default()
            raise

    def log_trace(self, data: Dict[str, Any]) -> None:
        """Fire-and-forget trace logging.  Errors are silently swallowed."""
        payload = {**data}
        if self._project_id:
            payload.setdefault("project_id", self._project_id)
        try:
            self._session.post(
                f"{self._base_url}/v1/traces",
                json=payload,
                timeout=self._timeout,
            )
        except Exception:
            logger.debug("Trace log failed", exc_info=True)

    def close(self) -> None:
        """Close the underlying HTTP session."""
        self._session.close()


class GuardrailViolation(Exception):
    """Raised when a guardrail check blocks a request."""

    def __init__(self, violations: List[Dict[str, Any]], message: str = "Blocked by EvalGuard guardrail"):
        super().__init__(message)
        self.violations = violations
