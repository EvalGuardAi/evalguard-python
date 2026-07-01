"""BrowserUse integration for EvalGuard.

Usage::

    from evalguard.browseruse import EvalGuardBrowserCallback

    from browseruse import Agent
    from langchain_openai import ChatOpenAI

    callback = EvalGuardBrowserCallback(
        api_key="eg_...",
        project_id="proj_...",
        blocked_domains=["evil.com", "*.malware.net"],
        block_sensitive_fields=True,
    )

    agent = Agent(
        task="Find the best restaurant in SF",
        llm=ChatOpenAI(model="gpt-4o"),
        on_step_callback=callback.on_step,
    )
    await agent.run()

    # Access the full trace after the run
    trace = callback.get_trace()

Works with BrowserUse >= 0.1.0.  No hard dependency on the ``browseruse``
package -- the integration duck-types against the callback protocol.
"""

from __future__ import annotations

import fnmatch
import logging
import re
import time
import uuid
from typing import Any, Dict, List, Optional, Set
from urllib.parse import urlparse

from .guardrails import GuardrailClient, GuardrailViolation

logger = logging.getLogger("evalguard.browseruse")

# Default patterns for sensitive form fields (name attributes / labels)
_SENSITIVE_FIELD_PATTERNS: List[str] = [
    r"(?i)password",
    r"(?i)passwd",
    r"(?i)ssn",
    r"(?i)social.?security",
    r"(?i)credit.?card",
    r"(?i)card.?number",
    r"(?i)cvv",
    r"(?i)cvc",
    r"(?i)secret",
    r"(?i)api.?key",
    r"(?i)token",
    r"(?i)bank.?account",
    r"(?i)routing.?number",
]

_SENSITIVE_FIELD_RE = [re.compile(p) for p in _SENSITIVE_FIELD_PATTERNS]


class EvalGuardBrowserCallback:
    """BrowserUse callback handler that traces browser agent activity and
    applies guardrails to prevent unsafe navigations and data submissions.

    Captures page navigations, actions (click, type, scroll), screenshots,
    and LLM decisions.  Sends structured traces to EvalGuard.  Optionally
    blocks navigations to unsafe URLs and prevents the agent from submitting
    data into sensitive form fields.

    Parameters
    ----------
    api_key:
        EvalGuard API key.
    project_id:
        Optional project ID for trace grouping.
    rules:
        Guardrail rules for content checking.
    block_on_violation:
        Raise :class:`GuardrailViolation` when a guardrail check fails.
    blocked_domains:
        Domain patterns to block navigation to.  Supports glob patterns
        (e.g. ``"*.malware.net"``).  The agent will be prevented from
        navigating to any matching URL.
    blocked_url_patterns:
        Regex patterns for URLs to block (e.g. ``r"https://evil\\.com/.*"``).
    block_sensitive_fields:
        If *True*, prevent the agent from typing into form fields that
        match sensitive patterns (password, SSN, credit card, etc.).
    sensitive_field_patterns:
        Additional regex patterns for sensitive field detection.  These
        are checked against field ``name``, ``id``, ``placeholder``, and
        ``aria-label`` attributes.
    check_extracted_content:
        If *True*, run guardrail checks on content extracted from pages.
    """

    def __init__(
        self,
        api_key: str,
        project_id: Optional[str] = None,
        base_url: str = "https://evalguard.ai/api",
        rules: Optional[List[str]] = None,
        block_on_violation: bool = True,
        blocked_domains: Optional[List[str]] = None,
        blocked_url_patterns: Optional[List[str]] = None,
        block_sensitive_fields: bool = True,
        sensitive_field_patterns: Optional[List[str]] = None,
        check_extracted_content: bool = False,
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
        self._blocked_domains = blocked_domains or []
        self._blocked_url_re = [re.compile(p) for p in (blocked_url_patterns or [])]
        self._block_sensitive = block_sensitive_fields
        self._sensitive_re = list(_SENSITIVE_FIELD_RE)
        if sensitive_field_patterns:
            self._sensitive_re.extend(re.compile(p) for p in sensitive_field_patterns)
        self._check_content = check_extracted_content

        # Session state
        self._session_id = str(uuid.uuid4())
        self._session_start = time.monotonic()
        self._steps: List[Dict[str, Any]] = []
        self._navigations: List[Dict[str, Any]] = []
        self._violations_log: List[Dict[str, Any]] = []

    # ── URL safety ───────────────────────────────────────────────────

    def is_url_blocked(self, url: str) -> bool:
        """Check if a URL is blocked by domain or pattern rules.

        Returns *True* if the URL matches any blocked domain glob or
        any blocked URL regex pattern.
        """
        if not url:
            return False
        try:
            parsed = urlparse(url)
            domain = parsed.hostname or ""
        except Exception:
            return True  # Malformed URLs are blocked

        # Check domain globs
        for pattern in self._blocked_domains:
            if fnmatch.fnmatch(domain, pattern) or fnmatch.fnmatch(domain, f"*.{pattern}"):
                return True

        # Check URL regex patterns
        for regex in self._blocked_url_re:
            if regex.search(url):
                return True

        return False

    def is_sensitive_field(self, field_info: Dict[str, Any]) -> bool:
        """Check if a form field matches sensitive patterns.

        Parameters
        ----------
        field_info:
            Dictionary with field attributes (``name``, ``id``,
            ``placeholder``, ``aria-label``, ``type``).

        Returns
        -------
        bool
            *True* if the field is considered sensitive.
        """
        # HTML password fields are always sensitive
        if field_info.get("type", "").lower() == "password":
            return True

        # Check all text attributes against patterns
        attrs_to_check = [
            str(field_info.get("name", "")),
            str(field_info.get("id", "")),
            str(field_info.get("placeholder", "")),
            str(field_info.get("aria-label", "")),
            str(field_info.get("label", "")),
        ]
        combined = " ".join(attrs_to_check)
        return any(regex.search(combined) for regex in self._sensitive_re)

    # ── BrowserUse step callback ─────────────────────────────────────

    def on_step(
        self,
        step: Any = None,
        *,
        action: Optional[Dict[str, Any]] = None,
        result: Any = None,
        **kwargs: Any,
    ) -> None:
        """Main callback invoked by BrowserUse after each agent step.

        Compatible with BrowserUse's ``on_step_callback`` signature.
        Inspects the step for navigations, actions, and LLM decisions,
        applying guardrails as appropriate.

        Parameters
        ----------
        step:
            The BrowserUse ``AgentStep`` object (or similar).
        action:
            Parsed action dictionary if available separately.
        result:
            The result of the step execution.
        """
        step_record: Dict[str, Any] = {
            "step_number": len(self._steps) + 1,
            "timestamp": time.time(),
        }

        # Extract action details from the step object or action dict
        act = action or {}
        if step is not None:
            act = act or getattr(step, "action", {}) or {}
            if isinstance(act, str):
                act = {"type": act}
            elif not isinstance(act, dict):
                act = {"type": str(type(act).__name__), "raw": str(act)[:500]}

        action_type = act.get("type", act.get("action_type", "unknown"))
        step_record["action_type"] = action_type

        # ── Navigation guardrail ─────────────────────────────────────
        url = act.get("url", act.get("goto", ""))
        if not url and action_type in ("goto", "navigate", "go_to_url"):
            url = act.get("value", "")

        if url:
            step_record["url"] = url
            self._navigations.append({
                "url": url,
                "timestamp": time.time(),
                "blocked": self.is_url_blocked(url),
            })
            if self.is_url_blocked(url):
                violation = {
                    "rule": "blocked_url",
                    "message": f"Navigation to blocked URL: {url}",
                    "url": url,
                }
                self._violations_log.append(violation)
                step_record["blocked"] = True
                if self._block:
                    raise GuardrailViolation(
                        [violation],
                        message=f"EvalGuard blocked navigation to {url}",
                    )

        # ── Sensitive field guardrail ────────────────────────────────
        if self._block_sensitive and action_type in ("type", "fill", "input_text", "fill_form"):
            field_info = act.get("field", act.get("element", {}))
            if isinstance(field_info, dict) and self.is_sensitive_field(field_info):
                violation = {
                    "rule": "sensitive_field",
                    "message": f"Attempted to type into sensitive field: {field_info.get('name', field_info.get('id', 'unknown'))}",
                    "field": field_info.get("name", field_info.get("id", "unknown")),
                }
                self._violations_log.append(violation)
                step_record["blocked"] = True
                if self._block:
                    raise GuardrailViolation(
                        [violation],
                        message="EvalGuard blocked typing into a sensitive form field",
                    )

        # ── Capture action details ───────────────────────────────────
        if action_type in ("click", "click_element"):
            step_record["element"] = str(act.get("element", act.get("selector", "")))[:200]
        elif action_type in ("type", "fill", "input_text"):
            # Truncate typed text to avoid logging sensitive data
            step_record["text_length"] = len(str(act.get("text", act.get("value", ""))))
        elif action_type in ("scroll", "scroll_down", "scroll_up"):
            step_record["direction"] = act.get("direction", "down")
            step_record["amount"] = act.get("amount", act.get("pixels", ""))
        elif action_type in ("screenshot", "take_screenshot"):
            step_record["screenshot_taken"] = True
        elif action_type in ("extract", "extract_content", "get_text"):
            extracted = str(act.get("content", act.get("text", "")))
            step_record["extracted_length"] = len(extracted)
            if self._check_content and extracted:
                check = self._guard.check_output(
                    extracted[:4000],
                    metadata={"framework": "browseruse", "action": "extract"},
                )
                if check.get("violations"):
                    step_record["content_violations"] = check["violations"]
                    self._violations_log.extend(check["violations"])

        # ── LLM decision capture ─────────────────────────────────────
        llm_output = act.get("llm_output", "")
        if not llm_output and step is not None:
            llm_output = getattr(step, "llm_output", getattr(step, "thought", ""))
        if llm_output:
            step_record["llm_decision"] = str(llm_output)[:1000]

        # ── Result capture ───────────────────────────────────────────
        if result is not None:
            result_str = str(result)[:500] if not isinstance(result, str) else result[:500]
            step_record["result"] = result_str

        self._steps.append(step_record)

    # ── Session-level callbacks ──────────────────────────────────────

    def on_agent_start(
        self,
        *,
        task: str = "",
        agent: Any = None,
        **kwargs: Any,
    ) -> None:
        """Called when the BrowserUse agent begins a task.

        Checks the task description against guardrails.
        """
        self._session_id = str(uuid.uuid4())
        self._session_start = time.monotonic()
        self._steps.clear()
        self._navigations.clear()
        self._violations_log.clear()

        if task:
            check = self._guard.check_input(
                task,
                rules=self._rules,
                metadata={"framework": "browseruse"},
            )
            if not check.get("allowed", True) and self._block:
                raise GuardrailViolation(check.get("violations", []))

    def on_agent_end(
        self,
        *,
        task: str = "",
        result: Any = None,
        agent: Any = None,
        **kwargs: Any,
    ) -> None:
        """Called when the BrowserUse agent completes a task.

        Logs the full session trace to EvalGuard.
        """
        elapsed_ms = (time.monotonic() - self._session_start) * 1000
        output_text = str(result)[:4000] if result else ""

        # Optional output check
        output_violations: List[Dict[str, Any]] = []
        if output_text:
            out_check = self._guard.check_output(
                output_text,
                metadata={"framework": "browseruse"},
            )
            output_violations = out_check.get("violations", [])
            if not out_check.get("allowed", True) and self._block:
                raise GuardrailViolation(output_violations)

        self._guard.log_trace({
            "provider": "browseruse",
            "session_id": self._session_id,
            "task": task[:2000],
            "output": output_text[:2000],
            "agent_latency_ms": round(elapsed_ms, 2),
            "total_steps": len(self._steps),
            "total_navigations": len(self._navigations),
            "steps": self._steps[-50:],  # Cap to last 50 steps
            "navigations": self._navigations[-20:],  # Cap to last 20
            "violations": self._violations_log,
            "output_violations": output_violations,
        })

    # ── Trace access ─────────────────────────────────────────────────

    def get_trace(self) -> Dict[str, Any]:
        """Return the accumulated trace for the current session.

        Useful for inspection after a run completes.

        Returns
        -------
        dict
            Session trace including steps, navigations, and violations.
        """
        return {
            "session_id": self._session_id,
            "elapsed_ms": round((time.monotonic() - self._session_start) * 1000, 2),
            "total_steps": len(self._steps),
            "total_navigations": len(self._navigations),
            "steps": list(self._steps),
            "navigations": list(self._navigations),
            "violations": list(self._violations_log),
        }

    def get_navigation_history(self) -> List[Dict[str, Any]]:
        """Return the list of all page navigations."""
        return list(self._navigations)

    def get_violations(self) -> List[Dict[str, Any]]:
        """Return all guardrail violations detected during the session."""
        return list(self._violations_log)


def create_browseruse_callback(
    api_key: str,
    project_id: Optional[str] = None,
    base_url: str = "https://evalguard.ai/api",
    blocked_domains: Optional[List[str]] = None,
    block_sensitive_fields: bool = True,
    **kwargs: Any,
) -> EvalGuardBrowserCallback:
    """Convenience factory for creating a BrowserUse callback.

    Usage::

        from evalguard.browseruse import create_browseruse_callback

        callback = create_browseruse_callback(
            api_key="eg_...",
            blocked_domains=["evil.com"],
        )
        agent = Agent(task="...", llm=llm, on_step_callback=callback.on_step)

    Returns
    -------
    EvalGuardBrowserCallback
        Configured callback instance.
    """
    return EvalGuardBrowserCallback(
        api_key=api_key,
        project_id=project_id,
        base_url=base_url,
        blocked_domains=blocked_domains,
        block_sensitive_fields=block_sensitive_fields,
        **kwargs,
    )
