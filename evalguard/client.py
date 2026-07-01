"""EvalGuard Python SDK — HTTP client for the EvalGuard API."""

from __future__ import annotations

import base64
import time
import uuid
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlparse

import requests

from .types import (
    BenchmarkResult,
    ComplianceReport,
    DriftReport,
    EvalResult,
    FirewallResult,
    SecurityScanResult,
)


def _normalize_base_url(base_url: str) -> str:
    """Normalize a base URL so the same value works across the EvalGuard SDKs.

    Request paths in this client are version-prefixed (``/v1/...``), so the
    stored base must end at ``/api`` (default ``https://evalguard.ai/api`` →
    ``https://evalguard.ai/api/v1/...``). The TS and Go SDKs and the raw API,
    however, document ``…/api/v1`` as their base. Passing that value here
    previously produced a doubled ``/api/v1/v1/...`` → 404. Drop a redundant
    trailing ``/v1`` so both conventions resolve identically:

      https://evalguard.ai/api        -> https://evalguard.ai/api   (unchanged)
      https://evalguard.ai/api/v1     -> https://evalguard.ai/api   (/v1 dropped)
      https://self-host/x/api/v1/     -> https://self-host/x/api
      https://api.example.com/        -> https://api.example.com    (unchanged)
    """
    base = (base_url or "").rstrip("/")
    if base.endswith("/v1"):
        base = base[: -len("/v1")].rstrip("/")
    return base


# SDK version — single source of truth within this module. Sent on every request
# (User-Agent + x-evalguard-client-version) so the server can enforce version
# policy, and compared against the org's pinned range by check_version_policy().
# Keep in lockstep with evalguard/__init__.py __version__.
_SDK_VERSION = "2.1.0"


def _parse_semver(v: Optional[str]) -> Optional[tuple]:
    """Parse 'MAJOR.MINOR.PATCH' (ignoring any -prerelease / +build) into a
    numeric tuple. Returns None when the string isn't usable semver, so the
    caller treats it as unpinned (fail-open)."""
    if not v or not isinstance(v, str):
        return None
    core = v.strip().lstrip("v").split("+", 1)[0].split("-", 1)[0]
    try:
        nums = [int(p) for p in core.split(".")[:3]]
    except ValueError:
        return None
    while len(nums) < 3:
        nums.append(0)
    return tuple(nums[:3])


def _cmp_semver(a: tuple, b: tuple) -> int:
    """-1 if a < b, 0 if equal, 1 if a > b."""
    return (a > b) - (a < b)


class EvalGuardError(Exception):
    """Base exception for EvalGuard API errors."""

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        body: Any = None,
        code: str | None = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.body = body
        # The server's stable machine-readable error code (from the
        # { error: { code } } envelope), when available.
        self.code = code


class EvalGuardClient:
    """Client for the EvalGuard REST API.

    Example::

        from evalguard import EvalGuardClient

        client = EvalGuardClient(api_key="eg_live_...")
        result = client.run_eval({
            "name": "Greeting eval",          # required by POST /v1/evals
            "model": "gpt-4o",
            "prompt": "Answer: {{input}}",
            "cases": [{"input": "hello", "expectedOutput": "hello"}],
            "scorers": ["exact-match"],
        })
        print(result)
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://evalguard.ai/api",
        timeout: float = 120.0,
    ) -> None:
        self.api_key = api_key
        self.base_url = _normalize_base_url(base_url)
        self.timeout = timeout
        self._subject: Optional[Dict[str, str]] = None
        # Lazily-resolved default project id. Populated on first call to a
        # project-scoped method that wasn't given an explicit projectId, then
        # reused so repeated calls don't re-hit /project/current.
        self._resolved_project_id: Optional[str] = None
        # Lazily-resolved default org id (same /project/current call returns
        # both projectId and orgId). Cached so org-scoped methods that auto-
        # resolve (classify_intent, set_gateway_routing_config) don't re-fetch.
        self._resolved_org_id: Optional[str] = None

        # Enforce HTTPS for non-local URLs
        parsed = urlparse(self.base_url)
        is_local = parsed.hostname in ('localhost', '127.0.0.1')
        if parsed.scheme != 'https' and not is_local:
            raise ValueError(
                "EvalGuard: base_url must use HTTPS. "
                "Only localhost/127.0.0.1 may use HTTP."
            )
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "User-Agent": f"evalguard-sdk-python/{_SDK_VERSION}",
                # Sent on every request so an org that pins allowed client
                # versions can enforce its policy on this SDK (deep-audit
                # 2026-06-21 — the TS SDK already sends this; Python/Go/Java
                # were silently exempt). Single-sourced from _SDK_VERSION so it
                # can never drift from the User-Agent above.
                "x-evalguard-client-version": _SDK_VERSION,
            }
        )

    # ── Helpers ──────────────────────────────────────────────────────────

    def with_subject(
        self,
        email: Optional[str] = None,
        id: Optional[str] = None,
        purpose: Optional[str] = None,
    ) -> "EvalGuardClient":
        """Return a new client bound to a subject for consent enforcement.

        The gateway proxy reads ``x-evalguard-subject-email`` /
        ``x-evalguard-subject-id`` and ``x-evalguard-purpose`` to look up
        consent. If consent is revoked or denied, the gateway returns
        HTTP 451 BEFORE forwarding to the upstream LLM provider.

        Either ``email`` or ``id`` is required. ``purpose`` defaults to
        ``model_inference`` on the server side.

        Returns a new client so a single shared client can fan out
        per-request scoped clients without mutation::

            client = EvalGuardClient(api_key="eg_live_...")
            user_client = client.with_subject(email=user.email, purpose="support_chat")
            user_client.gateway_proxy(...)  # 451 if user revoked consent
        """
        if not email and not id:
            raise ValueError("with_subject: at least one of email or id is required")
        new = EvalGuardClient(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
        )
        subject: Dict[str, str] = {}
        if email:
            subject["x-evalguard-subject-email"] = email
        if id:
            subject["x-evalguard-subject-id"] = id
        if purpose:
            subject["x-evalguard-purpose"] = purpose
        new._subject = subject
        return new

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        kwargs.setdefault("timeout", self.timeout)
        # Inject consent headers per-request — using session.headers would
        # leak them across all clients sharing a Session, but since
        # with_subject() returns a fresh client (and therefore a fresh
        # Session), per-request injection is correctness AND a paranoia
        # double-up that doesn't cost anything.
        if self._subject:
            headers = kwargs.pop("headers", {}) or {}
            headers = {**headers, **self._subject}
            kwargs["headers"] = headers
        # Generate ONE Idempotency-Key per logical call (not per attempt) for
        # non-idempotent writes, so the retry loop below reuses it across every
        # retry. A transient 5xx/network blip then dedups server-side
        # (idempotency.py keys on the `idempotency-key` header) instead of
        # creating duplicate scans/runs and double-billing the customer. GET and
        # DELETE are naturally idempotent and need no key.
        if method.upper() in ("POST", "PUT", "PATCH"):
            headers = kwargs.pop("headers", {}) or {}
            headers = {"Idempotency-Key": str(uuid.uuid4()), **headers}
            kwargs["headers"] = headers
        max_retries = 3
        last_error = None

        for attempt in range(max_retries + 1):
            try:
                response = self.session.request(method, self._url(path), **kwargs)

                if response.status_code == 429:
                    retry_after = int(response.headers.get("Retry-After", 0))
                    delay = retry_after if retry_after > 0 else min(2 ** attempt, 60)
                    if attempt < max_retries:
                        time.sleep(delay)
                        continue

                if response.status_code >= 500 and attempt < max_retries:
                    time.sleep(2 ** attempt)
                    continue

                if not response.ok:
                    raise EvalGuardError(
                        f"API error {response.status_code}: {response.text[:500]}",
                        status_code=response.status_code,
                        body=response.text[:500],
                    )
                if response.status_code == 204:
                    return None
                return response.json()
            except EvalGuardError:
                raise
            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    time.sleep(2 ** attempt)
                    continue
                raise EvalGuardError(f"Request failed: {e}")

        raise EvalGuardError(f"Request failed after {max_retries} retries: {last_error}")

    def _get(self, path: str, params: Dict[str, Any] | None = None) -> Any:
        return self._request("GET", path, params=params)

    def _post(self, path: str, json: Any = None) -> Any:
        return self._request("POST", path, json=json)

    def _patch(self, path: str, json: Any = None) -> Any:
        return self._request("PATCH", path, json=json)

    def _put(self, path: str, json: Any = None) -> Any:
        return self._request("PUT", path, json=json)

    def _delete(self, path: str) -> Any:
        return self._request("DELETE", path)

    def _request_text(self, method: str, path: str, **kwargs: Any) -> str:
        """Like ``_request`` (consent + idempotency headers, retry on 429/5xx)
        but for endpoints that return a TEXT body (e.g. JSONL / XML exports)
        instead of the JSON ``{success, data}`` envelope. On a non-ok response
        it parses the standard ``{error: {code, message}}`` envelope into an
        ``EvalGuardError`` (carrying the server's stable ``code``); on success
        it returns ``response.text``."""
        kwargs.setdefault("timeout", self.timeout)
        if self._subject:
            headers = kwargs.pop("headers", {}) or {}
            kwargs["headers"] = {**headers, **self._subject}
        if method.upper() in ("POST", "PUT", "PATCH"):
            headers = kwargs.pop("headers", {}) or {}
            kwargs["headers"] = {"Idempotency-Key": str(uuid.uuid4()), **headers}

        max_retries = 3
        last_error: Exception | None = None
        for attempt in range(max_retries + 1):
            try:
                response = self.session.request(method, self._url(path), **kwargs)

                if response.status_code == 429:
                    retry_after = int(response.headers.get("Retry-After", 0))
                    delay = retry_after if retry_after > 0 else min(2 ** attempt, 60)
                    if attempt < max_retries:
                        time.sleep(delay)
                        continue

                if response.status_code >= 500 and attempt < max_retries:
                    time.sleep(2 ** attempt)
                    continue

                if not response.ok:
                    code: str | None = None
                    message = response.text[:500]
                    try:
                        err = response.json()
                        if isinstance(err, dict) and isinstance(err.get("error"), dict):
                            code = err["error"].get("code")
                            message = err["error"].get("message", message)
                    except Exception:
                        pass
                    raise EvalGuardError(
                        f"API error {response.status_code}: {message}",
                        status_code=response.status_code,
                        body=response.text[:500],
                        code=code,
                    )
                return response.text
            except EvalGuardError:
                raise
            except Exception as e:  # network / decode
                last_error = e
                if attempt < max_retries:
                    time.sleep(2 ** attempt)
                    continue
                raise EvalGuardError(f"Request failed: {e}")

        raise EvalGuardError(f"Request failed after {max_retries} retries: {last_error}")

    def _get_text(self, path: str, params: Dict[str, Any] | None = None) -> str:
        return self._request_text("GET", path, params=params)

    @staticmethod
    def _unwrap(resp: Any) -> Any:
        """Return the inner ``data`` of a standard ``{success, data}`` API
        envelope. Tolerates a non-enveloped body (older servers / mocks) by
        returning it unchanged."""
        if isinstance(resp, dict) and "data" in resp:
            return resp["data"]
        return resp

    def check_version_policy(self) -> Dict[str, Any]:
        """Check this SDK version against the org's pinned version policy.

        Parity with the TS SDK's ``checkVersionPolicy()``. Returns
        ``{allowed, requiredMinimumVersion, requiredMaximumVersion, reason?}``.
        A pure READ — never mutates. Best-effort + FAIL-OPEN: a 3s timeout, NO
        retry, and any network/endpoint error yields ``allowed=True`` (a
        transient policy-read blip must not brick the customer's SDK fleet — the
        server also sees the version header on every request and can enforce
        there). Returns ``allowed=True`` (unpinned) when the org sets no bounds.
        """
        policy: Dict[str, Any] = {}
        try:
            resp = self.session.get(
                self._url("/v1/client/policy"),
                params={"version": _SDK_VERSION},
                timeout=3,
            )
            body = resp.json()
            # { success, data } envelope — unwrap if present.
            if isinstance(body, dict) and isinstance(body.get("data"), dict):
                policy = body["data"]
            elif isinstance(body, dict):
                policy = body
        except Exception:
            return {
                "allowed": True,
                "requiredMinimumVersion": None,
                "requiredMaximumVersion": None,
            }

        min_v = policy.get("requiredMinimumVersion")
        max_v = policy.get("requiredMaximumVersion")
        result: Dict[str, Any] = {
            "allowed": True,
            "requiredMinimumVersion": min_v,
            "requiredMaximumVersion": max_v,
        }
        if not min_v and not max_v:
            return result  # unpinned

        ver = _parse_semver(_SDK_VERSION)
        min_t = _parse_semver(min_v)
        max_t = _parse_semver(max_v)
        if ver and min_t and _cmp_semver(ver, min_t) < 0:
            result["allowed"] = False
            result["reason"] = (
                f"evalguard-sdk-python {_SDK_VERSION} is below the minimum "
                f"version ({min_v}) required by this organization. Upgrade to continue."
            )
        elif ver and max_t and _cmp_semver(ver, max_t) > 0:
            result["allowed"] = False
            result["reason"] = (
                f"evalguard-sdk-python {_SDK_VERSION} is above the maximum "
                f"version ({max_v}) allowed by this organization. Downgrade to a supported release."
            )
        return result

    def assert_version_allowed(self) -> None:
        """Like :meth:`check_version_policy` but RAISES ``EvalGuardError`` when
        this SDK version is outside the org's pinned range. Call once at startup
        to hard-stop an out-of-policy client before it issues real requests."""
        v = self.check_version_policy()
        if not v.get("allowed"):
            raise EvalGuardError(
                v.get("reason") or "EvalGuard client version not allowed by org policy"
            )

    def _resolve_project_id(self) -> str:
        """Resolve (and cache) this org's default project id.

        Calls ``GET /v1/project/current`` ONCE per client instance — the
        endpoint returns the raw ``{"projectId": ..., "orgId": ...}`` body
        (not the ``{success, data}`` envelope) and auto-creates a default
        project for a fresh org. The resolved id is cached on the instance so
        subsequent project-scoped calls don't re-fetch.
        """
        if self._resolved_project_id:
            return self._resolved_project_id
        resp = self._get("/v1/project/current")
        project_id = resp.get("projectId") if isinstance(resp, dict) else None
        if not project_id:
            raise EvalGuardError(
                "Could not resolve a default project; pass projectId explicitly."
            )
        self._resolved_project_id = project_id
        # The same response carries orgId — opportunistically cache it so a
        # later org-scoped call (classify_intent / set_gateway_routing_config)
        # can avoid a second round trip.
        if isinstance(resp, dict) and resp.get("orgId") and not self._resolved_org_id:
            self._resolved_org_id = resp["orgId"]
        return project_id

    def _resolve_org_id(self) -> str:
        """Resolve (and cache) this caller's default org id.

        Uses the same ``GET /v1/project/current`` call as
        :meth:`_resolve_project_id` (the body is the raw
        ``{"projectId": ..., "orgId": ...}`` shape, not the ``{success, data}``
        envelope). Cached on the instance so repeated org-scoped calls don't
        re-fetch. Parity with the TS SDK's auto-resolution in
        ``classifyIntent`` / ``setGatewayRoutingConfig``.
        """
        if self._resolved_org_id:
            return self._resolved_org_id
        resp = self._get("/v1/project/current")
        org_id = resp.get("orgId") if isinstance(resp, dict) else None
        if not org_id:
            raise EvalGuardError(
                "Could not resolve a default org; pass org_id explicitly."
            )
        self._resolved_org_id = org_id
        if isinstance(resp, dict) and resp.get("projectId") and not self._resolved_project_id:
            self._resolved_project_id = resp["projectId"]
        return org_id

    # ── Eval endpoints ───────────────────────────────────────────────────

    def run_eval(
        self, config: Dict[str, Any], name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Run an evaluation with the given config and return results.

        ``name`` is REQUIRED by ``POST /v1/evals`` (createEvalSchema, min 1 char) —
        the server 400s ("Name is required") without it. Provide it either as the
        ``name`` argument or inside ``config["name"]``; a ``config["name"]`` value
        wins when both are given. If neither is set, a ``ValueError`` is raised
        before any request (fail-fast, no wasted round trip).

        If ``config`` omits ``projectId``, the SDK resolves (and caches) the
        org's default project via ``/v1/project/current``. An explicit
        ``projectId`` in ``config`` always wins and skips that lookup.

        Example::

            client.run_eval(
                {
                    "model": "gpt-4o",
                    "prompt": "Answer: {{input}}",
                    "cases": [{"input": "hello", "expectedOutput": "hello"}],
                    "scorers": ["exact-match"],
                },
                name="Greeting eval",
            )
        """
        if not isinstance(config, dict):
            raise ValueError("run_eval: config must be a dict")
        config = dict(config)
        # name: arg fills it in only when config doesn't already carry one.
        if not config.get("name") and name:
            config["name"] = name
        if not config.get("name"):
            raise ValueError(
                "run_eval: a non-empty `name` is required (pass name=... or "
                'config["name"]) — POST /v1/evals rejects a missing name with 400.'
            )
        if not config.get("projectId"):
            config["projectId"] = self._resolve_project_id()
        # #36: /v1/evals replies with the standard { success, data } envelope —
        # unwrap so callers get the run object directly (matching check_firewall
        # / get_compliance_report / detect_drift). Tolerates a non-enveloped
        # body (older servers / mocks).
        return self._unwrap(self._post("/v1/evals", json=config))

    def get_eval(self, run_id: str) -> Dict[str, Any]:
        """Fetch a specific eval run by ID."""
        return self._unwrap(self._get(f"/v1/evals/{run_id}"))

    def list_evals(self, project_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """List eval runs for a project.

        If ``project_id`` is omitted, the SDK resolves (and caches) the org's
        default project via ``/v1/project/current``. An explicit ``project_id``
        always wins and skips that lookup.
        """
        if not project_id:
            project_id = self._resolve_project_id()
        return self._unwrap(self._get("/v1/evals", params={"projectId": project_id}))

    # ── Security scan endpoints ──────────────────────────────────────────

    def run_scan(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run a security scan (red-team) against a model.

        Security scans are created at ``POST /v1/security`` (there is no
        ``/v1/scans`` route). The request body must match createSecurityScanSchema::

            {
                "projectId": "<uuid>",          # required (auto-resolved if omitted)
                "model": "gpt-4o",              # required, 1..100 chars
                "prompt": "You are a helpful assistant",  # required, 1..100000 chars
                "attackTypes": ["prompt-injection", "jailbreak"],  # required, 1..50
            }

        A prior version of this method sent ``{model, provider, categories}``,
        which the route rejected with 400 VALIDATION_ERROR (``provider`` /
        ``categories`` are not schema fields and ``prompt`` / ``attackTypes`` were
        missing). The payload is now normalized to the real contract:

          - ``projectId`` is auto-resolved (and cached) via ``/v1/project/current``
            when ``config`` omits it; an explicit ``projectId`` always wins.
          - ``attackTypes`` is REQUIRED (>= 1). For back-compat a legacy
            ``categories`` list is accepted as an alias and copied into
            ``attackTypes`` when the latter is absent.

        A missing ``model`` / ``prompt`` / ``attackTypes`` raises ``ValueError``
        before any request (fail-fast, no wasted round trip).

        ``security_scan()`` is an equivalent alias.

        Example::

            client.run_scan({
                "model": "gpt-4o",
                "prompt": "You are a helpful assistant",
                "attackTypes": ["prompt-injection", "jailbreak"],
            })
        """
        if not isinstance(config, dict):
            raise ValueError("run_scan: config must be a dict")
        config = dict(config)
        # Back-compat: accept a legacy `categories` alias for `attackTypes`.
        if not config.get("attackTypes") and config.get("categories"):
            config["attackTypes"] = config["categories"]
        # `provider`/`categories` are not part of createSecurityScanSchema — drop
        # them so the (strict) server validator doesn't choke and so the request
        # body matches the documented contract exactly.
        config.pop("provider", None)
        config.pop("categories", None)
        if not config.get("model"):
            raise ValueError("run_scan: `model` is required")
        if not config.get("prompt"):
            raise ValueError("run_scan: `prompt` is required")
        attack_types = config.get("attackTypes")
        if not isinstance(attack_types, list) or len(attack_types) == 0:
            raise ValueError(
                "run_scan: `attackTypes` is required and must be a non-empty list "
                '(e.g. ["prompt-injection"]).'
            )
        if not config.get("projectId"):
            config["projectId"] = self._resolve_project_id()
        return self._post("/v1/security", json=config)

    def get_scan(self, scan_id: str) -> Dict[str, Any]:
        """Fetch a specific security scan (with findings) by ID.

        Backed by ``GET /v1/security/{scanId}`` — returns the scan row merged
        with its findings, computed ``severityCounts``, ``score``, ``totalTests``
        and ``passedCount``. ``scan_id`` is the id returned by ``run_scan`` /
        ``security_scan``.
        """
        if not scan_id:
            raise ValueError("get_scan: scan_id is required")
        return self._get(f"/v1/security/{scan_id}")

    # ── Scorers & plugins ────────────────────────────────────────────────

    def list_scorers(self) -> List[Dict[str, Any]]:
        """List all available evaluation scorers."""
        return self._get("/v1/scorers")

    def list_plugins(self) -> List[Dict[str, Any]]:
        """List all available security plugins."""
        return self._get("/v1/plugins")

    # ── Firewall ─────────────────────────────────────────────────────────

    def check_firewall(
        self,
        input_text: str,
        rules: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Check input text against the firewall.

        Parameters
        ----------
        rules:
            Optional list of rule-name strings (the API expects ``string[]``,
            e.g. ``["pii", "prompt-injection"]``), not rule objects.

        Returns
        -------
        dict
            The firewall result ``{blocked, score, category, subcategory,
            latencyMs, hits}`` — the ``{success, data}`` envelope is unwrapped.
        """
        payload: Dict[str, Any] = {"input": input_text}
        if rules is not None:
            payload["rules"] = rules
        resp = self._post("/v1/firewall/check", json=payload)
        # All v1 endpoints reply with the standard { success, data } envelope;
        # return the inner result so callers get `result["blocked"]` directly.
        # Tolerate a non-enveloped body (older servers / mocks).
        if isinstance(resp, dict) and "data" in resp:
            return resp["data"]
        return resp

    # ── Benchmarks ───────────────────────────────────────────────────────

    def submit_benchmark(
        self,
        benchmark: str,
        model: str,
        total_score: float,
        scores: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Submit a completed benchmark run to the leaderboard.

        ``POST /v1/benchmarks`` records a benchmark *result* — the benchmark
        name (e.g. ``"mmlu"``), the model, its total score, and optional
        per-category ``scores``. It does not execute suites.

        Contract verified against the live API 2026-06-17: the endpoint
        requires ``{benchmark, model, totalScore}`` (``scores`` optional).
        """
        body: Dict[str, Any] = {
            "benchmark": benchmark,
            "model": model,
            "totalScore": total_score,
        }
        if scores is not None:
            body["scores"] = scores
        return self._post("/v1/benchmarks", json=body)

    def run_benchmarks(
        self,
        suites: List[str],
        model: str,
    ) -> Dict[str, Any]:
        """Deprecated. The old ``{suites, model}`` payload was rejected by the
        API (400) — ``POST /v1/benchmarks`` records a result, not a suite run.
        Use :meth:`submit_benchmark` instead.
        """
        raise EvalGuardError(
            "run_benchmarks(suites, model) is not supported by the API — it "
            "records a benchmark result. Use submit_benchmark(benchmark, model, "
            "total_score, scores=...) instead."
        )

    # ── Export ────────────────────────────────────────────────────────────

    def export_dpo(self, run_id: str, project_id: str) -> str:
        """Export eval results as DPO training data (JSONL).

        Repointed to the real ``/v1/exports`` contract — the old
        ``/v1/evals/{id}/export/dpo`` path 404'd (audit 2026-06-14 #7).
        ``project_id`` is required by the export API. Goes through
        ``_get_text`` for retry + standard error-envelope parsing (#87).
        """
        return self._get_text(
            "/v1/exports",
            params={"runId": run_id, "format": "dpo", "projectId": project_id},
        )

    def export_burp(self, scan_id: str, project_id: str) -> str:
        """Export security scan results as Burp Suite XML.

        Repointed to the real ``/v1/exports`` contract — the old
        ``/v1/scans/{id}/export/burp`` path 404'd (audit 2026-06-14 #7).
        ``project_id`` is required by the export API. Goes through
        ``_get_text`` for retry + standard error-envelope parsing (#87).
        """
        return self._get_text(
            "/v1/exports",
            params={"runId": scan_id, "format": "burp", "projectId": project_id},
        )

    # ── Compliance ───────────────────────────────────────────────────────

    def get_compliance_report(
        self, scan_id: str, framework: str
    ) -> Dict[str, Any]:
        """Map a security scan's findings onto a compliance framework.

        Backed by ``GET /v1/security/{scanId}/compliance`` (audit 2026-06-14 #7).
        Returns the inner report (the ``{success, data}`` envelope is unwrapped).
        """
        resp = self._get(
            f"/v1/security/{scan_id}/compliance",
            params={"framework": framework},
        )
        if isinstance(resp, dict) and "data" in resp:
            return resp["data"]
        return resp

    # ── Drift detection ──────────────────────────────────────────────────

    def detect_drift(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Detect performance drift between a baseline and a current eval run.

        Backed by ``POST /v1/monitoring/drift/detect`` (audit 2026-06-14 #7).
        ``config`` must include ``baselineRunId`` and ``currentRunId``.
        Returns the inner result (the ``{success, data}`` envelope is unwrapped).
        """
        resp = self._post("/v1/monitoring/drift/detect", json=config)
        if isinstance(resp, dict) and "data" in resp:
            return resp["data"]
        return resp

    # ── Guardrails ───────────────────────────────────────────────────────

    def generate_guardrails(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Auto-generate firewall rules from scan findings."""
        return self._post("/v1/guardrails/generate", json=config)

    # ── Smart Routing ───────────────────────────────────────────────────

    def smart_route(self, test_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Route test cases to optimal model tiers by complexity."""
        return self._post("/v1/smart-routing/test-cases", json={"testCases": test_cases})

    # ── Autopilot ───────────────────────────────────────────────────────

    def autopilot(self, description: str, depth: str, project_id: str, compliance_frameworks: Optional[List[str]] = None) -> Dict[str, Any]:
        """Launch automated audit pipeline."""
        return self._post("/v1/autopilot", json={"description": description, "depth": depth, "projectId": project_id, "complianceFrameworks": compliance_frameworks})

    def get_autopilot_config(self) -> Dict[str, Any]:
        """Get autopilot depth configurations."""
        return self._get("/v1/autopilot")

    # ── Pipelines ───────────────────────────────────────────────────────

    def create_pipeline(self, template_id: str, project_id: str) -> Dict[str, Any]:
        """Create eval pipeline from template."""
        return self._post("/v1/pipelines", json={"templateId": template_id, "projectId": project_id})

    def list_pipelines(self) -> List[Dict[str, Any]]:
        """List pipeline templates."""
        return self._get("/v1/pipelines")

    # ── Leaderboard ─────────────────────────────────────────────────────

    def get_leaderboard(self, category: str = "overall") -> Dict[str, Any]:
        """Get model safety/performance leaderboard."""
        return self._get("/v1/leaderboard", params={"category": category})

    # ── Cost / FinOps ───────────────────────────────────────────────────

    def get_cost(self, project_id: str, period: str = "30d") -> Dict[str, Any]:
        """Get cost analytics."""
        return self._get("/v1/cost", params={"projectId": project_id, "period": period})

    def get_cost_savings(self, project_id: str, period: str = "30d") -> Dict[str, Any]:
        """Get ROI / cost savings report."""
        return self._get("/v1/cost/savings", params={"projectId": project_id, "period": period})

    def get_cost_forecast(self, project_id: str) -> Dict[str, Any]:
        """Get cost forecast."""
        return self._get("/v1/cost/forecast", params={"projectId": project_id})

    # ── Security (extended) ─────────────────────────────────────────────

    def get_security_effectiveness(self, project_id: str) -> Dict[str, Any]:
        """Get attack effectiveness analytics."""
        return self._get("/v1/security/effectiveness", params={"projectId": project_id})

    def get_security_report(self, assessment_id: str) -> Dict[str, Any]:
        """Get a previously generated security assessment report.

        Backed by ``GET /v1/security/report?assessmentId=...`` — the route keys
        the report store on ``assessmentId`` (a prior ``scanId`` query param was
        silently ignored → 404).
        """
        return self._get("/v1/security/report", params={"assessmentId": assessment_id})

    # ── Support ─────────────────────────────────────────────────────────

    def submit_ticket(self, ticket_type: str, subject: str, description: str, priority: str = "medium", metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """Submit a support ticket."""
        return self._post("/v1/support", json={"type": ticket_type, "subject": subject, "description": description, "priority": priority, "metadata": metadata or {}})

    def list_tickets(self, status: Optional[str] = None) -> Dict[str, Any]:
        """List user support tickets."""
        params = {"status": status} if status else {}
        return self._get("/v1/support", params=params)

    # ── Traces ──────────────────────────────────────────────────────────

    def list_traces(self, project_id: str) -> List[Dict[str, Any]]:
        """List traces for a project."""
        return self._get("/v1/traces", params={"projectId": project_id})

    def search_traces(self, project_id: str, query: str) -> List[Dict[str, Any]]:
        """Search traces."""
        return self._get("/v1/traces/search", params={"projectId": project_id, "q": query})

    def ingest_otlp(self, resource_spans: List[Dict]) -> Dict[str, Any]:
        """Ingest OTLP traces."""
        return self._post("/v1/ingest/otlp/traces", json={"resourceSpans": resource_spans})

    # ── Monitoring ──────────────────────────────────────────────────────

    def get_monitoring_analytics(self, project_id: str) -> Dict[str, Any]:
        """Get monitoring analytics."""
        return self._get("/v1/monitoring/analytics", params={"projectId": project_id})

    def get_monitoring_alerts(self, project_id: str) -> Dict[str, Any]:
        """Get monitoring alerts."""
        return self._get("/v1/monitoring/alerts", params={"projectId": project_id})

    # ── Compliance (extended) ───────────────────────────────────────────

    def check_compliance(self, project_id: str, framework: Optional[str] = None) -> Dict[str, Any]:
        """Run compliance check."""
        params = {"projectId": project_id}
        if framework: params["framework"] = framework
        return self._get("/v1/compliance/check", params=params)

    def get_compliance_gaps(self, framework: str) -> Dict[str, Any]:
        """Get the gap analysis for a compliance framework.

        Backed by ``GET /v1/compliance/gaps?framework=...`` — ``framework`` is
        required (e.g. ``india-dpdp-act``, ``hipaa``, ``fedramp``, ``pci-dss``);
        the route 400s without it. (A prior ``projectId`` query param was
        ignored.)
        """
        if not framework:
            raise ValueError("get_compliance_gaps: framework is required")
        return self._get("/v1/compliance/gaps", params={"framework": framework})

    # ── Prompts ─────────────────────────────────────────────────────────

    def create_prompt(self, project_id: str, name: str, content: str, model: str = "gpt-4o", tags: Optional[List[str]] = None) -> Dict[str, Any]:
        """Create a prompt template."""
        return self._post("/v1/prompts", json={"projectId": project_id, "name": name, "content": content, "model": model, "tags": tags or []})

    def list_prompts(self, project_id: str) -> List[Dict[str, Any]]:
        """List prompts."""
        return self._get("/v1/prompts", params={"projectId": project_id})

    # ── Datasets ────────────────────────────────────────────────────────

    def create_dataset(self, project_id: str, name: str, cases: Optional[List[Dict]] = None, description: str = "") -> Dict[str, Any]:
        """Create a dataset."""
        return self._post("/v1/datasets", json={"projectId": project_id, "name": name, "cases": cases or [], "description": description})

    def list_datasets(self, project_id: str) -> List[Dict[str, Any]]:
        """List datasets."""
        return self._get("/v1/datasets", params={"projectId": project_id})

    # ── Dataset versioning (Phase 6b, 2026-05-22) ───────────────────────
    #
    # Immutable per-dataset snapshots for reproducible evals. Each
    # snapshot freezes the full `dataset_cases` set at a point in time;
    # restore replays a snapshot's cases back into the live table after
    # auto-snapshotting the pre-restore state.

    def list_dataset_versions(self, dataset_id: str) -> Dict[str, Any]:
        """List immutable snapshots for a dataset (newest first)."""
        return self._get(f"/v1/datasets/{dataset_id}/versions")

    def snapshot_dataset(
        self,
        dataset_id: str,
        description: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Snapshot the dataset's current cases into a new version.

        Returns ``{"unchanged": True, "version": {...}}`` if the content
        hash matches the latest version (no new row written).
        """
        body: Dict[str, Any] = {}
        if description is not None:
            body["description"] = description
        return self._post(f"/v1/datasets/{dataset_id}/versions", json=body)

    def get_dataset_version(self, dataset_id: str, version_id: str) -> Dict[str, Any]:
        """Fetch a single immutable snapshot including its inline cases."""
        return self._get(f"/v1/datasets/{dataset_id}/versions/{version_id}")

    def restore_dataset_version(self, dataset_id: str, version_id: str) -> Dict[str, Any]:
        """Restore a dataset to a frozen version.

        Auto-snapshots the pre-restore state first so the operation is
        reversible. Returns ``{"restoredFromVersion": int, "caseCount":
        int, "preRestoreVersionNum": Optional[int]}``.
        """
        return self._post(f"/v1/datasets/{dataset_id}/versions/{version_id}/restore")

    def diff_dataset_versions(
        self,
        dataset_id: str,
        from_version_id: str,
        to_version_id: str,
    ) -> Dict[str, Any]:
        """Diff two snapshots — returns added/removed/modified counts + samples."""
        return self._get(
            f"/v1/datasets/{dataset_id}/versions/{from_version_id}/diff",
            params={"to": to_version_id},
        )

    # ── Evaluator Hub (versioned, reusable evaluator registry) ──────────
    #
    # Arize-parity registry: one row per (project, name, version), content-hash
    # deduped. Mirrors the TS SDK + the `evalguard evaluators` CLI. The JSON
    # shape is documented in the OpenAPI spec.

    def list_evaluators(self, project_id: str, name: Optional[str] = None) -> Any:
        """List evaluator versions (newest-first). Pass ``name`` for one evaluator's history."""
        if not project_id:
            raise ValueError("project_id is required")
        params: Dict[str, Any] = {"projectId": project_id}
        if name is not None:
            params["name"] = name
        return self._get("/v1/evaluators", params=params)

    def create_evaluator(
        self,
        project_id: str,
        name: str,
        definition: Dict[str, Any],
        notes: Optional[str] = None,
        activate: bool = True,
    ) -> Dict[str, Any]:
        """Create a new evaluator version (content-hash deduped against the latest).

        ``definition`` is ``{"kind": "llm-judge"|"code"|"heuristic"|"composite",
        "config": {...}, "threshold": float}``.
        """
        body: Dict[str, Any] = {
            "projectId": project_id,
            "name": name,
            "definition": definition,
            "activate": activate,
        }
        if notes is not None:
            body["notes"] = notes
        return self._post("/v1/evaluators", json=body)

    def diff_evaluator_versions(
        self,
        project_id: str,
        name: str,
        from_version: int,
        to_version: int,
    ) -> Dict[str, Any]:
        """Field-level diff between two versions of a named evaluator."""
        return self._post(
            "/v1/evaluators/diff",
            json={
                "projectId": project_id,
                "name": name,
                "fromVersion": from_version,
                "toVersion": to_version,
            },
        )

    # ── Scorer calibration (CLHF — continuous learning from human feedback) ──

    def calibrate_scorer(
        self,
        pairs: Optional[List[Dict[str, bool]]] = None,
        scored: Optional[List[Dict[str, Any]]] = None,
        project_id: Optional[str] = None,
        scorer_id: Optional[str] = None,
        current_threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Evaluator/human agreement (chance-corrected Cohen's kappa) + best-threshold suggestion.

        Provide ``pairs`` (``[{"human": bool, "machine": bool}]``) and/or
        ``scored`` (``[{"humanPass": bool, "machineScore": float}]``).
        """
        if not pairs and not scored:
            raise ValueError("calibrate_scorer: provide at least one of `pairs` or `scored`")
        body: Dict[str, Any] = {}
        if pairs is not None:
            body["pairs"] = pairs
        if scored is not None:
            body["scored"] = scored
        if project_id is not None:
            body["projectId"] = project_id
        if scorer_id is not None:
            body["scorerId"] = scorer_id
        if current_threshold is not None:
            body["currentThreshold"] = current_threshold
        return self._post("/v1/scorers/calibrate", json=body)

    # ── NL Pipeline ─────────────────────────────────────────────────────

    def ask(self, question: str, project_id: Optional[str] = None) -> Dict[str, Any]:
        """Ask the AI copilot."""
        return self._post("/v1/ask", json={"question": question, "projectId": project_id})

    def generate_eval_suite(self, description: str, project_id: Optional[str] = None) -> Dict[str, Any]:
        """Generate eval test suite from description."""
        return self._post("/v1/generate-eval-suite", json={"description": description, "projectId": project_id})

    # ── AI SBOM ─────────────────────────────────────────────────────────

    def get_ai_sbom(self, project_id: str) -> Dict[str, Any]:
        """Get AI System Bill of Materials."""
        return self._get("/v1/ai-sbom", params={"projectId": project_id})

    # ── Threat Intelligence ─────────────────────────────────────────────

    def get_threat_intelligence(self, project_id: str) -> Dict[str, Any]:
        """Get threat intelligence data."""
        return self._get("/v1/threat-intelligence", params={"projectId": project_id})

    # ── Audit Logs ──────────────────────────────────────────────────────

    def get_audit_logs(self, org_id: str) -> List[Dict[str, Any]]:
        """Get audit logs."""
        return self._get("/v1/audit-logs", params={"orgId": org_id})

    # ── Notifications ───────────────────────────────────────────────────

    def list_notifications(self) -> List[Dict[str, Any]]:
        """List notifications."""
        return self._get("/v1/notifications")

    # ── Templates ───────────────────────────────────────────────────────

    def list_templates(self) -> List[Dict[str, Any]]:
        """List eval templates."""
        return self._get("/v1/templates")

    # ── Marketplace ─────────────────────────────────────────────────────

    def get_marketplace(self) -> Dict[str, Any]:
        """Get marketplace."""
        return self._get("/v1/marketplace")

    # ── Missing methods (parity with JS SDK) ───────────────────────────

    def get_eval_run(self, run_id: str) -> Dict[str, Any]:
        """Get a specific eval run by ID."""
        return self._get(f"/v1/evals/{run_id}")

    def get_trace(self, trace_id: str) -> Dict[str, Any]:
        """Get a specific trace by ID."""
        return self._get(f"/v1/traces/{trace_id}")

    def trace(self, project_id: str, session_id: str, steps: List[Dict] = None) -> Dict[str, Any]:
        """Create a trace."""
        return self._post("/v1/traces", json={"projectId": project_id, "sessionId": session_id, "steps": steps or []})

    def security_scan(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Start a security scan (canonical alias for :meth:`run_scan`).

        Shares run_scan's normalized contract: the body must satisfy
        createSecurityScanSchema ``{projectId, model, prompt, attackTypes>=1}``.
        ``projectId`` is auto-resolved via ``/v1/project/current`` when omitted;
        a legacy ``categories`` list is accepted as an ``attackTypes`` alias and
        non-schema keys (``provider``/``categories``) are dropped. Delegates to
        run_scan so both methods enforce the exact same payload (single source
        of truth — previously this posted ``config`` raw and could 400).
        """
        return self.run_scan(config)

    def create_annotation(self, project_id: str, log_id: str, label: str, score: float = None, notes: str = None) -> Dict[str, Any]:
        """Create an annotation on a log entry."""
        body: Dict[str, Any] = {"projectId": project_id, "logId": log_id, "label": label}
        if score is not None:
            body["score"] = score
        if notes:
            body["notes"] = notes
        return self._post("/v1/annotations", json=body)

    def list_annotations(self, project_id: str) -> List[Dict[str, Any]]:
        """List annotations for a project."""
        return self._get(f"/v1/annotations?projectId={project_id}")

    def list_eval_schedules(self, project_id: str) -> List[Dict[str, Any]]:
        """List eval schedules for a project."""
        return self._get(f"/v1/eval-schedules?projectId={project_id}")

    def list_incidents(self, project_id: str) -> List[Dict[str, Any]]:
        """List incidents for a project."""
        return self._get(f"/v1/incidents?projectId={project_id}")

    def list_feature_flags(self, project_id: str) -> List[Dict[str, Any]]:
        """List feature flags for a project."""
        return self._get(f"/v1/feature-flags?projectId={project_id}")

    def list_guardrails(self, project_id: str) -> Dict[str, Any]:
        """List guardrails for a project."""
        return self._get(f"/v1/guardrails?projectId={project_id}")

    def list_team(self, org_id: str) -> List[Dict[str, Any]]:
        """List team members for an organization."""
        return self._get(f"/v1/team?orgId={org_id}")

    def list_webhooks(self, org_id: str) -> List[Dict[str, Any]]:
        """List webhooks for an organization."""
        return self._get(f"/v1/webhooks?orgId={org_id}")

    def get_gateway_health(self) -> Dict[str, Any]:
        """Get gateway health status."""
        return self._get("/v1/gateway/health")

    def get_gateway_stats(self, project_id: str) -> Dict[str, Any]:
        """Get gateway usage statistics."""
        return self._get(f"/v1/gateway/stats?projectId={project_id}")

    def get_gateway_config(self, project_id: str) -> Dict[str, Any]:
        """Get gateway configuration."""
        return self._get(f"/v1/gateway?projectId={project_id}")

    def get_monitoring_drift(self, project_id: str) -> Dict[str, Any]:
        """Get drift detection status."""
        return self._get(f"/v1/monitoring/drift?projectId={project_id}")

    def get_monitoring_sla(self, project_id: str) -> Dict[str, Any]:
        """Get SLA monitoring data."""
        return self._get(f"/v1/monitoring/sla?projectId={project_id}")

    def get_cost_budget(self, project_id: str) -> Dict[str, Any]:
        """Get cost budget for a project."""
        return self._get(f"/v1/cost/budget?projectId={project_id}")

    def get_siem_connectors(self, project_id: str) -> Dict[str, Any]:
        """Get SIEM connector configuration."""
        return self._get(f"/v1/siem?projectId={project_id}")

    def get_settings(self, project_id: str) -> Dict[str, Any]:
        """Get project settings."""
        return self._get(f"/v1/settings?projectId={project_id}")

    def get_model_cards(
        self,
        project_id: str,
        model_name: str,
        provider: str,
        format: str = "json",
        eval_run_ids: Optional[List[str]] = None,
        scan_ids: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Generate a model card from eval/scan results for compliance.

        Backed by ``POST /v1/compliance/model-cards`` — the route is POST-only
        and requires ``projectId``, ``modelName`` and ``provider`` (a prior GET
        returned 405). ``format`` ∈ json | markdown. ``eval_run_ids`` /
        ``scan_ids`` pull in those runs' results to populate the card.
        """
        body: Dict[str, Any] = {
            "projectId": project_id,
            "modelName": model_name,
            "provider": provider,
            "format": format,
        }
        if eval_run_ids is not None:
            body["evalRunIds"] = eval_run_ids
        if scan_ids is not None:
            body["scanIds"] = scan_ids
        if metadata is not None:
            body["metadata"] = metadata
        return self._post("/v1/compliance/model-cards", json=body)

    def export_compliance(
        self,
        framework: str,
        organization_name: str,
        system_name: str,
        format: str = "json",
        scan_results: Optional[Dict[str, Any]] = None,
        use_case: Optional[str] = None,
        data_types: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Export a compliance audit package for a framework.

        Backed by ``POST /v1/compliance/export`` — the route is POST-only and
        requires ``framework``, ``organizationName`` and ``systemName`` (a prior
        GET with ``projectId`` returned 405). ``format`` ∈ html | json |
        markdown | pdf.
        """
        body: Dict[str, Any] = {
            "framework": framework,
            "organizationName": organization_name,
            "systemName": system_name,
            "format": format,
        }
        if scan_results is not None:
            body["scanResults"] = scan_results
        if use_case is not None:
            body["useCase"] = use_case
        if data_types is not None:
            body["dataTypes"] = data_types
        return self._post("/v1/compliance/export", json=body)

    def export_results(self, run_id: str, format: str, project_id: str) -> Dict[str, Any]:
        """Export eval results in specified format."""
        return self._get(f"/v1/exports?runId={run_id}&format={format}&projectId={project_id}")

    def generate_ai_sbom(
        self,
        project_name: str,
        project_version: Optional[str] = None,
        format: Optional[str] = None,
        package_json: Optional[Dict[str, Any]] = None,
        python_requirements: Optional[str] = None,
        provider_keys: Optional[List[str]] = None,
        live_cve_scan: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Generate an AI Software Bill of Materials from project manifests.

        Backed by ``POST /v1/ai-sbom/generate`` — ``projectName`` is the
        required field (a prior ``{"projectId": ...}`` body was rejected as
        invalid). ``format`` ∈ json | cyclonedx | spdx. Pass lockfiles /
        manifests for supply-chain CVE + typosquat coverage.
        """
        body: Dict[str, Any] = {"projectName": project_name}
        if project_version is not None:
            body["projectVersion"] = project_version
        if format is not None:
            body["format"] = format
        if package_json is not None:
            body["packageJson"] = package_json
        if python_requirements is not None:
            body["pythonRequirements"] = python_requirements
        if provider_keys is not None:
            body["providerKeys"] = provider_keys
        if live_cve_scan is not None:
            body["liveCveScan"] = live_cve_scan
        return self._post("/v1/ai-sbom/generate", json=body)

    def ingest_otlp_traces(self, resource_spans: List[Dict]) -> Dict[str, Any]:
        """Ingest OpenTelemetry traces."""
        return self._post("/v1/ingest/otlp/traces", json={"resourceSpans": resource_spans})

    def ingest_otlp_logs(self, resource_logs: List[Dict]) -> Dict[str, Any]:
        """Ingest OpenTelemetry logs."""
        return self._post("/v1/ingest/otlp/logs", json={"resourceLogs": resource_logs})

    def ingest_otlp_metrics(self, resource_metrics: List[Dict]) -> Dict[str, Any]:
        """Ingest OpenTelemetry metrics."""
        return self._post("/v1/ingest/otlp/metrics", json={"resourceMetrics": resource_metrics})

    # ─── Provider Keys (BYOK vault) ─────────────────────────────────────
    #
    # Plaintext API keys are encrypted server-side via Supabase Vault envelope
    # encryption; responses never include the plaintext.

    def list_provider_keys(
        self, org_id: str, project_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """List metadata for all BYOK provider keys in the org.

        Returns ``{"keys": [...], "total": N}``. Keys include ``id``,
        ``provider``, ``project_id``, ``label``, ``key_last4``, ``created_at``,
        ``rotated_at`` — never plaintext or ciphertext.
        """
        params = f"?orgId={org_id}"
        if project_id:
            params += f"&projectId={project_id}"
        return self._get(f"/v1/provider-keys{params}")

    def upsert_provider_key(
        self,
        org_id: str,
        provider: str,
        api_key: str,
        project_id: Optional[str] = None,
        label: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create or rotate a BYOK key. If a row already exists for
        (org, project, provider), the vault row is rewrapped in place.
        Returns ``{"key": {...metadata}, "rotated": bool}``.
        """
        body: Dict[str, Any] = {"orgId": org_id, "provider": provider, "apiKey": api_key}
        if project_id is not None:
            body["projectId"] = project_id
        if label is not None:
            body["label"] = label
        return self._post("/v1/provider-keys", json=body)

    def delete_provider_key(self, org_id: str, key_id: str) -> Dict[str, Any]:
        """Revoke a BYOK key. The underlying vault.secrets row is auto-cleaned."""
        return self._delete(f"/v1/provider-keys?id={key_id}&orgId={org_id}")

    # ─── Models Registry (custom pricing overrides) ────────────────────

    def list_models(
        self, org_id: str, project_id: Optional[str] = None, model: Optional[str] = None
    ) -> Dict[str, Any]:
        """List custom model pricing overrides."""
        params = f"?orgId={org_id}"
        if project_id:
            params += f"&projectId={project_id}"
        if model:
            params += f"&model={model}"
        return self._get(f"/v1/models/registry{params}")

    def upsert_model(
        self,
        org_id: str,
        model_name: str,
        input_price_per_1m_usd: float,
        output_price_per_1m_usd: float,
        project_id: Optional[str] = None,
        provider: Optional[str] = None,
        display_name: Optional[str] = None,
        context_window: Optional[int] = None,
        notes: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create or update a custom pricing override. Prices in USD per
        million tokens. Effect propagates within 60 s via server-side cache."""
        body: Dict[str, Any] = {
            "orgId": org_id,
            "modelName": model_name,
            "inputPricePer1mUsd": input_price_per_1m_usd,
            "outputPricePer1mUsd": output_price_per_1m_usd,
        }
        if project_id is not None:
            body["projectId"] = project_id
        if provider is not None:
            body["provider"] = provider
        if display_name is not None:
            body["displayName"] = display_name
        if context_window is not None:
            body["contextWindow"] = context_window
        if notes is not None:
            body["notes"] = notes
        return self._post("/v1/models/registry", json=body)

    def delete_model(self, org_id: str, model_id: str) -> Dict[str, Any]:
        """Remove a custom pricing override — the model falls back to built-in rates."""
        return self._delete(f"/v1/models/registry?id={model_id}&orgId={org_id}")

    # ─── API-key budget caps ───────────────────────────────────────────

    def get_api_key_budget(self, key_id: str) -> Dict[str, Any]:
        """Get the current month's spend + cap + percent-used for a virtual key.

        Returns ``{"keyId", "name", "monthlyBudgetUsd", "currentPeriodSpentUsd",
        "currentPeriodStartedAt", "remainingUsd", "percentUsed", "staleReset"}``.
        ``staleReset: true`` means the next gateway request will reset the
        counter due to month rollover.
        """
        return self._get(f"/v1/api-keys/{key_id}/budget")

    def set_api_key_budget(
        self, key_id: str, monthly_budget_usd: Optional[float]
    ) -> Dict[str, Any]:
        """Set or remove the monthly USD cap. Pass ``None`` to remove the cap.

        Once spend reaches the cap, the gateway proxy returns
        ``402 Payment Required`` until the next period begins.
        """
        return self._patch(
            f"/v1/api-keys/{key_id}/budget",
            json={"monthlyBudgetUsd": monthly_budget_usd},
        )

    def remove_api_key_budget(self, key_id: str) -> Dict[str, Any]:
        """Convenience alias for ``set_api_key_budget(key_id, None)``."""
        return self._delete(f"/v1/api-keys/{key_id}/budget")

    # ─── Trace attachments (inline blob storage) ───────────────────────

    def list_trace_attachments(self, trace_id: str, project_id: str) -> Dict[str, Any]:
        """List all attachments on a trace (metadata only — use
        ``fetch_trace_attachment`` to download binary payload).
        """
        return self._get(
            f"/v1/traces/{trace_id}/attachments?projectId={project_id}"
        )

    def upload_trace_attachment(
        self,
        trace_id: str,
        project_id: str,
        span_id: str,
        name: str,
        mime_type: str,
        data: Union[bytes, bytearray, str],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Attach a blob (image / audio / text / json / pdf) to a span.

        ``data`` may be raw bytes, a bytearray, or a base64-encoded string.
        1 MB size cap enforced client-side to avoid a wasted round trip.
        """
        if isinstance(data, (bytes, bytearray)):
            if len(data) > 1_048_576:
                raise ValueError(
                    f"Attachment exceeds 1 MB (got {len(data)} bytes); "
                    "V1 only supports inline storage."
                )
            payload_b64 = base64.b64encode(bytes(data)).decode("ascii")
        elif isinstance(data, str):
            # Strip data-URL prefix if caller passed one.
            payload_b64 = data.split(",", 1)[1] if data.startswith("data:") else data
            padding = payload_b64.count("=")
            decoded_bytes = (len(payload_b64) * 3) // 4 - padding
            if decoded_bytes > 1_048_576:
                raise ValueError(
                    f"Attachment exceeds 1 MB (decoded {decoded_bytes} bytes)."
                )
        else:
            raise TypeError("data must be bytes, bytearray, or base64 str")

        body: Dict[str, Any] = {
            "projectId": project_id,
            "spanId": span_id,
            "name": name,
            "mimeType": mime_type,
            "dataBase64": payload_b64,
        }
        if metadata is not None:
            body["metadata"] = metadata
        return self._post(f"/v1/traces/{trace_id}/attachments", json=body)

    def fetch_trace_attachment(
        self, trace_id: str, attachment_id: str, project_id: str
    ) -> bytes:
        """Download the raw bytes of an attachment. Returns the decoded
        binary payload; the caller should respect ``mime_type`` (available
        via ``list_trace_attachments``) when deciding how to render.
        """
        # Use the raw requests session so we get bytes, not JSON.
        url = f"{self.base_url}/v1/traces/{trace_id}/attachments/{attachment_id}?projectId={project_id}"
        resp = self.session.get(url, timeout=self.timeout)
        if resp.status_code >= 400:
            raise EvalGuardError(
                f"Failed to fetch attachment (status {resp.status_code})",
                status_code=resp.status_code,
                body=resp.text,
            )
        return resp.content

    def delete_trace_attachment(
        self, trace_id: str, attachment_id: str, project_id: str
    ) -> Dict[str, Any]:
        """Revoke an attachment."""
        return self._delete(
            f"/v1/traces/{trace_id}/attachments?id={attachment_id}&projectId={project_id}"
        )

    # ─── Agent-run metered billing (Gap #5) ─────────────────────────────

    def start_agent_run(
        self,
        api_key_id: Optional[str] = None,
        end_customer_id: Optional[str] = None,
        trace_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Open a new metered agent run. Returns ``{runId, status, startedAt}``.
        Pass ``x-evalguard-run-id: <runId>`` on gateway calls to meter them."""
        body: Dict[str, Any] = {}
        if api_key_id: body["apiKeyId"] = api_key_id
        if end_customer_id: body["endCustomerId"] = end_customer_id
        if trace_id: body["traceId"] = trace_id
        if metadata: body["metadata"] = metadata
        return self._post("/v1/agent-runs/start", json=body)

    def end_agent_run(
        self,
        run_id: str,
        cost_usd: float = 0.0,
        tokens_in: int = 0,
        tokens_out: int = 0,
        status: str = "completed",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Close an agent run. Idempotent. ``status`` ∈
        completed | failed | budget_exceeded."""
        body: Dict[str, Any] = {
            "costUsd": cost_usd, "tokensIn": tokens_in, "tokensOut": tokens_out, "status": status,
        }
        if metadata: body["metadata"] = metadata
        return self._post(f"/v1/agent-runs/{run_id}/end", json=body)

    def list_agent_runs(
        self,
        api_key_id: Optional[str] = None,
        agent_tag: Optional[str] = None,
        end_customer_id: Optional[str] = None,
        since: Optional[str] = None,
        limit: int = 100,
        group_by: Optional[str] = None,
    ) -> Dict[str, Any]:
        """List agent runs, optionally grouped for chargeback by
        agent_tag / end_customer_id / api_key_id."""
        params: List[str] = []
        if api_key_id: params.append(f"apiKeyId={api_key_id}")
        if agent_tag: params.append(f"agentTag={agent_tag}")
        if end_customer_id: params.append(f"endCustomerId={end_customer_id}")
        if since: params.append(f"since={since}")
        params.append(f"limit={limit}")
        if group_by: params.append(f"groupBy={group_by}")
        return self._get(f"/v1/agent-runs?{'&'.join(params)}")

    # ─── Model-scan governance (Gap #1) ─────────────────────────────────

    def promote_model_scan(
        self,
        scan_id: str,
        to_env: str,
        from_env: Optional[str] = None,
        override: bool = False,
        reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Promote a scanned model to an environment. Suspicious/malicious
        verdicts 403 unless override=True + reason (>=8 chars)."""
        body: Dict[str, Any] = {"toEnv": to_env, "override": override}
        if from_env: body["fromEnv"] = from_env
        if reason: body["reason"] = reason
        return self._post(f"/v1/security/model-scan/{scan_id}/promote", json=body)

    def get_model_scan_attestation(self, scan_id: str) -> Dict[str, Any]:
        """Fetch the CycloneDX-ML 1.6 attestation for a scan. Cached on first call."""
        return self._get(f"/v1/security/model-scan/{scan_id}/attestation")

    # ─── Shadow-AI discovery (Gap #2) ───────────────────────────────────

    def ingest_shadow_ai_sightings(
        self,
        source: str,
        rows: List[Dict[str, Any]],
        project_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Ingest egress / SSO / CASB log rows. ``source`` ∈ zscaler |
        netskope | cloudflare | okta | generic. Server-side merge is
        additive — re-ingesting the same rows increments counts,
        never overwrites."""
        body: Dict[str, Any] = {"source": source, "rows": rows}
        if project_id: body["projectId"] = project_id
        return self._post("/v1/shadow-ai/ingest", json=body)

    def set_shadow_ai_policy(
        self,
        domain: str,
        status: str,
        rationale: Optional[str] = None,
        project_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Per-domain policy override. ``status`` ∈ approved | blocked | pending."""
        body: Dict[str, Any] = {"domain": domain, "status": status}
        if rationale: body["rationale"] = rationale
        if project_id: body["projectId"] = project_id
        return self._post("/v1/shadow-ai/policy", json=body)

    def list_shadow_ai_policies(self, project_id: str) -> Dict[str, Any]:
        return self._get(f"/v1/shadow-ai/policy?projectId={project_id}")

    def delete_shadow_ai_policy(self, domain: str, project_id: str) -> Dict[str, Any]:
        return self._delete(f"/v1/shadow-ai/policy?domain={domain}&projectId={project_id}")

    # ─── SIEM inbound tokens (Gap #6) ───────────────────────────────────

    def create_siem_inbound_token(
        self,
        source: str,
        label: str,
        allowed_actions: Optional[List[str]] = None,
        rate_limit_per_min: int = 30,
        project_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Mint a SIEM inbound-webhook HMAC token. ``source`` ∈ splunk |
        sentinel | qradar | generic_webhook. The ``hmacSecret`` in the
        response is shown ONCE — save it into the SIEM now."""
        body: Dict[str, Any] = {
            "source": source,
            "label": label,
            "allowedActions": allowed_actions or ["quarantine_key"],
            "rateLimitPerMin": rate_limit_per_min,
        }
        if project_id: body["projectId"] = project_id
        return self._post("/v1/siem/inbound/tokens", json=body)

    def list_siem_inbound_tokens(self, project_id: str) -> Dict[str, Any]:
        return self._get(f"/v1/siem/inbound/tokens?projectId={project_id}")

    def revoke_siem_inbound_token(self, token_id: str, project_id: str) -> Dict[str, Any]:
        return self._delete(f"/v1/siem/inbound/tokens?id={token_id}&projectId={project_id}")

    # ─── Debug agent (Gap #4) ───────────────────────────────────────────

    def analyze_trace(
        self,
        trace_id: str,
        scorer_result_ids: Optional[List[str]] = None,
        analyzer_model: Optional[str] = None,
        analyzer_provider: Optional[str] = None,
        expected_output: Optional[str] = None,
        inline_context: Optional[Dict[str, Any]] = None,
        project_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Ask the debug agent to analyze a failing trace. Returns
        ``{sessionId, fixKind, confidence, rationale, suggestedFix,
        analyzerCostUsd}``. Passes BYOK OpenAI key first, falls back
        to server key only when none is stored. ``inlineContext`` bypasses
        the DB lookup — useful when your trace lives outside EvalGuard."""
        body: Dict[str, Any] = {"traceId": trace_id}
        if scorer_result_ids: body["scorerResultIds"] = scorer_result_ids
        if analyzer_model: body["analyzerModel"] = analyzer_model
        if analyzer_provider: body["analyzerProvider"] = analyzer_provider
        if expected_output: body["expectedOutput"] = expected_output
        if inline_context: body["inlineContext"] = inline_context
        if project_id: body["projectId"] = project_id
        return self._post("/v1/debug-agent", json=body)

    # ─── Agent Tools (the tool builder — full CRUD + test) ──────────────
    #
    # An ``AgentTool`` is ``{id?, name, description, type, parameters, rest?,
    # code?, mcp?, hasSecret?}`` where ``type`` ∈ rest | code | mcp and
    # ``parameters`` is a JSON-Schema object. ``test_agent_tool`` dry-runs the
    # tool against sample ``args`` and reports which stage (validate / execute)
    # passed plus any issues.

    def list_agent_tools(self, project_id: str) -> Dict[str, Any]:
        """List all agent tools for a project. Returns ``{"tools": [...]}``."""
        return self._get("/v1/agent-tools", params={"projectId": project_id})

    def create_agent_tool(
        self, project_id: str, tool: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create an agent tool. ``tool`` is an ``AgentTool`` (without ``id``).
        Returns the created ``AgentTool`` (with ``id``)."""
        return self._post(
            "/v1/agent-tools", json={"projectId": project_id, "tool": tool}
        )

    def get_agent_tool(self, tool_id: str, project_id: str) -> Dict[str, Any]:
        """Fetch a single agent tool by ID. Returns the ``AgentTool``."""
        return self._get(
            f"/v1/agent-tools/{tool_id}", params={"projectId": project_id}
        )

    def update_agent_tool(
        self, tool_id: str, project_id: str, tool: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Update an agent tool. ``tool`` carries the changed fields.
        Returns the updated ``AgentTool``."""
        return self._patch(
            f"/v1/agent-tools/{tool_id}",
            json={"projectId": project_id, "tool": tool},
        )

    def delete_agent_tool(self, tool_id: str, project_id: str) -> Dict[str, Any]:
        """Delete an agent tool. Returns ``{"id": ..., "deleted": True}``."""
        return self._delete(
            f"/v1/agent-tools/{tool_id}?projectId={project_id}"
        )

    def test_agent_tool(
        self, tool_id: str, project_id: str, args: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Dry-run an agent tool against sample ``args``. Returns
        ``{"ok", "stage", "status"?, "body"?, "issues"?, "message"?}`` where
        ``stage`` ∈ validate | execute."""
        return self._post(
            f"/v1/agent-tools/{tool_id}/test",
            json={"projectId": project_id, "args": args},
        )

    # ─── Abuse Reports (defense-in-depth intake) ────────────────────────
    #
    # ``category`` ∈ csam | violence | self_harm | harassment | hate | fraud |
    # privacy | spam | other. POST returns the stored ``report`` plus a
    # server-computed ``triage`` ``{severity, category, dedupKey, autoEscalate,
    # feedToDetector, reasons}``.

    def list_abuse_reports(
        self, project_id: str, status: Optional[str] = None
    ) -> Dict[str, Any]:
        """List abuse reports for a project. ``status`` ∈ open | reviewing |
        actioned | dismissed. Returns ``{"reports": [...]}``."""
        params: Dict[str, Any] = {"projectId": project_id}
        if status is not None:
            params["status"] = status
        return self._get("/v1/abuse-reports", params=params)

    def report_abuse(
        self,
        project_id: str,
        category: str,
        description: Optional[str] = None,
        subject_id: Optional[str] = None,
        reporter_id: Optional[str] = None,
        evidence: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """File an abuse report. ``category`` ∈ csam | violence | self_harm |
        harassment | hate | fraud | privacy | spam | other. Returns
        ``{"report": {...}, "triage": {...}}``."""
        body: Dict[str, Any] = {"projectId": project_id, "category": category}
        if description is not None:
            body["description"] = description
        if subject_id is not None:
            body["subjectId"] = subject_id
        if reporter_id is not None:
            body["reporterId"] = reporter_id
        if evidence is not None:
            body["evidence"] = evidence
        return self._post("/v1/abuse-reports", json=body)

    # ─── Agent Deployments (publish a workflow as a chat widget) ─────────
    #
    # A deployment publishes a workflow on a ``channel`` ∈ web | slack |
    # whatsapp | api and returns a ``public_id`` used to embed the widget.

    def list_agent_deployments(
        self, workflow_id: str, project_id: str
    ) -> Dict[str, Any]:
        """List deployments for a workflow. Returns ``{"deployments": [...]}``."""
        return self._get(
            f"/v1/workflows/{workflow_id}/deploy",
            params={"projectId": project_id},
        )

    def deploy_agent(
        self,
        workflow_id: str,
        project_id: str,
        channel: str,
        allowed_origins: Optional[List[str]] = None,
        greeting: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Publish a workflow as a chat widget. ``channel`` ∈ web | slack |
        whatsapp | api. Returns the created ``Deployment`` (with ``public_id``)."""
        body: Dict[str, Any] = {"projectId": project_id, "channel": channel}
        if allowed_origins is not None:
            body["allowedOrigins"] = allowed_origins
        if greeting is not None:
            body["greeting"] = greeting
        return self._post(f"/v1/workflows/{workflow_id}/deploy", json=body)

    def update_agent_deployment(
        self,
        deployment_id: str,
        project_id: str,
        status: Optional[str] = None,
        greeting: Optional[str] = None,
        allowed_origins: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Update a deployment. ``status`` ∈ active | paused. Returns the
        updated ``Deployment``."""
        body: Dict[str, Any] = {"projectId": project_id}
        if status is not None:
            body["status"] = status
        if greeting is not None:
            body["greeting"] = greeting
        if allowed_origins is not None:
            body["allowedOrigins"] = allowed_origins
        return self._patch(f"/v1/deployments/{deployment_id}", json=body)

    def delete_agent_deployment(
        self, deployment_id: str, project_id: str
    ) -> Dict[str, Any]:
        """Delete a deployment. Returns ``{"id": ..., "deleted": True}``."""
        return self._delete(
            f"/v1/deployments/{deployment_id}?projectId={project_id}"
        )

    # ─── Agent memory (two-tier: long-term semantic recall) ─────────────

    def remember_memory(
        self,
        project_id: str,
        session_key: str,
        facts: Optional[List[str]] = None,
        turns: Optional[List[Dict[str, str]]] = None,
        agent_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Store durable facts (or a conversation to extract facts from) for a
        session. Returns ``{"written": [...], "skipped": [...]}``."""
        if not facts and not turns:
            raise ValueError("remember_memory: provide facts or turns")
        body: Dict[str, Any] = {"projectId": project_id, "sessionKey": session_key}
        if facts is not None:
            body["facts"] = facts
        if turns is not None:
            body["turns"] = turns
        if agent_id is not None:
            body["agentId"] = agent_id
        return self._post("/v1/agent-memory", json=body)

    def recall_memory(
        self,
        project_id: str,
        session_key: str,
        query: Optional[str] = None,
        limit: Optional[int] = None,
        min_score: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Recall a session's long-term memory by semantic similarity to
        ``query`` (omit to list recent). Returns ``{"semantic": [...]}``."""
        params: Dict[str, Any] = {"projectId": project_id, "sessionKey": session_key}
        if query is not None:
            params["query"] = query
        if limit is not None:
            params["limit"] = limit
        if min_score is not None:
            params["minScore"] = min_score
        return self._get("/v1/agent-memory", params=params)

    def forget_memory(self, project_id: str, session_key: str) -> Dict[str, Any]:
        """Forget a session's long-term memory. Returns ``{"forgotten": n}``."""
        from urllib.parse import urlencode

        qs = urlencode({"projectId": project_id, "sessionKey": session_key})
        return self._delete(f"/v1/agent-memory?{qs}")

    # ─── Voice ML (word-level ASR + deepfake detection via sidecar) ─────

    def transcribe_voice(
        self,
        project_id: str,
        audio_base64: str,
        language: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Transcribe a base64-encoded WAV with WORD-LEVEL timestamps. Requires
        the operator-deployed voice-ML sidecar (503 otherwise). Returns
        ``{"language", "text", "words": [...], "segments": [...]}``."""
        body: Dict[str, Any] = {"projectId": project_id, "audioBase64": audio_base64}
        if language is not None:
            body["language"] = language
        return self._post("/v1/voice/transcribe", json=body)

    def score_voice_deepfake(
        self, project_id: str, audio_base64: str
    ) -> Dict[str, Any]:
        """Score a base64-encoded WAV for synthetic-speech / deepfake
        probability in [0,1]. Returns ``{"probability", "model", "scores"}``."""
        return self._post(
            "/v1/voice/deepfake-score",
            json={"projectId": project_id, "audioBase64": audio_base64},
        )

    # ─── Language detection (text → language) ──────────────────────────

    def detect_language(
        self, project_id: str, text: str, min_length: Optional[int] = None
    ) -> Dict[str, Any]:
        """Identify the language of a text snippet (franc-min, 82 languages).
        Returns ``{"iso6393", "iso6391", "name", "confidence", "reliable"}``."""
        if not text:
            raise ValueError("detect_language: text is required")
        body: Dict[str, Any] = {"projectId": project_id, "text": text}
        if min_length is not None:
            body["minLength"] = min_length
        return self._post("/v1/language/detect", json=body)

    # ─── MCP / agent security ──────────────────────────────────────────

    def audit_mcp_server(
        self,
        project_id: str,
        server: Dict[str, Any],
        tools: Optional[List[Dict[str, Any]]] = None,
        signoff: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Pre-deploy security audit of an MCP server config. Scans tool/parameter
        descriptions for injection, validates auth + encryption, flags dangerous
        tools without RBAC. Returns a severity report + approve/block verdict."""
        if not server:
            raise ValueError("audit_mcp_server: server is required")
        body: Dict[str, Any] = {"projectId": project_id, "server": server, "tools": tools or []}
        if signoff is not None:
            body["signoff"] = signoff
        return self._post("/v1/security/mcp-predeployment-audit", json=body)

    def run_agent_exec_redteam(
        self,
        project_id: str,
        target_provider: str,
        target_model: str,
        system_prompt: Optional[str] = None,
        attack_prompts: Optional[List[str]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Execution-layer red-team: drive a target agent with injections, intercept
        attempted tool calls, and report whether a dangerous call slipped past the
        firewall. Uses the org's BYOK provider key."""
        if not target_provider or not target_model:
            raise ValueError("run_agent_exec_redteam: target_provider and target_model are required")
        body: Dict[str, Any] = {
            "projectId": project_id,
            "target_provider": target_provider,
            "target_model": target_model,
        }
        if system_prompt is not None:
            body["system_prompt"] = system_prompt
        if attack_prompts is not None:
            body["attack_prompts"] = attack_prompts
        if tools is not None:
            body["tools"] = tools
        return self._post("/v1/security/agent-exec-redteam", json=body)

    def get_agent_graph(self, project_id: str, window_hours: Optional[int] = None) -> Dict[str, Any]:
        """Agent-to-agent (A2A) communication graph — who-calls-whom, aggregated
        from traces. Returns ``{"services", "edges", "totalCalls", "totalErrors"}``."""
        params: Dict[str, Any] = {"projectId": project_id}
        if window_hours is not None:
            params["windowHours"] = window_hours
        return self._get("/v1/traces/graph", params=params)

    # ─── Competitor-gap parity (mirrors the TS SDK gap methods) ──────────
    #
    # These methods mirror the TypeScript SDK's "competitor-gap" surface
    # (scanIac, scanSecrets, CVE waivers, governanceRisk, consensus,
    # lookupVulnerabilities, getScorecard, SBOM monitor, data-boundary,
    # runIncidentRca, syncIssues). Each is grounded in the corresponding
    # TS endpoint + payload so both SDKs hit the same contract.

    # ── AI-infra IaC / manifest static scan (G8) ───────────────────────

    def scan_iac(self, files: List[Dict[str, str]]) -> Dict[str, Any]:
        """Statically scan IaC / deployment manifests for AI-infra
        misconfigurations (model server bound 0.0.0.0 w/o auth, exposed
        AI-service ports, baked-in secrets, privileged GPU containers w/o
        limits). Stateless — no storage.

        ``files`` is ``[{"filename": str, "content": str}]``. Returns
        ``{"scannedFiles", "findingsCount", "bySeverity", "findings": [...]}``.
        Backed by ``POST /v1/security/iac-scan``.
        """
        if not files:
            raise ValueError("scan_iac: at least one file is required")
        return self._post("/v1/security/iac-scan", json={"files": files})

    # ── Committed-secret detection (G10) ───────────────────────────────

    def scan_secrets(
        self,
        content: Optional[str] = None,
        path: Optional[str] = None,
        files: Optional[List[Dict[str, str]]] = None,
        min_severity: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Detect committed secrets (API keys, private keys, cloud/SaaS tokens)
        in file contents — gitleaks-style. Pass a single ``content`` blob (with
        an optional ``path``) or a ``files`` array (e.g. a PR's changed files).
        Findings carry only the REDACTED match, never the raw secret.

        ``files`` is ``[{"path": str, "content": str}]``. ``min_severity`` ∈
        low | medium | high | critical. Returns ``{"scannedFiles",
        "filesWithFindings", "findingsCount", "findings", "severityCounts"}``.
        Backed by ``POST /v1/security/secret-scan``.
        """
        if not content and not files:
            raise ValueError("scan_secrets: provide `content` or a non-empty `files` array")
        body: Dict[str, Any] = {}
        if content is not None:
            body["content"] = content
        if path is not None:
            body["path"] = path
        if files is not None:
            body["files"] = files
        if min_severity is not None:
            body["minSeverity"] = min_severity
        return self._post("/v1/security/secret-scan", json=body)

    # ── Per-CVE waivers (G2) ───────────────────────────────────────────

    def list_cve_waivers(self, project_id: str) -> Dict[str, Any]:
        """List a project's CVE waivers. A waiver suppresses a specific
        (CVE, package) tuple from the supply-chain CI gate while keeping the
        finding visible. Returns ``{"waivers": [...], "total": N}``. Backed by
        ``GET /v1/supply-chain/waivers?projectId=``.
        """
        if not project_id:
            raise ValueError("list_cve_waivers: project_id is required")
        return self._get("/v1/supply-chain/waivers", params={"projectId": project_id})

    def add_cve_waiver(
        self,
        project_id: str,
        cve_id: str,
        affected_package: str,
        reason: str,
        severity: Optional[str] = None,
        expires_at: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create (or upsert) a CVE waiver for a (CVE, package) tuple.
        Owner/admin only. Set ``expires_at`` (ISO timestamp) so the CVE
        re-surfaces and re-fails the gate once it lapses; omit / ``None`` =
        never expires. Returns ``{"waiver": {...}}``. Backed by
        ``POST /v1/supply-chain/waivers``.
        """
        if not project_id:
            raise ValueError("add_cve_waiver: project_id is required")
        if not cve_id:
            raise ValueError("add_cve_waiver: cve_id is required")
        if not affected_package:
            raise ValueError("add_cve_waiver: affected_package is required")
        if not reason:
            raise ValueError("add_cve_waiver: reason is required")
        body: Dict[str, Any] = {
            "projectId": project_id,
            "cveId": cve_id,
            "affectedPackage": affected_package,
            "reason": reason,
        }
        if severity is not None:
            body["severity"] = severity
        if expires_at is not None:
            body["expiresAt"] = expires_at
        return self._post("/v1/supply-chain/waivers", json=body)

    def remove_cve_waiver(self, waiver_id: str) -> Dict[str, Any]:
        """Revoke a CVE waiver by id, re-exposing its (CVE, package) to the
        gate. Owner/admin only. Returns ``{"deleted": bool}``. Backed by
        ``DELETE /v1/supply-chain/waivers/{id}``.
        """
        if not waiver_id:
            raise ValueError("remove_cve_waiver: waiver_id is required")
        from urllib.parse import quote

        return self._delete(f"/v1/supply-chain/waivers/{quote(waiver_id, safe='')}")

    # ── Governance risk (G12) ──────────────────────────────────────────

    def governance_risk(
        self,
        security_findings: Optional[Dict[str, int]] = None,
        supply_chain_score: Optional[float] = None,
        vulnerability_score: Optional[float] = None,
        compliance_coverage: Optional[float] = None,
        firewall_hits: Optional[Dict[str, int]] = None,
        eval_pass_rate: Optional[float] = None,
        weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """Composite multi-axis AI governance risk score (G12). Combines the
        per-axis risk signals you provide (security findings, supply-chain /
        vulnerability scores, compliance coverage, firewall hits, eval pass
        rate) into one weighted 0-100 score with a per-axis breakdown +
        recommendations. Missing axes are excluded (not penalized).

        ``security_findings`` / ``firewall_hits`` are
        ``{"critical"?, "high"?, "medium"?, "low"?}``. Returns
        ``{"overallScore", "level", "axes", "missingAxes", "recommendations"}``.
        Backed by ``POST /v1/governance/risk``.
        """
        body: Dict[str, Any] = {}
        if security_findings is not None:
            body["securityFindings"] = security_findings
        if supply_chain_score is not None:
            body["supplyChainScore"] = supply_chain_score
        if vulnerability_score is not None:
            body["vulnerabilityScore"] = vulnerability_score
        if compliance_coverage is not None:
            body["complianceCoverage"] = compliance_coverage
        if firewall_hits is not None:
            body["firewallHits"] = firewall_hits
        if eval_pass_rate is not None:
            body["evalPassRate"] = eval_pass_rate
        if weights is not None:
            body["weights"] = weights
        return self._post("/v1/governance/risk", json=body)

    # ── Multi-LLM consensus (G13) ──────────────────────────────────────

    def gateway_consensus(
        self,
        candidates: List[Dict[str, Any]],
        method: Optional[str] = None,
        threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Reach consensus over N model responses to the same prompt (G13).
        You generate the completions (via any provider / the gateway); this
        clusters them and returns the agreed answer + an agreement score to
        gate high-stakes actions on.

        ``candidates`` is ``[{"model": str, "content"?: str, "error"?: str}]``.
        ``method`` ∈ similarity | exact. Returns ``{"chosen", "chosenModels",
        "agreement", "isMajority", "method", "clusters", "candidateCount",
        "successCount", "errorCount"}``. Backed by ``POST /v1/gateway/consensus``.
        """
        if not candidates:
            raise ValueError("gateway_consensus: candidates must be a non-empty list")
        body: Dict[str, Any] = {"candidates": candidates}
        if method is not None:
            body["method"] = method
        if threshold is not None:
            body["threshold"] = threshold
        return self._post("/v1/gateway/consensus", json=body)

    # ── Supply-chain: PURL vuln lookup + OpenSSF Scorecard ─────────────

    def lookup_vuln(self, purls: List[str]) -> Dict[str, Any]:
        """Look up known vulnerabilities for a list of Package URLs (PURLs) via
        OSV.dev. Supported ecosystems: npm, PyPI, Go. Invalid / unsupported
        PURLs are reported in-band (never silently dropped). Backed by
        ``POST /v1/supply-chain/lookup``.

        Example::

            client.lookup_vuln(["pkg:npm/lodash@4.17.21", "pkg:pypi/requests@2.31.0"])
        """
        if not purls:
            raise ValueError("lookup_vuln: purls must be a non-empty list")
        return self._post("/v1/supply-chain/lookup", json={"purls": purls})

    def get_scorecard(self, repo: str) -> Dict[str, Any]:
        """Fetch the OpenSSF Scorecard project-health signal (0-10) for a
        repository, plus the derived supply-chain risk contribution.
        Best-effort — unavailable projects return ``available: false``.
        Backed by ``POST /v1/supply-chain/scorecard``.

        Example::

            client.get_scorecard("github.com/lodash/lodash")
        """
        if not repo:
            raise ValueError("get_scorecard: repo is required")
        return self._post("/v1/supply-chain/scorecard", json={"repo": repo})

    # ── Continuous SBOM monitoring (G1) ────────────────────────────────

    def get_sbom_monitor(self, project_id: str) -> Dict[str, Any]:
        """Read a project's SBOM monitor config + recent snapshot history. The
        monitor is ``null`` when the project has never been configured. Any
        org member. Returns ``{"monitor": {...} | None, "snapshots": [...]}``.
        Backed by ``GET /v1/sbom-monitor?projectId=``.
        """
        if not project_id:
            raise ValueError("get_sbom_monitor: project_id is required")
        return self._get("/v1/sbom-monitor", params={"projectId": project_id})

    def set_sbom_monitor(
        self,
        project_id: str,
        enabled: Optional[bool] = None,
        epss_threshold: Optional[float] = None,
        alert_on_kev: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Enable / configure continuous SBOM monitoring for a project.
        Owner/admin only. Once enabled the worker re-runs the supply-chain scan
        every 24h and alerts on newly-disclosed KEV / high-EPSS CVEs.

        ``epss_threshold`` (0..1) = alert when a new CVE's exploit-probability
        is >= this; ``alert_on_kev`` = always alert on a new CISA KEV listing.
        Returns ``{"monitor": {...}}``. Backed by ``POST /v1/sbom-monitor``.
        """
        if not project_id:
            raise ValueError("set_sbom_monitor: project_id is required")
        body: Dict[str, Any] = {"projectId": project_id}
        if enabled is not None:
            body["enabled"] = enabled
        if epss_threshold is not None:
            body["epssThreshold"] = epss_threshold
        if alert_on_kev is not None:
            body["alertOnKev"] = alert_on_kev
        return self._post("/v1/sbom-monitor", json=body)

    def run_sbom_monitor(self, project_id: str) -> Dict[str, Any]:
        """Run the SBOM monitor for a project NOW (synchronous inline scan) and
        return the diff vs the last snapshot. Owner/admin only. Returns
        ``{"projectId", "vulnCount", "kevCount", "highEpssCount", "newVulns",
        "alertable", "scanMode", "liveStatus", "scannedAt"}``. Backed by
        ``POST /v1/sbom-monitor/run``.
        """
        if not project_id:
            raise ValueError("run_sbom_monitor: project_id is required")
        return self._post("/v1/sbom-monitor/run", json={"projectId": project_id})

    # ── Data-boundary façade (G11) — unified four-boundary policy ───────

    def list_data_boundary_policies(self, org_id: str) -> Dict[str, Any]:
        """List the org's data-boundary policies. A clearance-aware policy ties
        data classification to all four exposure boundaries (user-can-see /
        workflow-can-use / model-can-receive / output-can-reveal). Returns
        ``{"policies": [...], "total": N}``. Backed by
        ``GET /v1/data-boundary?orgId=``.
        """
        if not org_id:
            raise ValueError("list_data_boundary_policies: org_id is required")
        return self._get("/v1/data-boundary", params={"orgId": org_id})

    def create_data_boundary_policy(
        self,
        org_id: str,
        name: str,
        project_id: Optional[str] = None,
        classification_levels: Optional[List[str]] = None,
        boundary_rules: Optional[Dict[str, Any]] = None,
        enabled: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Create or update a data-boundary policy (keyed by org+name).
        Returns ``{"policy": {...}}``. Backed by ``POST /v1/data-boundary``.
        """
        if not org_id:
            raise ValueError("create_data_boundary_policy: org_id is required")
        if not name:
            raise ValueError("create_data_boundary_policy: name is required")
        body: Dict[str, Any] = {"orgId": org_id, "name": name}
        if project_id is not None:
            body["projectId"] = project_id
        if classification_levels is not None:
            body["classificationLevels"] = classification_levels
        if boundary_rules is not None:
            body["boundaryRules"] = boundary_rules
        if enabled is not None:
            body["enabled"] = enabled
        return self._post("/v1/data-boundary", json=body)

    def evaluate_data_boundary(
        self,
        org_id: str,
        boundary: str,
        policy_id: Optional[str] = None,
        policy_name: Optional[str] = None,
        content: Optional[str] = None,
        classification: Optional[str] = None,
        clearance: Optional[str] = None,
        agent_client_id: Optional[str] = None,
        tool: Optional[str] = None,
        action: Optional[str] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        data_scope: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Evaluate one boundary crossing against a stored policy. Composes the
        four engines (intent / per-agent authz / DLP / clearance ladder)
        server-side into one allow / redactions / reason verdict.

        ``boundary`` ∈ user-can-see | workflow-can-use | model-can-receive |
        output-can-reveal. Returns ``{"policyId", "policyName", "decision"}``.
        Backed by ``POST /v1/data-boundary/evaluate``.
        """
        if not org_id:
            raise ValueError("evaluate_data_boundary: org_id is required")
        if not boundary:
            raise ValueError("evaluate_data_boundary: boundary is required")
        body: Dict[str, Any] = {"orgId": org_id, "boundary": boundary}
        if policy_id is not None:
            body["policyId"] = policy_id
        if policy_name is not None:
            body["policyName"] = policy_name
        if content is not None:
            body["content"] = content
        if classification is not None:
            body["classification"] = classification
        if clearance is not None:
            body["clearance"] = clearance
        if agent_client_id is not None:
            body["agentClientId"] = agent_client_id
        if tool is not None:
            body["tool"] = tool
        if action is not None:
            body["action"] = action
        if provider is not None:
            body["provider"] = provider
        if model is not None:
            body["model"] = model
        if data_scope is not None:
            body["dataScope"] = data_scope
        return self._post("/v1/data-boundary/evaluate", json=body)

    # ── Alert-triggered incident RCA (G6) ──────────────────────────────

    def run_incident_rca(
        self,
        project_id: str,
        trigger: Optional[str] = None,
        window_minutes: Optional[int] = None,
        alert_message: Optional[str] = None,
        metric: Optional[str] = None,
        value: Optional[float] = None,
        threshold: Optional[float] = None,
        use_llm: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Run the alert-triggered RCA loop (G6) on demand over a trace window.
        Composes the error-classifier + trace-assistant (the same orchestrator
        the worker fires automatically on error_spike / sla_breach alerts) and
        returns a structured RCA: probable cause, evidence trace ids,
        recommendations.

        ``trigger`` ∈ error_spike | sla_breach. Backed by
        ``POST /v1/incidents/rca``.
        """
        if not project_id:
            raise ValueError("run_incident_rca: project_id is required")
        body: Dict[str, Any] = {"projectId": project_id}
        if trigger is not None:
            body["trigger"] = trigger
        if window_minutes is not None:
            body["windowMinutes"] = window_minutes
        if alert_message is not None:
            body["alertMessage"] = alert_message
        if metric is not None:
            body["metric"] = metric
        if value is not None:
            body["value"] = value
        if threshold is not None:
            body["threshold"] = threshold
        if use_llm is not None:
            body["useLLM"] = use_llm
        return self._post("/v1/incidents/rca", json=body)

    # ── Idempotent issue sync (G5) ─────────────────────────────────────

    def sync_issues(
        self,
        project_id: str,
        provider: str,
        findings: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Sync a project's security findings to its configured bug tracker
        (GitHub Issues / Jira) idempotently (G5). Each finding maps to ONE
        tracker issue via a stable dedup fingerprint (CVE/rule + file), so
        re-syncing UPDATES the same issue instead of creating a duplicate; a
        finding marked ``resolved`` (or one that disappeared since the last
        sync) CLOSES its issue. Owner/admin only. The tracker token comes from
        the org's integration config (never the request).

        ``provider`` ∈ github | jira. Each finding is
        ``{"title": str, "cveId"?, "rule"?, "file"?, "description"?,
        "severity"?, "remediation"?, "references"?, "status"?}``. Returns
        ``{"provider", "createdCount", "updatedCount", "closedCount",
        "errorCount", "created", "updated", "closed", "errors"}``. Backed by
        ``POST /v1/integrations/issue-sync``.
        """
        if not project_id:
            raise ValueError("sync_issues: project_id is required")
        if provider not in ("github", "jira"):
            raise ValueError('sync_issues: provider must be "github" or "jira"')
        if not findings:
            raise ValueError("sync_issues: findings must be a non-empty list")
        return self._post(
            "/v1/integrations/issue-sync",
            json={"projectId": project_id, "provider": provider, "findings": findings},
        )

    # ── Multimodal moderation (BYO vision / forensic models) ─────────────
    #
    # EvalGuard ships ZERO vision/forensic weights — these endpoints run the
    # moderation ENGINE (normalization, threshold, fail-CLOSED, frame
    # aggregation) against the project's BYO vendor. Image/video moderate via
    # the project's OpenAI omni-moderation key; deepfake detection proxies to
    # the operator's ML sidecar (DEEPFAKE_ML_SIDECAR_URL → 503 if unconfigured).
    # All three reply with the standard ``{success, data}`` envelope (unwrapped).

    def moderate_image(
        self,
        org_id: str,
        project_id: str,
        image_url: Optional[str] = None,
        image_base64: Optional[str] = None,
        mime_type: Optional[str] = None,
        threshold: Optional[float] = None,
        provider: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Vision content moderation of a single image (BYO vision model).

        Backed by ``POST /v1/moderation/image``. Supply exactly one of
        ``image_url`` (fetched server-side; SSRF-guarded) or ``image_base64``
        (inline). Requires a project OpenAI provider key configured under
        Settings → API Keys (400 ``PROVIDER_KEY_UNAVAILABLE`` otherwise).

        Returns ``{flagged, score, categories, categoryScores?, provider,
        latencyMs}`` (the ``{success, data}`` envelope is unwrapped).
        """
        if not org_id:
            raise ValueError("moderate_image: org_id is required")
        if not project_id:
            raise ValueError("moderate_image: project_id is required")
        if not image_url and not image_base64:
            raise ValueError("moderate_image: image_url or image_base64 is required")
        body: Dict[str, Any] = {"orgId": org_id, "projectId": project_id}
        if image_url is not None:
            body["imageUrl"] = image_url
        if image_base64 is not None:
            body["imageBase64"] = image_base64
        if mime_type is not None:
            body["mimeType"] = mime_type
        if threshold is not None:
            body["threshold"] = threshold
        if provider is not None:
            body["provider"] = provider
        return self._unwrap(self._post("/v1/moderation/image", json=body))

    def moderate_video(
        self,
        org_id: str,
        project_id: str,
        frames: List[Dict[str, Any]],
        threshold: Optional[float] = None,
        max_frames: Optional[int] = None,
        sample_every_n: Optional[int] = None,
        provider: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Frame-sampled video moderation (BYO vision model).

        Backed by ``POST /v1/moderation/video``. Frame EXTRACTION needs ffmpeg
        (a runtime concern), so the caller supplies the frames and the engine
        owns sampling + aggregation + per-frame fail-closed. Each frame is
        ``{"imageUrl"?: str, "imageBase64"?: str, "mimeType"?: str,
        "timestampMs"?: int}`` and needs ``imageUrl`` OR ``imageBase64``.

        Returns ``{flagged, score, categories, firstFlaggedFrame, framesTotal,
        framesEvaluated, frames[], provider, latencyMs}`` (envelope unwrapped).
        """
        if not org_id:
            raise ValueError("moderate_video: org_id is required")
        if not project_id:
            raise ValueError("moderate_video: project_id is required")
        if not isinstance(frames, list) or len(frames) == 0:
            raise ValueError("moderate_video: at least one frame is required")
        body: Dict[str, Any] = {
            "orgId": org_id,
            "projectId": project_id,
            "frames": frames,
        }
        if threshold is not None:
            body["threshold"] = threshold
        if max_frames is not None:
            body["maxFrames"] = max_frames
        if sample_every_n is not None:
            body["sampleEveryN"] = sample_every_n
        if provider is not None:
            body["provider"] = provider
        return self._unwrap(self._post("/v1/moderation/video", json=body))

    def detect_deepfake(
        self,
        org_id: str,
        project_id: str,
        kind: Optional[str] = None,
        image_url: Optional[str] = None,
        image_base64: Optional[str] = None,
        mime_type: Optional[str] = None,
        frames: Optional[List[Dict[str, Any]]] = None,
        threshold: Optional[float] = None,
        max_frames: Optional[int] = None,
        sample_every_n: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Visual deepfake / synthetic-media detection (BYO forensic sidecar).

        Backed by ``POST /v1/moderation/deepfake``. For an image pass
        ``image_url`` | ``image_base64``; for a video pass ``frames`` (the
        engine samples + aggregates). Fails CLOSED (synthetic) on detector
        error. The route returns 503 ``SIDECAR_UNCONFIGURED`` when no ML
        sidecar is configured (``DEEPFAKE_ML_SIDECAR_URL``).

        Returns ``{kind, synthetic, probability, label?, ...}`` (the
        ``{success, data}`` envelope is unwrapped).
        """
        if not org_id:
            raise ValueError("detect_deepfake: org_id is required")
        if not project_id:
            raise ValueError("detect_deepfake: project_id is required")
        has_frames = isinstance(frames, list) and len(frames) > 0
        if not image_url and not image_base64 and not has_frames:
            raise ValueError(
                "detect_deepfake: provide image_url/image_base64 (image) or "
                "frames[] (video)"
            )
        body: Dict[str, Any] = {"orgId": org_id, "projectId": project_id}
        if kind is not None:
            body["kind"] = kind
        if image_url is not None:
            body["imageUrl"] = image_url
        if image_base64 is not None:
            body["imageBase64"] = image_base64
        if mime_type is not None:
            body["mimeType"] = mime_type
        if frames is not None:
            body["frames"] = frames
        if threshold is not None:
            body["threshold"] = threshold
        if max_frames is not None:
            body["maxFrames"] = max_frames
        if sample_every_n is not None:
            body["sampleEveryN"] = sample_every_n
        return self._unwrap(self._post("/v1/moderation/deepfake", json=body))

    # ── Gateway (router-aware chat + routing-config management) ──────────

    def gateway_chat(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        tenant_id: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        fallback_models: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Run a chat completion through the gateway router.

        Backed by ``POST /v1/gateway``. When the hosted proxy's router is
        enabled for the org this exercises learned routing (priority / weighted
        / least-latency / least-cost / least-load / quality-cost / thompson) +
        per-provider failover; otherwise it falls back to a direct
        single-provider call. ``fallback_models`` are tried in order if the
        primary model's provider has no resolvable key (sent as
        ``options.fallbackModels`` to match the route schema).

        ``messages`` is ``[{"role": str, "content": str}]``. Returns
        ``{requestId?, model, provider, content, usage, cached, retries,
        latencyMs, costUsd?}`` (the ``{success, data}`` envelope is unwrapped).
        """
        if not isinstance(messages, list) or len(messages) == 0:
            raise ValueError("gateway_chat: at least one message is required")
        if not model:
            raise ValueError("gateway_chat: model is required")
        body: Dict[str, Any] = {"messages": messages, "model": model}
        if tenant_id is not None:
            body["tenantId"] = tenant_id
        if temperature is not None:
            body["temperature"] = temperature
        if max_tokens is not None:
            body["maxTokens"] = max_tokens
        if fallback_models:
            body["options"] = {"fallbackModels": fallback_models}
        return self._unwrap(self._post("/v1/gateway", json=body))

    def set_gateway_routing_config(
        self,
        org_id: Optional[str] = None,
        routing_strategy: Optional[str] = None,
        enabled: Optional[bool] = None,
        cache_enabled: Optional[bool] = None,
        cache_ttl_sec: Optional[int] = None,
        rate_limit_enabled: Optional[bool] = None,
        requests_per_minute: Optional[int] = None,
        tokens_per_minute: Optional[int] = None,
        circuit_breaker_enabled: Optional[bool] = None,
        providers: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Upsert this org's gateway routing config (admin-only server-side).

        Backed by ``PUT /v1/gateway``. Persists a per-org
        ``gateway_routing_config`` row (org-scoped RLS) that the hosted proxy
        reads for REAL learned routing + failover. Providers carry only
        non-secret routing knobs (``{"name", "enabled"?, "weight"?,
        "priority"?, "models"?}``) — provider API keys resolve from your stored
        Provider Keys (Vault) at request time, NEVER from this call.

        ``org_id`` is REQUIRED by the route (the server runs an admin-membership
        gate against it); when omitted the SDK auto-resolves the caller's
        default org via ``/v1/project/current``. ``routing_strategy`` ∈
        priority | round-robin | weighted | least-latency | least-cost |
        least-load | random | quality-cost | thompson.

        Returns the persisted config (the ``{success, data}`` envelope is
        unwrapped).
        """
        if not org_id:
            org_id = self._resolve_org_id()
        body: Dict[str, Any] = {"orgId": org_id}
        if routing_strategy is not None:
            body["routingStrategy"] = routing_strategy
        if enabled is not None:
            body["enabled"] = enabled
        if cache_enabled is not None:
            body["cacheEnabled"] = cache_enabled
        if cache_ttl_sec is not None:
            body["cacheTtlSec"] = cache_ttl_sec
        if rate_limit_enabled is not None:
            body["rateLimitEnabled"] = rate_limit_enabled
        if requests_per_minute is not None:
            body["requestsPerMinute"] = requests_per_minute
        if tokens_per_minute is not None:
            body["tokensPerMinute"] = tokens_per_minute
        if circuit_breaker_enabled is not None:
            body["circuitBreakerEnabled"] = circuit_breaker_enabled
        if providers is not None:
            body["providers"] = providers
        return self._unwrap(self._put("/v1/gateway", json=body))

    # ── Async batch inference (discounted tier) ──────────────────────────
    #
    # Submit many chat requests as one async batch processed off the gateway
    # hot path. Billed at a discount off the synchronous list price (default
    # 50%, like OpenAI/Fireworks); cost is surfaced as observability on the
    # batch (list_cost_usd vs cost_usd). See POST/GET /v1/batches.

    def create_batch(
        self,
        project_id: str,
        requests: List[Dict[str, Any]],
        model: Optional[str] = None,
        endpoint: Optional[str] = None,
        completion_window: Optional[str] = None,
        discount_pct: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Submit an async batch of chat requests.

        Backed by ``POST /v1/batches`` (returns 201). Each request is
        ``{"custom_id"?: str, "model"?: str, "messages": [{"role", "content"}]}``;
        up to 1000 per batch. ``discount_pct`` is the % off the list price for
        this batch's recorded cost (default = platform ``BATCH_DISCOUNT_PCT``,
        else 50). ``completion_window`` like ``"24h"`` / ``"30m"``.

        Returns ``{id, status, endpoint, total_requests, created_at,
        expires_at, discount_pct}`` (the ``{success, data}`` envelope is
        unwrapped).
        """
        if not project_id:
            raise ValueError("create_batch: project_id is required")
        if not isinstance(requests, list) or len(requests) == 0:
            raise ValueError("create_batch: requests must be a non-empty list")
        body: Dict[str, Any] = {"projectId": project_id, "requests": requests}
        if model is not None:
            body["model"] = model
        if endpoint is not None:
            body["endpoint"] = endpoint
        if completion_window is not None:
            body["completion_window"] = completion_window
        if discount_pct is not None:
            body["discount_pct"] = discount_pct
        if metadata is not None:
            body["metadata"] = metadata
        return self._unwrap(self._post("/v1/batches", json=body))

    def get_batch(self, batch_id: str) -> Dict[str, Any]:
        """Poll a batch's status, counts, results, and cost (list vs discounted).

        Backed by ``GET /v1/batches/{batchId}``. Returns the batch row (the
        ``{success, data}`` envelope is unwrapped). A missing / cross-org id
        surfaces as a 404 ``NOT_FOUND``.
        """
        if not batch_id:
            raise ValueError("get_batch: batch_id is required")
        return self._unwrap(self._get(f"/v1/batches/{batch_id}"))

    def list_batches(self, project_id: str) -> List[Dict[str, Any]]:
        """List recent batches for a project (newest first, capped at 50).

        Backed by ``GET /v1/batches?projectId=...`` (the ``{success, data}``
        envelope is unwrapped).
        """
        if not project_id:
            raise ValueError("list_batches: project_id is required")
        return self._unwrap(
            self._get("/v1/batches", params={"projectId": project_id})
        )

    def cancel_batch(self, batch_id: str) -> Dict[str, Any]:
        """Cancel an in-flight batch. Completed requests keep their results.

        Backed by ``POST /v1/batches/{batchId}/cancel``. Only non-terminal
        batches can be cancelled (409 ``BATCH_TERMINAL`` otherwise). Returns
        ``{id, status}`` (the ``{success, data}`` envelope is unwrapped).
        """
        if not batch_id:
            raise ValueError("cancel_batch: batch_id is required")
        return self._unwrap(self._post(f"/v1/batches/{batch_id}/cancel"))

    # ── Eval compare ─────────────────────────────────────────────────────

    def compare_evals(
        self, run_a: str, run_b: str, project_id: str
    ) -> Dict[str, Any]:
        """Compare two eval runs (regressions / improvements / per-case diff).

        Backed by ``GET /v1/evals/compare?runA=...&runB=...&projectId=...``.
        ``project_id`` is REQUIRED — the route is a cross-tenant defense that
        verifies BOTH runs belong to that project before returning any case
        data (a 404 masks runs in another project). Returns ``{run_a, run_b,
        score_diff, regressions, improvements, unchanged, cases}`` (the
        ``{success, data}`` envelope is unwrapped).
        """
        if not run_a or not run_b:
            raise ValueError("compare_evals: run_a and run_b are required")
        if not project_id:
            raise ValueError(
                "compare_evals: project_id is required (cross-tenant defense)"
            )
        return self._unwrap(
            self._get(
                "/v1/evals/compare",
                params={"runA": run_a, "runB": run_b, "projectId": project_id},
            )
        )

    # ── RAG ingest ───────────────────────────────────────────────────────

    def rag_ingest(
        self,
        documents: List[Dict[str, Any]],
        chunking: Optional[Dict[str, Any]] = None,
        embed: Optional[bool] = None,
        embed_model: Optional[str] = None,
        project_id: Optional[str] = None,
        dlp: Optional[Dict[str, Any]] = None,
        injection: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Managed chunk(+embed) pipeline for a batch of documents.

        Backed by ``POST /v1/rag/ingest``. Retriever-agnostic: returns the
        chunks (and, when ``embed=True``, their embeddings) so you store them
        in whatever vector DB you own — EvalGuard does NOT provision a managed
        vector store. Embedding uses the tenant's BYOK OpenAI key (resolved via
        ``project_id``), falling back to the server key.

        Each document is ``{"id"?: str, "text": str, "metadata"?: {...}}``.
        ``dlp`` (``{"mode": off|scan|redact|block, ...}``) screens secrets/PII
        BEFORE chunking; ``injection`` (``{"mode": off|scan|block, ...}``) vets
        for indirect prompt injection. Returns ``{chunks, chunkCount, embedded,
        model?, dlp?, injection?}`` (the ``{success, data}`` envelope is
        unwrapped).
        """
        if not isinstance(documents, list) or len(documents) == 0:
            raise ValueError("rag_ingest: at least one document is required")
        body: Dict[str, Any] = {"documents": documents}
        if chunking is not None:
            body["chunking"] = chunking
        if embed is not None:
            body["embed"] = embed
        if embed_model is not None:
            body["embedModel"] = embed_model
        if project_id is not None:
            body["projectId"] = project_id
        if dlp is not None:
            body["dlp"] = dlp
        if injection is not None:
            body["injection"] = injection
        return self._unwrap(self._post("/v1/rag/ingest", json=body))

    # ── Firewall (advanced / server-side controls) ───────────────────────

    def check_firewall_advanced(
        self,
        input_text: str,
        rules: Optional[List[str]] = None,
        sensitivity: Optional[Union[str, int]] = None,
        project_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run an INPUT firewall check with the advanced server-side controls.

        Backed by ``POST /v1/firewall/check``. Beyond the base
        :meth:`check_firewall`, this exposes the L1–L4 ``sensitivity`` dial
        (``"monitor"|"balanced"|"strict"|"lockdown"`` or the ordinal 1–4) and
        the ``rules`` attack-category enforcement that the route honors at the
        BLOCK level.

        NOTE on parity: the TS/JS SDK's ``checkFirewallAdvanced`` runs the core
        ``FirewallEngine`` IN-PROCESS (GCG perplexity / embedding paraphrase /
        YARA / RAG-grounding rails) with no network. The Python SDK is a thin
        HTTP client and ships no engine, so the in-process-only rails are not
        available here — this method drives the same firewall through the
        hosted route, which runs the full 5-layer ensemble plus the
        sensitivity + force-block controls.

        Returns ``{blocked, score, category, subcategory, sensitivity,
        latencyMs, hits}`` (the ``{success, data}`` envelope is unwrapped).
        """
        if not input_text:
            raise ValueError("check_firewall_advanced: input_text is required")
        body: Dict[str, Any] = {"input": input_text}
        if rules is not None:
            body["rules"] = rules
        if sensitivity is not None:
            body["sensitivity"] = sensitivity
        if project_id is not None:
            body["projectId"] = project_id
        return self._unwrap(self._post("/v1/firewall/check", json=body))

    def check_firewall_output_advanced(
        self,
        output: str,
        rules: Optional[List[str]] = None,
        sensitivity: Optional[Union[str, int]] = None,
        project_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run an OUTPUT firewall check with the advanced server-side controls.

        There is no dedicated hosted OUTPUT firewall route — the TS/JS SDK's
        ``checkFirewallOutputAdvanced`` runs the core ``FirewallEngine``'s
        ``scanOutput`` IN-PROCESS (YARA output rails + RAG retrieval-grounding),
        which the Python thin client cannot replicate. As the closest
        server-side equivalent, this screens the model OUTPUT text through the
        hosted ``POST /v1/firewall/check`` ensemble (PII / secret-leak /
        system-prompt-leak detection all apply to output text), exposing the
        same ``sensitivity`` dial + ``rules`` force-block controls.

        Returns ``{blocked, score, category, subcategory, sensitivity,
        latencyMs, hits}`` (the ``{success, data}`` envelope is unwrapped).
        """
        if not output:
            raise ValueError("check_firewall_output_advanced: output is required")
        return self.check_firewall_advanced(
            output, rules=rules, sensitivity=sensitivity, project_id=project_id
        )

    # ── Governance: intent classification ────────────────────────────────

    def classify_intent(
        self,
        prompt: str,
        org_id: Optional[str] = None,
        sensitivity_floor: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Classify a prompt's intent, data-sensitivity, and governance risk.

        Backed by ``POST /v1/governance/intent/classify`` — a stateless,
        deterministic core classifier (no model round-trip, no DB write).
        Powers intent-based routing + intent-conditioned policy. ``org_id`` is
        REQUIRED by the route (org-scoped for the membership gate +
        rate-limiting); when omitted the SDK auto-resolves the caller's default
        org via ``/v1/project/current``. ``sensitivity_floor`` ∈ public |
        internal | confidential | restricted raises the classified sensitivity
        to at least that level.

        Returns ``{intent, confidence, sensitivity, riskScore, signals,
        scores}`` (the ``{success, data}`` envelope is unwrapped).
        """
        if not prompt:
            raise ValueError("classify_intent: prompt is required")
        if not org_id:
            org_id = self._resolve_org_id()
        body: Dict[str, Any] = {"orgId": org_id, "prompt": prompt}
        if sensitivity_floor is not None:
            body["sensitivityFloor"] = sensitivity_floor
        return self._unwrap(
            self._post("/v1/governance/intent/classify", json=body)
        )
