"""EvalGuard Python SDK — Domain types matching the TypeScript definitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ── Enums as string literals ──

EvalStatus = str  # "pending" | "running" | "passed" | "failed" | "error"
Severity = str  # "critical" | "high" | "medium" | "low" | "info"
PlanTier = str  # "free" | "pro" | "team" | "enterprise"


@dataclass
class TokenUsage:
    prompt: int
    completion: int
    total: int


@dataclass
class EvalCase:
    id: str
    eval_run_id: str
    input: str
    expected_output: Optional[str] = None
    actual_output: Optional[str] = None
    score: Optional[float] = None
    passed: Optional[bool] = None
    latency: Optional[float] = None
    token_usage: Optional[TokenUsage] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalRun:
    id: str
    project_id: str
    name: str
    status: EvalStatus
    score: Optional[float]
    max_score: float
    duration: Optional[float]
    created_at: str
    completed_at: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CaseResult:
    input: str
    actual_output: str
    score: float
    passed: bool
    latency: float
    expected_output: Optional[str] = None
    scorer_results: Dict[str, Any] = field(default_factory=dict)
    token_usage: Optional[TokenUsage] = None


@dataclass
class EvalResult:
    cases: List[CaseResult]
    score: float
    max_score: float
    pass_rate: float
    total_latency: float
    total_tokens: int


@dataclass
class SecurityFinding:
    id: str
    scan_id: str
    type: str
    severity: Severity
    title: str
    description: str
    input: str
    output: str
    passed: bool = True
    plugin_id: Optional[str] = None
    strategy_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SecurityScanResult:
    findings: List[SecurityFinding]
    pass_rate: float
    critical_count: int
    high_count: int
    medium_count: int
    low_count: int
    total_tests: int
    duration: float


@dataclass
class FirewallResult:
    action: str  # "allow" | "block" | "flag"
    reasons: List[Dict[str, Any]]
    latency_ms: float


@dataclass
class FirewallRule:
    id: str
    name: str
    type: str  # "pii" | "injection" | "toxic" | "topic" | "custom"
    enabled: bool
    config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ComplianceReport:
    framework: str
    total_controls: int
    tested_controls: int
    passed_controls: int
    failed_controls: int
    coverage: float
    findings: List[Dict[str, Any]]


@dataclass
class DriftReport:
    has_drift: bool
    overall_delta: float
    metric_deltas: List[Dict[str, Any]]
    alerts: List[str]


@dataclass
class BenchmarkResult:
    suite: str
    model: str
    score: float
    cases: List[Dict[str, Any]]
    duration: float
