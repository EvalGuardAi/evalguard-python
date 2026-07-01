"""EvalGuard Python SDK -- evaluate, red-team, and guard LLM applications."""

from .client import EvalGuardClient, EvalGuardError
from .guardrails import GuardrailClient, GuardrailViolation
try:
    from .pytest_plugin import evalguard_test
except ImportError:
    evalguard_test = None  # pytest not installed — plugin available when pytest is
from .tracing import (
    Span,
    configure as configure_tracing,
    flush as flush_traces,
    get_current_span,
    get_current_trace_id,
    get_session,
    set_session,
    trace,
    traceable,
)
# R9-1: Pydantic validator span emission. Public surface is the
# configure helper + counter snapshot — the plugin itself is wired
# via the pydantic entry point in setup.py.
from .pydantic_integration import (
    configure_pydantic,
    get_validation_counters,
    reset_validation_counters,
)
from .types import (
    BenchmarkResult,
    CaseResult,
    ComplianceReport,
    DriftReport,
    EvalCase,
    EvalResult,
    EvalRun,
    FirewallResult,
    FirewallRule,
    SecurityFinding,
    SecurityScanResult,
    TokenUsage,
)

__version__ = "2.1.0"

# Cross-SDK naming parity: the TypeScript SDK's primary export is `EvalGuard`
# (`import { EvalGuard } from "@evalguard/sdk"`). Expose the same class name in
# Python so both SDKs share one name. `EvalGuardClient` stays the canonical name
# and remains fully supported — `EvalGuard` is a non-breaking alias of it.
EvalGuard = EvalGuardClient

# Framework integrations -- lazy-imported to avoid hard dependencies.
# Users import from the submodule directly:
#   from evalguard.dify import DifyWebhookHandler, DifyGuardrail
#   from evalguard.chainlit import chainlit_trace, ChainlitTracer, ChainlitFeedback
#   from evalguard.gradio_integration import GradioGuard, traced_chat, guarded_chat
#   from evalguard.google_adk import EvalGuardAgentCallback
#   from evalguard.agno import EvalGuardTool, EvalGuardMonitor
#   from evalguard.browseruse import EvalGuardBrowserCallback
#   from evalguard.smolagents import EvalGuardMonitor, EvalGuardTool
#   from evalguard.composio import EvalGuardMiddleware, wrap_toolset
#   from evalguard.camel_ai import EvalGuardCamelMonitor, guard_society
#   from evalguard.mlflow_integration import EvalGuardMLflowCallback, log_evalguard_run, import_mlflow_experiment, make_evalguard_metric
#   from evalguard.mirascope import evalguard_trace, EvalGuardMirascopeMiddleware
#   from evalguard.controlflow import EvalGuardTaskObserver, observe_flow, guard_task_output
#   from evalguard.dspy import EvalGuardDSPyCallback, guard_module
#   from evalguard.strands import guard, guard_agent

__all__ = [
    # Core client
    "EvalGuard",
    "EvalGuardClient",
    "EvalGuardError",
    # Guardrails
    "GuardrailClient",
    "GuardrailViolation",
    # Pytest plugin
    "evalguard_test",
    # Tracing
    "traceable",
    "trace",
    "Span",
    "configure_tracing",
    "flush_traces",
    "get_current_span",
    "get_current_trace_id",
    "set_session",
    "get_session",
    # R9-1 Pydantic plugin
    "configure_pydantic",
    "get_validation_counters",
    "reset_validation_counters",
    # Types
    "BenchmarkResult",
    "CaseResult",
    "ComplianceReport",
    "DriftReport",
    "EvalCase",
    "EvalResult",
    "EvalRun",
    "FirewallResult",
    "FirewallRule",
    "SecurityFinding",
    "SecurityScanResult",
    "TokenUsage",
]
