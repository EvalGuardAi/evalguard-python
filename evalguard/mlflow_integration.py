"""MLflow integration for EvalGuard.

Bridges MLflow's experiment-tracking / evaluation system with EvalGuard's
eval, scoring, and tracing APIs.

Usage::

    from evalguard.mlflow_integration import (
        EvalGuardMLflowCallback,
        log_evalguard_run,
        import_mlflow_experiment,
    )

    # 1. Auto-log eval results as MLflow metrics
    callback = EvalGuardMLflowCallback(api_key="eg_...", project_id="proj_...")
    result = client.run_eval({...})
    callback.on_eval_complete(result)

    # 2. Push a single eval run to MLflow
    log_evalguard_run(eval_result, experiment_name="my-experiment")

    # 3. Pull an MLflow experiment into an EvalGuard dataset
    dataset = import_mlflow_experiment("12345", api_key="eg_...", project_id="proj_...")

Requires ``mlflow`` to be installed (``pip install mlflow``).
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from .client import EvalGuardClient
from .guardrails import GuardrailClient

logger = logging.getLogger("evalguard.mlflow")


def _require_mlflow() -> Any:
    """Import mlflow or raise a helpful error."""
    try:
        import mlflow
        return mlflow
    except ImportError:
        raise ImportError(
            "mlflow is required for this integration. "
            "Install it with: pip install mlflow"
        )


# ── Callback ────────────────────────────────────────────────────────────


class EvalGuardMLflowCallback:
    """Logs EvalGuard evaluation results as MLflow metrics and artifacts.

    Automatically creates (or reuses) an MLflow experiment and logs:
    - Overall score, pass rate, latency, and token count as metrics
    - Per-case scorer breakdowns as nested metrics
    - The full eval result dict as a JSON artifact

    Parameters
    ----------
    api_key:
        EvalGuard API key.
    project_id:
        EvalGuard project ID for trace grouping.
    experiment_name:
        MLflow experiment name.  Created if it does not exist.
    tracking_uri:
        MLflow tracking URI.  Defaults to the active tracking URI.
    guardrail_rules:
        Optional guardrail rules to check inputs before eval.
    """

    def __init__(
        self,
        api_key: str,
        project_id: Optional[str] = None,
        base_url: str = "https://evalguard.ai/api",
        experiment_name: str = "evalguard-evals",
        tracking_uri: Optional[str] = None,
        guardrail_rules: Optional[List[str]] = None,
        timeout: float = 5.0,
    ) -> None:
        self._mlflow = _require_mlflow()
        self._guard = GuardrailClient(
            api_key=api_key,
            base_url=base_url,
            project_id=project_id,
            timeout=timeout,
        )
        self._rules = guardrail_rules
        self._experiment_name = experiment_name

        if tracking_uri:
            self._mlflow.set_tracking_uri(tracking_uri)

        self._mlflow.set_experiment(experiment_name)

    # ── Public API ──────────────────────────────────────────────────────

    def on_eval_complete(
        self,
        eval_result: Dict[str, Any],
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> str:
        """Log an EvalGuard eval result as an MLflow run.

        Parameters
        ----------
        eval_result:
            The dict returned by ``EvalGuardClient.run_eval()``.
        run_name:
            Optional human-readable run name.
        tags:
            Extra MLflow tags to attach to the run.

        Returns
        -------
        str
            The MLflow run ID.
        """
        mlflow = self._mlflow

        with mlflow.start_run(run_name=run_name or "evalguard-eval") as run:
            # Tags
            mlflow.set_tag("evalguard.source", "evalguard-python-sdk")
            mlflow.set_tag("evalguard.eval_id", eval_result.get("id", ""))
            if tags:
                for k, v in tags.items():
                    mlflow.set_tag(k, v)

            # Top-level metrics
            mlflow.log_metric("evalguard/score", eval_result.get("score", 0.0))
            mlflow.log_metric("evalguard/max_score", eval_result.get("maxScore", eval_result.get("max_score", 0.0)))
            mlflow.log_metric("evalguard/pass_rate", eval_result.get("passRate", eval_result.get("pass_rate", 0.0)))
            mlflow.log_metric("evalguard/total_latency_ms", eval_result.get("totalLatency", eval_result.get("total_latency", 0.0)))
            mlflow.log_metric("evalguard/total_tokens", eval_result.get("totalTokens", eval_result.get("total_tokens", 0)))

            # Per-case metrics
            cases = eval_result.get("cases", [])
            for i, case in enumerate(cases):
                prefix = f"evalguard/case_{i}"
                mlflow.log_metric(f"{prefix}/score", case.get("score", 0.0))
                mlflow.log_metric(f"{prefix}/passed", 1.0 if case.get("passed") else 0.0)
                mlflow.log_metric(f"{prefix}/latency_ms", case.get("latency", 0.0))

                # Scorer-level results
                for scorer_name, scorer_val in case.get("scorerResults", case.get("scorer_results", {})).items():
                    score_val = scorer_val if isinstance(scorer_val, (int, float)) else scorer_val.get("score", 0.0) if isinstance(scorer_val, dict) else 0.0
                    mlflow.log_metric(f"{prefix}/{scorer_name}", score_val)

            # Log params
            config = eval_result.get("config", {})
            if config.get("model"):
                mlflow.log_param("model", config["model"])
            if config.get("scorers"):
                mlflow.log_param("scorers", ",".join(config["scorers"]) if isinstance(config["scorers"], list) else str(config["scorers"]))

            # Full result as artifact
            import json
            import tempfile
            import os

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False, prefix="evalguard_result_"
            ) as f:
                json.dump(eval_result, f, indent=2, default=str)
                tmp_path = f.name
            try:
                mlflow.log_artifact(tmp_path, artifact_path="evalguard")
            finally:
                os.unlink(tmp_path)

            # Trace to EvalGuard
            self._guard.log_trace({
                "provider": "mlflow",
                "event": "eval_logged",
                "mlflow_run_id": run.info.run_id,
                "mlflow_experiment": self._experiment_name,
                "eval_id": eval_result.get("id", ""),
                "score": eval_result.get("score", 0.0),
            })

            return run.info.run_id

    def on_eval_start(
        self,
        config: Dict[str, Any],
    ) -> None:
        """Optional pre-eval hook: log config and check guardrails.

        Parameters
        ----------
        config:
            The eval config dict about to be sent to EvalGuard.
        """
        mlflow = self._mlflow

        # Guardrail check on all case inputs
        cases = config.get("cases", [])
        for case in cases:
            input_text = case.get("input", "")
            if input_text and self._rules:
                self._guard.check_input(
                    input_text,
                    rules=self._rules,
                    metadata={"framework": "mlflow", "event": "pre_eval"},
                )


# ── Standalone helpers ──────────────────────────────────────────────────


def log_evalguard_run(
    eval_result: Dict[str, Any],
    *,
    experiment_name: str = "evalguard-evals",
    run_name: Optional[str] = None,
    tracking_uri: Optional[str] = None,
    tags: Optional[Dict[str, str]] = None,
) -> str:
    """Push an EvalGuard eval result to MLflow as a tracked run.

    This is a convenience function that does not require an
    ``EvalGuardMLflowCallback`` instance -- it talks directly to MLflow.

    Parameters
    ----------
    eval_result:
        The dict returned by ``EvalGuardClient.run_eval()``.
    experiment_name:
        MLflow experiment name.
    run_name:
        Human-readable run name.
    tracking_uri:
        MLflow tracking URI.
    tags:
        Extra MLflow tags.

    Returns
    -------
    str
        The MLflow run ID.
    """
    mlflow = _require_mlflow()

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name or "evalguard-eval") as run:
        mlflow.set_tag("evalguard.source", "evalguard-python-sdk")
        mlflow.set_tag("evalguard.eval_id", eval_result.get("id", ""))
        if tags:
            for k, v in tags.items():
                mlflow.set_tag(k, v)

        # Metrics
        mlflow.log_metric("evalguard/score", eval_result.get("score", 0.0))
        mlflow.log_metric("evalguard/pass_rate", eval_result.get("passRate", eval_result.get("pass_rate", 0.0)))
        mlflow.log_metric("evalguard/total_latency_ms", eval_result.get("totalLatency", eval_result.get("total_latency", 0.0)))
        mlflow.log_metric("evalguard/total_tokens", eval_result.get("totalTokens", eval_result.get("total_tokens", 0)))

        cases = eval_result.get("cases", [])
        mlflow.log_metric("evalguard/num_cases", len(cases))

        passed = sum(1 for c in cases if c.get("passed"))
        mlflow.log_metric("evalguard/num_passed", passed)
        mlflow.log_metric("evalguard/num_failed", len(cases) - passed)

        # Params
        config = eval_result.get("config", {})
        if config.get("model"):
            mlflow.log_param("model", config["model"])
        if config.get("prompt"):
            mlflow.log_param("prompt_template", str(config["prompt"])[:250])

        # Artifact
        import json
        import tempfile
        import os

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, prefix="evalguard_"
        ) as f:
            json.dump(eval_result, f, indent=2, default=str)
            tmp_path = f.name
        try:
            mlflow.log_artifact(tmp_path, artifact_path="evalguard")
        finally:
            os.unlink(tmp_path)

        return run.info.run_id


def import_mlflow_experiment(
    experiment_id: str,
    *,
    api_key: str,
    project_id: str,
    base_url: str = "https://evalguard.ai/api",
    dataset_name: Optional[str] = None,
    max_runs: int = 500,
    tracking_uri: Optional[str] = None,
) -> Dict[str, Any]:
    """Pull runs from an MLflow experiment and create an EvalGuard dataset.

    Reads all runs from the specified MLflow experiment, extracts their
    parameters and metrics, and creates an EvalGuard dataset with one case
    per run.  This is useful for importing historical experiment data into
    EvalGuard for further analysis.

    Parameters
    ----------
    experiment_id:
        The MLflow experiment ID (numeric string).
    api_key:
        EvalGuard API key.
    project_id:
        EvalGuard project ID to create the dataset in.
    base_url:
        EvalGuard API base URL.
    dataset_name:
        Name for the created dataset.  Defaults to
        ``"mlflow-import-{experiment_id}"``.
    max_runs:
        Maximum number of MLflow runs to import.
    tracking_uri:
        MLflow tracking URI.

    Returns
    -------
    dict
        The created EvalGuard dataset object.
    """
    mlflow = _require_mlflow()
    from mlflow.tracking import MlflowClient as _MlflowClient

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    mlflow_client = _MlflowClient()
    experiment = mlflow_client.get_experiment(experiment_id)

    if experiment is None:
        raise ValueError(f"MLflow experiment '{experiment_id}' not found")

    # Search runs
    runs = mlflow_client.search_runs(
        experiment_ids=[experiment_id],
        max_results=max_runs,
        order_by=["start_time DESC"],
    )

    # Convert MLflow runs to EvalGuard dataset cases
    cases: List[Dict[str, Any]] = []
    for mlflow_run in runs:
        metrics = dict(mlflow_run.data.metrics)
        params = dict(mlflow_run.data.params)
        run_tags = dict(mlflow_run.data.tags)

        # Build an eval case from the run data
        case: Dict[str, Any] = {
            "input": params.get("prompt_template", params.get("input", f"mlflow-run-{mlflow_run.info.run_id}")),
            "metadata": {
                "mlflow_run_id": mlflow_run.info.run_id,
                "mlflow_experiment_id": experiment_id,
                "mlflow_status": mlflow_run.info.status,
                "mlflow_start_time": mlflow_run.info.start_time,
                "mlflow_end_time": mlflow_run.info.end_time,
                "model": params.get("model", run_tags.get("mlflow.runName", "")),
                "metrics": metrics,
                "params": params,
            },
        }

        # If there are evalguard-specific metrics, extract them
        if "evalguard/score" in metrics:
            case["expectedOutput"] = str(metrics["evalguard/score"])

        cases.append(case)

    # Create dataset in EvalGuard
    eg_client = EvalGuardClient(api_key=api_key, base_url=base_url)
    name = dataset_name or f"mlflow-import-{experiment_id}"

    dataset = eg_client.create_dataset(
        project_id=project_id,
        name=name,
        cases=cases,
        description=f"Imported from MLflow experiment '{experiment.name}' ({experiment_id}), {len(cases)} runs",
    )

    logger.info(
        "Imported %d MLflow runs from experiment '%s' into EvalGuard dataset '%s'",
        len(cases),
        experiment.name,
        name,
    )

    return dataset


# ── MLflow Evaluate bridge ──────────────────────────────────────────────


def make_evalguard_metric(
    api_key: str,
    scorer: str = "semantic-similarity",
    project_id: Optional[str] = None,
    base_url: str = "https://evalguard.ai/api",
) -> Any:
    """Create an MLflow ``make_metric``-compatible evaluator backed by EvalGuard.

    This allows using EvalGuard's scorers (semantic-similarity, toxicity,
    faithfulness, etc.) within ``mlflow.evaluate()``.

    Usage::

        import mlflow

        eg_metric = make_evalguard_metric(api_key="eg_...", scorer="toxicity")
        results = mlflow.evaluate(
            model="runs:/abc123/model",
            data=eval_data,
            extra_metrics=[eg_metric],
        )

    Parameters
    ----------
    api_key:
        EvalGuard API key.
    scorer:
        EvalGuard scorer name (e.g. "semantic-similarity", "toxicity",
        "faithfulness", "exact-match").
    project_id:
        Optional EvalGuard project ID.
    base_url:
        EvalGuard API base URL.

    Returns
    -------
    mlflow.metrics.EvaluationMetric
        An MLflow evaluation metric.
    """
    mlflow = _require_mlflow()
    from mlflow.metrics import make_metric

    eg_client = EvalGuardClient(api_key=api_key, base_url=base_url)

    def _eval_fn(predictions, targets=None, metrics=None) -> Any:
        """Score each prediction via EvalGuard."""
        from mlflow.metrics import MetricValue

        scores: List[float] = []
        for i, pred in enumerate(predictions):
            target = targets[i] if targets is not None and i < len(targets) else None
            case = {"input": str(target or ""), "actualOutput": str(pred)}
            if target is not None:
                case["expectedOutput"] = str(target)

            try:
                result = eg_client.run_eval({
                    "cases": [case],
                    "scorers": [scorer],
                    **({"projectId": project_id} if project_id else {}),
                })
                case_results = result.get("cases", [])
                score = case_results[0].get("score", 0.0) if case_results else 0.0
            except Exception:
                logger.debug("EvalGuard scorer '%s' failed for case %d", scorer, i, exc_info=True)
                score = 0.0

            scores.append(score)

        aggregate = sum(scores) / len(scores) if scores else 0.0
        return MetricValue(
            aggregate_results={"evalguard_score": aggregate},
            scores=scores,
        )

    return make_metric(
        eval_fn=_eval_fn,
        name=f"evalguard_{scorer.replace('-', '_')}",
        greater_is_better=True,
    )
