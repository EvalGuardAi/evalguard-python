"""LangChain callback handler for EvalGuard.

Usage::

    from evalguard.langchain import EvalGuardCallback
    from langchain_openai import ChatOpenAI

    callback = EvalGuardCallback(api_key="eg_...", project_id="proj_...")
    llm = ChatOpenAI(callbacks=[callback])
    # Every LLM call is now guarded and traced
    llm.invoke("Hello, world!")

Works with any LangChain LLM, chat model, or chain that supports callbacks.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Dict, List, Optional, Sequence, Union

from .guardrails import GuardrailClient, GuardrailViolation

# LangChain callback protocol: we implement the methods directly rather
# than inheriting from BaseCallbackHandler so the SDK has zero hard
# dependencies on LangChain.  When LangChain is installed, the duck-typed
# interface is fully compatible.


class EvalGuardCallback:
    """LangChain callback that guards every LLM call via EvalGuard.

    This class implements the LangChain callback protocol without importing
    LangChain, so it works with *any* version (0.1.x, 0.2.x, 0.3.x).

    Parameters
    ----------
    api_key:
        EvalGuard API key.
    project_id:
        Optional project ID for trace grouping.
    rules:
        Guardrail rules for input checking.
    block_on_violation:
        Raise :class:`GuardrailViolation` when input is blocked.
    """

    def __init__(
        self,
        api_key: str,
        project_id: Optional[str] = None,
        base_url: str = "https://evalguard.ai/api",
        rules: Optional[List[str]] = None,
        block_on_violation: bool = True,
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
        # Per-run state keyed by run_id
        self._runs: Dict[str, Dict[str, Any]] = {}

    # ── LangChain callback protocol ──────────────────────────────────

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        *,
        run_id: Optional[Any] = None,
        parent_run_id: Optional[Any] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Pre-LLM: check for prompt injection, PII, etc."""
        rid = str(run_id or uuid.uuid4())
        prompt_text = "\n".join(prompts)
        model_name = serialized.get("name", serialized.get("id", ["unknown"])[-1] if isinstance(serialized.get("id"), list) else "unknown")

        start = time.monotonic()
        check = self._guard.check_input(
            prompt_text,
            rules=self._rules,
            metadata={"model": model_name, "framework": "langchain"},
        )
        guard_ms = (time.monotonic() - start) * 1000

        self._runs[rid] = {
            "model": model_name,
            "input": prompt_text,
            "guard_ms": guard_ms,
            "violations": check.get("violations", []),
            "start": time.monotonic(),
        }

        if not check.get("allowed", True) and self._block:
            raise GuardrailViolation(check.get("violations", []))

    def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[Any],
        *,
        run_id: Optional[Any] = None,
        parent_run_id: Optional[Any] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Pre-chat-model: extract text from message objects and check."""
        prompts = []
        for message_group in messages:
            for msg in (message_group if isinstance(message_group, list) else [message_group]):
                content = getattr(msg, "content", "") if not isinstance(msg, dict) else msg.get("content", "")
                if isinstance(content, str):
                    prompts.append(content)
        self.on_llm_start(serialized, prompts, run_id=run_id, parent_run_id=parent_run_id, tags=tags, metadata=metadata, **kwargs)

    def on_llm_end(
        self,
        response: Any,
        *,
        run_id: Optional[Any] = None,
        parent_run_id: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        """Post-LLM: log the complete trace."""
        rid = str(run_id or "")
        run_data = self._runs.pop(rid, {})
        llm_ms = (time.monotonic() - run_data.get("start", time.monotonic())) * 1000

        output_text = _extract_lc_response(response)
        self._guard.log_trace(
            {
                "provider": "langchain",
                "model": run_data.get("model", "unknown"),
                "input": run_data.get("input", ""),
                "output": output_text,
                "guard_latency_ms": round(run_data.get("guard_ms", 0), 2),
                "llm_latency_ms": round(llm_ms, 2),
                "violations": run_data.get("violations", []),
            }
        )

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: Optional[Any] = None,
        parent_run_id: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        """Log error traces for failed LLM calls."""
        rid = str(run_id or "")
        run_data = self._runs.pop(rid, {})
        self._guard.log_trace(
            {
                "provider": "langchain",
                "model": run_data.get("model", "unknown"),
                "input": run_data.get("input", ""),
                "output": "",
                "error": str(error),
                "violations": run_data.get("violations", []),
            }
        )

    # ── Chain-level callbacks (no-ops, present for compatibility) ────

    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any) -> None:
        pass

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        pass

    def on_chain_error(self, error: BaseException, **kwargs: Any) -> None:
        pass

    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs: Any) -> None:
        pass

    def on_tool_end(self, output: str, **kwargs: Any) -> None:
        pass

    def on_tool_error(self, error: BaseException, **kwargs: Any) -> None:
        pass

    def on_text(self, text: str, **kwargs: Any) -> None:
        pass

    def on_retry(self, retry_state: Any, **kwargs: Any) -> None:
        pass


def _extract_lc_response(response: Any) -> str:
    """Extract text from a LangChain LLMResult or ChatResult."""
    try:
        # LLMResult / ChatResult have .generations
        generations = getattr(response, "generations", None)
        if generations:
            parts: list[str] = []
            for gen_list in generations:
                for gen in gen_list:
                    # ChatGeneration has .message.content; Generation has .text
                    msg = getattr(gen, "message", None)
                    if msg:
                        parts.append(getattr(msg, "content", str(msg)))
                    else:
                        parts.append(getattr(gen, "text", str(gen)))
            return "\n".join(parts)
    except Exception:
        pass
    return str(response) if response else ""
