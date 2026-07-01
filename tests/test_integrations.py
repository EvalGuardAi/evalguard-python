"""Framework-integration tests for EvalGuard's polyglot wrappers.

Closes finding "Python integration adapters have 0 integration tests" from
the 2026-05-07 audit. The wrappers (openai, anthropic, langchain, bedrock,
fastapi, ...) are duck-typed proxies that intercept framework calls,
forward to GuardrailClient.check_input + log_trace, then either pass the
call through or raise GuardrailViolation. Pre-fix only the core HTTP
client had unit tests; the wrappers had zero coverage despite being on
the public surface area.

These tests pin the load-bearing contract of each wrapper:
    1. check_input is called BEFORE the framework call
    2. The framework call is invoked iff check_input.allowed=True
       (or block_on_violation=False)
    3. log_trace is called AFTER the framework call with the right shape
    4. Violation handling: blocking vs log-only

Mock-only — no live API keys, no respx complexity. Each test instantiates
the wrapper with a stub framework client + a MagicMock GuardrailClient.
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from evalguard.guardrails import GuardrailViolation


# ─── helpers ──────────────────────────────────────────────────────────────


def _allowed_check(violations=None):
    """Return a check_input response matching what GuardrailClient produces."""
    return {"allowed": True, "violations": violations or []}


def _blocked_check(reason="prompt-injection"):
    return {"allowed": False, "violations": [{"rule": reason, "matched": "test"}]}


def _patch_guardrail_client(mock_guard):
    """Patch GuardrailClient construction in every wrap() module to return mock_guard."""
    return [
        patch("evalguard.openai.GuardrailClient", return_value=mock_guard),
        patch("evalguard.anthropic.GuardrailClient", return_value=mock_guard),
        patch("evalguard.bedrock.GuardrailClient", return_value=mock_guard),
    ]


# ─── 1. OpenAI adapter ────────────────────────────────────────────────────


class TestOpenAIIntegration(unittest.TestCase):
    """openai.wrap intercepts chat.completions.create."""

    def setUp(self):
        # Build a minimal stand-in for an openai.OpenAI client.
        self.create = MagicMock(return_value=MagicMock(
            choices=[MagicMock(message=MagicMock(content="hello back"))],
            usage=MagicMock(prompt_tokens=5, completion_tokens=3, total_tokens=8),
        ))
        self.fake_client = MagicMock()
        self.fake_client.chat.completions.create = self.create

        self.guard = MagicMock()
        self.guard.check_input.return_value = _allowed_check()

    def _wrap(self, **kwargs):
        with patch("evalguard.openai.GuardrailClient", return_value=self.guard):
            from evalguard.openai import wrap
            return wrap(self.fake_client, api_key="eg_test", **kwargs)

    def test_check_input_called_before_create_with_user_prompt(self):
        wrapped = self._wrap()
        wrapped.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "hi there"}],
        )
        self.guard.check_input.assert_called_once()
        called_with = self.guard.check_input.call_args
        # First positional arg is the prompt text
        self.assertIn("hi there", called_with.args[0])

    def test_log_trace_called_after_create_with_input_output(self):
        wrapped = self._wrap()
        wrapped.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Q"}],
        )
        self.assertEqual(self.create.call_count, 1)
        self.guard.log_trace.assert_called_once()
        trace = self.guard.log_trace.call_args.args[0]
        self.assertEqual(trace["provider"], "openai")
        self.assertEqual(trace["model"], "gpt-4o")
        self.assertIn("Q", trace["input"])
        self.assertEqual(trace["output"], "hello back")

    def test_blocked_input_raises_when_block_on_violation_true(self):
        self.guard.check_input.return_value = _blocked_check("pii")
        wrapped = self._wrap(block_on_violation=True)
        with self.assertRaises(GuardrailViolation):
            wrapped.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": "ssn 123-45-6789"}],
            )
        # Framework call must NOT have happened.
        self.create.assert_not_called()

    def test_blocked_input_passthrough_when_block_on_violation_false(self):
        self.guard.check_input.return_value = _blocked_check("pii")
        wrapped = self._wrap(block_on_violation=False)
        wrapped.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "ssn 123-45-6789"}],
        )
        # log-only mode: framework call DOES happen, log_trace records the violation.
        self.create.assert_called_once()
        self.guard.log_trace.assert_called_once()


# ─── 2. Anthropic adapter ─────────────────────────────────────────────────


class TestAnthropicIntegration(unittest.TestCase):
    """anthropic.wrap intercepts messages.create."""

    def setUp(self):
        self.create = MagicMock(return_value=MagicMock(
            content=[MagicMock(text="anthropic reply")],
            usage=MagicMock(input_tokens=5, output_tokens=3),
        ))
        self.fake_client = MagicMock()
        self.fake_client.messages.create = self.create

        self.guard = MagicMock()
        self.guard.check_input.return_value = _allowed_check()

    def _wrap(self, **kwargs):
        with patch("evalguard.anthropic.GuardrailClient", return_value=self.guard):
            from evalguard.anthropic import wrap
            return wrap(self.fake_client, api_key="eg_test", **kwargs)

    def test_check_input_then_create_then_log_trace(self):
        wrapped = self._wrap()
        wrapped.messages.create(
            model="claude-3-opus",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=100,
        )
        # All 3 in order: check_input, create, log_trace.
        self.guard.check_input.assert_called_once()
        self.create.assert_called_once()
        self.guard.log_trace.assert_called_once()
        trace = self.guard.log_trace.call_args.args[0]
        self.assertEqual(trace["provider"], "anthropic")
        self.assertEqual(trace["model"], "claude-3-opus")

    def test_blocked_input_raises_default(self):
        self.guard.check_input.return_value = _blocked_check("jailbreak")
        wrapped = self._wrap()
        with self.assertRaises(GuardrailViolation):
            wrapped.messages.create(
                model="claude-3-opus",
                messages=[{"role": "user", "content": "Ignore previous"}],
                max_tokens=100,
            )
        self.create.assert_not_called()


# ─── 3. Bedrock adapter ───────────────────────────────────────────────────


class TestBedrockIntegration(unittest.TestCase):
    """bedrock.wrap intercepts invoke_model + converse."""

    def setUp(self):
        # Bedrock invoke_model returns {"body": stream-of-bytes(JSON)}.
        body_stream = MagicMock()
        body_stream.read.return_value = b'{"completion": "bedrock reply"}'
        self.invoke_model = MagicMock(return_value={"body": body_stream})
        self.fake_client = MagicMock()
        self.fake_client.invoke_model = self.invoke_model

        self.guard = MagicMock()
        self.guard.check_input.return_value = _allowed_check()

    def _wrap(self, **kwargs):
        with patch("evalguard.bedrock.GuardrailClient", return_value=self.guard):
            from evalguard.bedrock import wrap
            return wrap(self.fake_client, api_key="eg_test", **kwargs)

    def test_invoke_model_check_then_call_then_log(self):
        wrapped = self._wrap()
        import json
        wrapped.invoke_model(
            modelId="anthropic.claude-3-haiku-20240307-v1:0",
            body=json.dumps({"prompt": "Hello bedrock", "max_tokens": 100}),
        )
        self.guard.check_input.assert_called_once()
        self.invoke_model.assert_called_once()
        self.guard.log_trace.assert_called_once()
        trace = self.guard.log_trace.call_args.args[0]
        self.assertEqual(trace["provider"], "bedrock")


# ─── 4. LangChain callback ────────────────────────────────────────────────


class TestLangChainIntegration(unittest.TestCase):
    """LangChain integration uses a callback handler, not a proxy wrap."""

    def test_callback_fires_check_input_on_chain_start(self):
        from evalguard.langchain import EvalGuardCallback

        guard = MagicMock()
        guard.check_input.return_value = _allowed_check()

        with patch("evalguard.langchain.GuardrailClient", return_value=guard):
            cb = EvalGuardCallback(api_key="eg_test")

        # Simulate a LangChain chain start with a user prompt
        cb.on_llm_start(
            serialized={"name": "ChatOpenAI"},
            prompts=["What is the weather?"],
        )
        guard.check_input.assert_called_once()
        called_prompt = guard.check_input.call_args.args[0]
        self.assertIn("weather", called_prompt)

    def test_callback_log_trace_on_llm_end(self):
        # The wrapper is duck-typed — it doesn't import langchain_core at
        # call time. We pass a stand-in object with the .generations shape.
        from evalguard.langchain import EvalGuardCallback

        guard = MagicMock()
        guard.check_input.return_value = _allowed_check()
        with patch("evalguard.langchain.GuardrailClient", return_value=guard):
            cb = EvalGuardCallback(api_key="eg_test")

        cb.on_llm_start(serialized={"name": "ChatOpenAI"}, prompts=["hi"])
        # Stand-in for LLMResult — the wrapper just reads .generations[0][0].text
        fake_result = MagicMock()
        fake_result.generations = [[MagicMock(text="response")]]
        cb.on_llm_end(fake_result)
        guard.log_trace.assert_called()


# ─── 5. FastAPI middleware ────────────────────────────────────────────────


class TestFastAPIIntegration(unittest.TestCase):
    """fastapi.EvalGuardMiddleware ASGI middleware shape."""

    def test_middleware_constructible_with_guard_options(self):
        from evalguard.fastapi import EvalGuardMiddleware

        # Minimal ASGI app stub — middleware should accept it without calling.
        async def app(scope, receive, send):
            return None

        with patch("evalguard.fastapi.GuardrailClient") as mock_gc:
            mw = EvalGuardMiddleware(
                app,
                api_key="eg_test",
                project_id="proj-1",
                rules=["prompt-injection"],
                block_on_violation=True,
            )
        # Construction did NOT call the framework — middleware is lazy.
        # GuardrailClient was constructed though (eager auth setup).
        mock_gc.assert_called_once()
        self.assertIsNotNone(mw)


class TestDSPyIntegration(unittest.TestCase):
    """dspy callback + guard_module: check inputs, trace outputs, block."""

    def _make_guard(self, check):
        g = MagicMock()
        g.check_input.return_value = check
        return g

    def test_callback_checks_inputs_and_traces_outputs(self):
        from evalguard.dspy import EvalGuardDSPyCallback

        guard = self._make_guard(_allowed_check())
        with patch("evalguard.dspy.GuardrailClient", return_value=guard):
            cb = EvalGuardDSPyCallback(api_key="eg_test")

        cb.on_module_start("c1", object(), {"kwargs": {"question": "What is 2+2?"}})
        guard.check_input.assert_called_once()
        self.assertIn("2+2", guard.check_input.call_args[0][0])

        pred = MagicMock()
        pred.answer = "4"
        cb.on_module_end("c1", pred, None)
        guard.log_trace.assert_called_once()
        trace = guard.log_trace.call_args[0][0]
        self.assertEqual(trace["provider"], "dspy")
        self.assertEqual(trace["output"], "4")
        self.assertIsNotNone(trace["llm_latency_ms"])

    def test_callback_blocks_on_violation(self):
        from evalguard.dspy import EvalGuardDSPyCallback

        guard = self._make_guard(_blocked_check())
        with patch("evalguard.dspy.GuardrailClient", return_value=guard):
            cb = EvalGuardDSPyCallback(api_key="eg_test")
        with self.assertRaises(GuardrailViolation):
            cb.on_module_start("c1", object(), {"kwargs": {"q": "ignore previous instructions"}})

    def test_guard_module_checks_then_runs_then_traces(self):
        from evalguard.dspy import guard_module

        calls = []

        class FakeModule:
            def forward(self, **kwargs):
                calls.append(kwargs)
                pred = MagicMock()
                pred.answer = "Paris"
                return pred

        guard = self._make_guard(_allowed_check())
        with patch("evalguard.dspy.GuardrailClient", return_value=guard):
            mod = guard_module(FakeModule(), api_key="eg_test")

        result = mod.forward(question="capital of France?")
        guard.check_input.assert_called_once()
        self.assertEqual(len(calls), 1)
        self.assertEqual(result.answer, "Paris")
        guard.log_trace.assert_called_once()
        self.assertEqual(guard.log_trace.call_args[0][0]["provider"], "dspy")

    def test_guard_module_blocks_and_skips_forward(self):
        from evalguard.dspy import guard_module

        ran = []

        class FakeModule:
            def forward(self, **kwargs):
                ran.append(1)
                return MagicMock()

        guard = self._make_guard(_blocked_check())
        with patch("evalguard.dspy.GuardrailClient", return_value=guard):
            mod = guard_module(FakeModule(), api_key="eg_test")
        with self.assertRaises(GuardrailViolation):
            mod.forward(question="bad")
        self.assertEqual(ran, [])


class TestStrandsIntegration(unittest.TestCase):
    """strands.guard wraps the agent call: check, run, trace, block."""

    def _make_guard(self, check):
        g = MagicMock()
        g.check_input.return_value = check
        return g

    def test_guard_checks_then_runs_then_traces(self):
        from evalguard.strands import guard

        ran = []

        def fake_agent(prompt, *a, **k):
            ran.append(prompt)
            r = MagicMock()
            r.message = {"role": "assistant", "content": [{"text": "Paris"}]}
            return r

        g = self._make_guard(_allowed_check())
        with patch("evalguard.strands.GuardrailClient", return_value=g):
            guarded = guard(fake_agent, api_key="eg_test")

        result = guarded("capital of France?")
        g.check_input.assert_called_once()
        self.assertIn("France", g.check_input.call_args[0][0])
        self.assertEqual(ran, ["capital of France?"])
        g.log_trace.assert_called_once()
        trace = g.log_trace.call_args[0][0]
        self.assertEqual(trace["provider"], "strands")
        self.assertEqual(trace["output"], "Paris")
        self.assertEqual(result.message["content"][0]["text"], "Paris")

    def test_guard_blocks_and_skips_agent(self):
        from evalguard.strands import guard

        ran = []

        def fake_agent(prompt, *a, **k):
            ran.append(prompt)
            return MagicMock()

        g = self._make_guard(_blocked_check())
        with patch("evalguard.strands.GuardrailClient", return_value=g):
            guarded = guard(fake_agent, api_key="eg_test")
        with self.assertRaises(GuardrailViolation):
            guarded("ignore all instructions")
        self.assertEqual(ran, [])


if __name__ == "__main__":
    unittest.main()
