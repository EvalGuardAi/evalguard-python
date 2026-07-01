"""Tests for EvalGuard Python SDK tracing — secret redaction + ingest envelope.

Bugs class:
  - Sensitive arg captured verbatim → API key / password leaks via the span
    `inputs` field sent to /v1/traces/ingest.
  - Batcher posts the wrong envelope / wrong URL → no span ever lands.
"""

import json
import os
import unittest
from unittest.mock import MagicMock, patch

from evalguard.tracing import (
    Span,
    _safe_serialize,
    _is_secret_key,
    _looks_secret_value,
    _TraceBatcher,
    traceable,
    set_session,
    get_session,
    get_current_span,
    _current_session_id,
    _current_user_id,
    _current_conversation_id,
)


class TestSecretRedaction(unittest.TestCase):
    def test_secret_key_detection(self):
        for key in ("api_key", "apiKey", "password", "AUTHORIZATION", "access_key", "client_secret"):
            self.assertTrue(_is_secret_key(key), f"{key} should be a secret key")
        for key in ("username", "region", "model", "count"):
            self.assertFalse(_is_secret_key(key), f"{key} should NOT be a secret key")

    def test_value_shape_detection(self):
        self.assertTrue(_looks_secret_value("eg_live_abcdefgh12345678"))
        self.assertTrue(_looks_secret_value("sk-abcdefghijklmnop1234"))
        self.assertTrue(_looks_secret_value("sk-ant-abcdefghijklmnop1234"))
        self.assertTrue(_looks_secret_value("Bearer abcdefghijkl.mnop"))
        # Ordinary prose must NOT be masked.
        self.assertFalse(_looks_secret_value("hello world"))
        self.assertFalse(_looks_secret_value("the quick brown fox"))

    def test_redacts_by_key_name(self):
        out = _safe_serialize({"api_key": "plain-by-key", "password": 1234, "user": "alice"})
        self.assertEqual(out["api_key"], "[REDACTED]")
        self.assertEqual(out["password"], "[REDACTED]")
        self.assertEqual(out["user"], "alice")

    def test_redacts_by_value_shape(self):
        out = _safe_serialize({"creds": "eg_live_abcdefgh12345678", "note": "ordinary text"})
        # "creds" matches the credential key pattern AND the value is token-shaped.
        self.assertEqual(out["creds"], "[REDACTED]")
        self.assertEqual(out["note"], "ordinary text")

    def test_token_shaped_value_under_neutral_key(self):
        out = _safe_serialize({"blob": "sk-abcdefghijklmnop1234"})
        self.assertEqual(out["blob"], "[REDACTED]")

    def test_nested_redaction(self):
        out = _safe_serialize({"cfg": {"token": "abc", "list": ["sk-abcdefghijklmnop1234", "ok"]}})
        self.assertEqual(out["cfg"]["token"], "[REDACTED]")
        self.assertEqual(out["cfg"]["list"][0], "[REDACTED]")
        self.assertEqual(out["cfg"]["list"][1], "ok")

    def test_span_to_dict_redacts_inputs(self):
        span = Span(
            trace_id="t1",
            name="login",
            inputs={"username": "alice", "password": "hunter2"},
            metadata={"token": "sk-abcdefghijklmnop1234"},
        )
        d = span.to_dict()
        self.assertEqual(d["inputs"]["password"], "[REDACTED]")
        self.assertEqual(d["inputs"]["username"], "alice")
        self.assertEqual(d["metadata"]["token"], "[REDACTED]")


class TestIngestEnvelope(unittest.TestCase):
    def test_batcher_posts_envelope_to_ingest_url(self):
        os.environ["EVALGUARD_API_KEY"] = "eg_test_key"
        os.environ["EVALGUARD_BASE_URL"] = "https://api.evalguard.test"
        os.environ["EVALGUARD_PROJECT_ID"] = "proj-1"
        try:
            batcher = _TraceBatcher()
            mock_session = MagicMock()
            with patch.object(batcher, "_get_session", return_value=mock_session):
                batcher._send([{"spanId": "s1", "traceId": "t1", "name": "x"}])
            self.assertEqual(mock_session.post.call_count, 1)
            args, kwargs = mock_session.post.call_args
            self.assertEqual(args[0], "https://api.evalguard.test/v1/traces/ingest")
            payload = kwargs["json"]
            self.assertEqual(payload["projectId"], "proj-1")
            self.assertEqual(len(payload["spans"]), 1)
            self.assertEqual(payload["spans"][0]["spanId"], "s1")
            self.assertEqual(kwargs["headers"]["Authorization"], "Bearer eg_test_key")
        finally:
            for k in ("EVALGUARD_API_KEY", "EVALGUARD_BASE_URL", "EVALGUARD_PROJECT_ID"):
                os.environ.pop(k, None)

    def test_traceable_does_not_break_user_code_on_send_failure(self):
        # Telemetry must NEVER throw into user code even if the send path errors.
        @traceable
        def add(a, b):
            return a + b

        self.assertEqual(add(2, 3), 5)


class TestTraceIdentity(unittest.TestCase):
    """observability-tracing-3 — session/user/conversation identity."""

    def setUp(self):
        for cv in (_current_session_id, _current_user_id, _current_conversation_id):
            cv.set(None)

    def tearDown(self):
        for cv in (_current_session_id, _current_user_id, _current_conversation_id):
            cv.set(None)

    def test_set_and_get_session(self):
        set_session(session_id="s1", user_id="u1")
        self.assertEqual(
            get_session(),
            {"session_id": "s1", "user_id": "u1", "conversation_id": None},
        )

    def test_auto_attaches_dotted_attributes_to_spans(self):
        set_session(session_id="s1", user_id="u1", conversation_id="c1")
        captured = {}

        @traceable
        def work():
            captured["meta"] = dict(get_current_span().metadata)
            return 1

        work()
        self.assertEqual(captured["meta"].get("session.id"), "s1")
        self.assertEqual(captured["meta"].get("user.id"), "u1")
        self.assertEqual(captured["meta"].get("conversation.id"), "c1")

    def test_no_identity_means_no_keys(self):
        captured = {}

        @traceable
        def work():
            captured["meta"] = dict(get_current_span().metadata)
            return 1

        work()
        self.assertNotIn("session.id", captured["meta"])
        self.assertNotIn("user.id", captured["meta"])

    def test_child_span_inherits_identity(self):
        set_session(session_id="S")
        captured = {}

        @traceable
        def child():
            captured["meta"] = dict(get_current_span().metadata)
            return 1

        @traceable
        def parent():
            return child()

        parent()
        self.assertEqual(captured["meta"].get("session.id"), "S")

    def test_explicit_metadata_overrides_ambient(self):
        set_session(session_id="ambient")
        captured = {}

        @traceable(metadata={"session.id": "explicit"})
        def work():
            captured["meta"] = dict(get_current_span().metadata)
            return 1

        work()
        self.assertEqual(captured["meta"]["session.id"], "explicit")

    def test_dotted_session_id_survives_redaction(self):
        set_session(session_id="sess-123")
        captured = {}

        @traceable
        def work():
            captured["d"] = get_current_span().to_dict()
            return 1

        work()
        # If "session.id" matched the secret-key regex, this would be "[REDACTED]".
        self.assertEqual(captured["d"]["metadata"]["session.id"], "sess-123")


if __name__ == "__main__":
    unittest.main()
