"""Contract tests for GuardrailClient against the real /api/v1/guardrails shape.

The framework-integration tests mock GuardrailClient.check_input wholesale, so
the actual HTTP request/response translation was never exercised. These pin it:
  - request body uses `text` + `projectId` (the route's schema), not `input`
  - the { success, data:{action, reasons} } envelope → {allowed, violations}
  - action block/flag/allow maps to allowed correctly
  - check_output targets the same endpoint (no 404 /guardrails/output)
  - fail_open swallows errors
"""

import unittest
from unittest.mock import MagicMock, patch

from evalguard.guardrails import GuardrailClient


def _resp(json_body, status=200):
    m = MagicMock()
    m.status_code = status
    m.json.return_value = json_body
    m.raise_for_status = MagicMock()
    return m


class TestGuardrailClientContract(unittest.TestCase):
    def setUp(self):
        self.client = GuardrailClient(api_key="eg_test_x", project_id="proj-1")

    @patch("evalguard.guardrails.requests.Session.post")
    def test_check_input_sends_text_and_projectId(self, mock_post):
        mock_post.return_value = _resp({"success": True, "data": {"action": "allow", "reasons": []}})
        self.client.check_input("hello world")
        _, kwargs = mock_post.call_args
        body = kwargs["json"]
        self.assertEqual(body["text"], "hello world")
        self.assertEqual(body["projectId"], "proj-1")
        self.assertNotIn("input", body)
        self.assertNotIn("project_id", body)
        # posts to the real endpoint
        self.assertTrue(mock_post.call_args.args[0].endswith("/api/v1/guardrails"))

    @patch("evalguard.guardrails.requests.Session.post")
    def test_block_translates_to_not_allowed(self, mock_post):
        mock_post.return_value = _resp(
            {"success": True, "data": {"action": "block", "reasons": [{"detail": "pii"}]}}
        )
        r = self.client.check_input("my ssn is 123-45-6789")
        self.assertFalse(r["allowed"])
        self.assertEqual(r["violations"], [{"detail": "pii"}])
        self.assertEqual(r["action"], "block")

    @patch("evalguard.guardrails.requests.Session.post")
    def test_flag_is_allowed_with_violations(self, mock_post):
        mock_post.return_value = _resp(
            {"success": True, "data": {"action": "flag", "reasons": [{"detail": "suspicious"}]}}
        )
        r = self.client.check_input("borderline")
        self.assertTrue(r["allowed"])  # flag != block
        self.assertEqual(r["violations"], [{"detail": "suspicious"}])

    @patch("evalguard.guardrails.requests.Session.post")
    def test_allow_translates_to_allowed(self, mock_post):
        mock_post.return_value = _resp({"success": True, "data": {"action": "allow", "reasons": []}})
        r = self.client.check_input("benign")
        self.assertTrue(r["allowed"])
        self.assertEqual(r["violations"], [])

    @patch("evalguard.guardrails.requests.Session.post")
    def test_check_output_uses_same_endpoint_with_text(self, mock_post):
        mock_post.return_value = _resp({"success": True, "data": {"action": "allow", "reasons": []}})
        self.client.check_output("model said something")
        url = mock_post.call_args.args[0]
        self.assertTrue(url.endswith("/api/v1/guardrails"))
        self.assertFalse(url.endswith("/output"))
        self.assertEqual(mock_post.call_args.kwargs["json"]["text"], "model said something")

    @patch("evalguard.guardrails.requests.Session.post")
    def test_fail_open_swallows_errors(self, mock_post):
        mock_post.side_effect = RuntimeError("network down")
        client = GuardrailClient(api_key="eg_test_x", fail_open=True)
        r = client.check_input("hi")
        self.assertTrue(r["allowed"])

    @patch("evalguard.guardrails.requests.Session.post")
    def test_fail_closed_raises(self, mock_post):
        mock_post.side_effect = RuntimeError("network down")
        client = GuardrailClient(api_key="eg_test_x", fail_open=False)
        with self.assertRaises(Exception):
            client.check_input("hi")


if __name__ == "__main__":
    unittest.main()
