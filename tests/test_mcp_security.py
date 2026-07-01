"""Tests for the MCP/agent-security + A2A-graph SDK methods (2026-06-12)."""

import unittest
from unittest.mock import patch, MagicMock

from evalguard import EvalGuardClient


def _ok(json_body, status=200):
    resp = MagicMock()
    resp.ok = True
    resp.status_code = status
    resp.json.return_value = json_body
    return resp


class TestMcpAgentSecurity(unittest.TestCase):
    def setUp(self):
        self.client = EvalGuardClient(api_key="eg_test_key123", base_url="https://evalguard.ai/api")

    @patch("evalguard.client.requests.Session.request")
    def test_audit_mcp_server(self, mock_request):
        mock_request.return_value = _ok({"verdict": "block", "riskScore": 60, "findings": [], "summary": {"critical": 1}})
        result = self.client.audit_mcp_server("proj", {"id": "s", "authSchemes": []}, tools=[{"name": "x"}])
        self.assertEqual(result["verdict"], "block")
        args, kwargs = mock_request.call_args
        self.assertEqual(args[0], "POST")
        self.assertIn("/v1/security/mcp-predeployment-audit", args[1])
        self.assertEqual(kwargs.get("json", {})["server"]["id"], "s")

    def test_audit_mcp_server_requires_server(self):
        with self.assertRaises(ValueError):
            self.client.audit_mcp_server("proj", {})

    @patch("evalguard.client.requests.Session.request")
    def test_run_agent_exec_redteam(self, mock_request):
        mock_request.return_value = _ok({"verdict": "breached", "breaches": 1, "totalAttacks": 5})
        result = self.client.run_agent_exec_redteam("proj", "openai", "gpt-4o-mini")
        self.assertEqual(result["verdict"], "breached")
        args, kwargs = mock_request.call_args
        self.assertIn("/v1/security/agent-exec-redteam", args[1])
        self.assertEqual(kwargs.get("json", {})["target_provider"], "openai")

    def test_run_agent_exec_redteam_requires_target(self):
        with self.assertRaises(ValueError):
            self.client.run_agent_exec_redteam("proj", "", "")

    @patch("evalguard.client.requests.Session.request")
    def test_get_agent_graph(self, mock_request):
        mock_request.return_value = _ok({"services": ["a", "b"], "edges": [], "totalCalls": 0, "totalErrors": 0})
        result = self.client.get_agent_graph("proj", window_hours=168)
        self.assertEqual(result["services"], ["a", "b"])
        args, kwargs = mock_request.call_args
        self.assertEqual(args[0], "GET")
        self.assertIn("/v1/traces/graph", args[1])
        self.assertEqual(kwargs.get("params", {})["windowHours"], 168)


if __name__ == "__main__":
    unittest.main()
