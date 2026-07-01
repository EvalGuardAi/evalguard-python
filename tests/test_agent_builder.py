"""Tests for the agent-builder SDK methods: agent tools, abuse reports,
and agent deployments."""

import unittest
from unittest.mock import patch, MagicMock

from evalguard import EvalGuardClient


class TestAgentTools(unittest.TestCase):
    """Test the agent-tool CRUD + test methods."""

    def setUp(self):
        self.client = EvalGuardClient(
            api_key="eg_test_key123",
            base_url="https://evalguard.ai/api",
        )

    @patch("evalguard.client.requests.Session.request")
    def test_list_agent_tools(self, mock_request):
        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.status_code = 200
        mock_response.json.return_value = {"tools": [{"id": "t1"}, {"id": "t2"}]}
        mock_request.return_value = mock_response

        result = self.client.list_agent_tools("proj_abc")
        self.assertEqual(len(result["tools"]), 2)
        args, kwargs = mock_request.call_args
        self.assertEqual(args[0], "GET")
        self.assertIn("/v1/agent-tools", args[1])
        self.assertEqual(kwargs.get("params", {}).get("projectId"), "proj_abc")

    @patch("evalguard.client.requests.Session.request")
    def test_create_agent_tool(self, mock_request):
        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.status_code = 201
        mock_response.json.return_value = {"id": "t1", "name": "weather"}
        mock_request.return_value = mock_response

        tool = {
            "name": "weather",
            "description": "Get weather",
            "type": "rest",
            "parameters": {"type": "object", "properties": {}},
            "rest": {"method": "GET", "url": "https://api.example.com/weather"},
        }
        result = self.client.create_agent_tool("proj_abc", tool)
        self.assertEqual(result["id"], "t1")
        args, kwargs = mock_request.call_args
        self.assertEqual(args[0], "POST")
        self.assertIn("/v1/agent-tools", args[1])
        body = kwargs.get("json", {})
        self.assertEqual(body["projectId"], "proj_abc")
        self.assertEqual(body["tool"]["name"], "weather")

    @patch("evalguard.client.requests.Session.request")
    def test_get_agent_tool(self, mock_request):
        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.status_code = 200
        mock_response.json.return_value = {"id": "t1", "type": "code"}
        mock_request.return_value = mock_response

        result = self.client.get_agent_tool("t1", "proj_abc")
        self.assertEqual(result["id"], "t1")
        args, kwargs = mock_request.call_args
        self.assertEqual(args[0], "GET")
        self.assertIn("/v1/agent-tools/t1", args[1])
        self.assertEqual(kwargs.get("params", {}).get("projectId"), "proj_abc")

    @patch("evalguard.client.requests.Session.request")
    def test_update_agent_tool(self, mock_request):
        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.status_code = 200
        mock_response.json.return_value = {"id": "t1", "name": "renamed"}
        mock_request.return_value = mock_response

        result = self.client.update_agent_tool(
            "t1", "proj_abc", {"name": "renamed"}
        )
        self.assertEqual(result["name"], "renamed")
        args, kwargs = mock_request.call_args
        self.assertEqual(args[0], "PATCH")
        self.assertIn("/v1/agent-tools/t1", args[1])
        body = kwargs.get("json", {})
        self.assertEqual(body["projectId"], "proj_abc")
        self.assertEqual(body["tool"]["name"], "renamed")

    @patch("evalguard.client.requests.Session.request")
    def test_delete_agent_tool(self, mock_request):
        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.status_code = 200
        mock_response.json.return_value = {"id": "t1", "deleted": True}
        mock_request.return_value = mock_response

        result = self.client.delete_agent_tool("t1", "proj_abc")
        self.assertTrue(result["deleted"])
        args, _ = mock_request.call_args
        self.assertEqual(args[0], "DELETE")
        self.assertIn("/v1/agent-tools/t1", args[1])
        self.assertIn("projectId=proj_abc", args[1])

    @patch("evalguard.client.requests.Session.request")
    def test_test_agent_tool(self, mock_request):
        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.status_code = 200
        mock_response.json.return_value = {"ok": True, "stage": "execute", "status": 200}
        mock_request.return_value = mock_response

        result = self.client.test_agent_tool("t1", "proj_abc", {"city": "Paris"})
        self.assertTrue(result["ok"])
        self.assertEqual(result["stage"], "execute")
        args, kwargs = mock_request.call_args
        self.assertEqual(args[0], "POST")
        self.assertIn("/v1/agent-tools/t1/test", args[1])
        body = kwargs.get("json", {})
        self.assertEqual(body["projectId"], "proj_abc")
        self.assertEqual(body["args"], {"city": "Paris"})


class TestAbuseReports(unittest.TestCase):
    """Test the abuse-report intake methods."""

    def setUp(self):
        self.client = EvalGuardClient(
            api_key="eg_test_key123",
            base_url="https://evalguard.ai/api",
        )

    @patch("evalguard.client.requests.Session.request")
    def test_list_abuse_reports_no_status(self, mock_request):
        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.status_code = 200
        mock_response.json.return_value = {"reports": []}
        mock_request.return_value = mock_response

        result = self.client.list_abuse_reports("proj_abc")
        self.assertEqual(result["reports"], [])
        args, kwargs = mock_request.call_args
        self.assertEqual(args[0], "GET")
        self.assertIn("/v1/abuse-reports", args[1])
        params = kwargs.get("params", {})
        self.assertEqual(params.get("projectId"), "proj_abc")
        self.assertNotIn("status", params)

    @patch("evalguard.client.requests.Session.request")
    def test_list_abuse_reports_with_status(self, mock_request):
        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.status_code = 200
        mock_response.json.return_value = {"reports": [{"id": "r1"}]}
        mock_request.return_value = mock_response

        self.client.list_abuse_reports("proj_abc", status="open")
        _, kwargs = mock_request.call_args
        self.assertEqual(kwargs.get("params", {}).get("status"), "open")

    @patch("evalguard.client.requests.Session.request")
    def test_report_abuse_minimal(self, mock_request):
        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.status_code = 201
        mock_response.json.return_value = {
            "report": {"id": "r1", "category": "spam"},
            "triage": {
                "severity": "low",
                "category": "spam",
                "dedupKey": "abc",
                "autoEscalate": False,
                "feedToDetector": False,
                "reasons": [],
            },
        }
        mock_request.return_value = mock_response

        result = self.client.report_abuse("proj_abc", "spam")
        self.assertEqual(result["report"]["category"], "spam")
        self.assertEqual(result["triage"]["severity"], "low")
        args, kwargs = mock_request.call_args
        self.assertEqual(args[0], "POST")
        self.assertIn("/v1/abuse-reports", args[1])
        body = kwargs.get("json", {})
        self.assertEqual(body["projectId"], "proj_abc")
        self.assertEqual(body["category"], "spam")
        # Optional fields omitted when not provided.
        self.assertNotIn("description", body)
        self.assertNotIn("subjectId", body)
        self.assertNotIn("reporterId", body)
        self.assertNotIn("evidence", body)

    @patch("evalguard.client.requests.Session.request")
    def test_report_abuse_full(self, mock_request):
        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.status_code = 201
        mock_response.json.return_value = {"report": {"id": "r1"}, "triage": {}}
        mock_request.return_value = mock_response

        self.client.report_abuse(
            "proj_abc",
            "harassment",
            description="repeated abuse",
            subject_id="user_1",
            reporter_id="user_2",
            evidence={"url": "https://example.com/post/1"},
        )
        _, kwargs = mock_request.call_args
        body = kwargs.get("json", {})
        self.assertEqual(body["category"], "harassment")
        self.assertEqual(body["description"], "repeated abuse")
        self.assertEqual(body["subjectId"], "user_1")
        self.assertEqual(body["reporterId"], "user_2")
        self.assertEqual(body["evidence"], {"url": "https://example.com/post/1"})


class TestAgentDeployments(unittest.TestCase):
    """Test the workflow-deployment methods."""

    def setUp(self):
        self.client = EvalGuardClient(
            api_key="eg_test_key123",
            base_url="https://evalguard.ai/api",
        )

    @patch("evalguard.client.requests.Session.request")
    def test_list_agent_deployments(self, mock_request):
        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.status_code = 200
        mock_response.json.return_value = {"deployments": [{"id": "d1"}]}
        mock_request.return_value = mock_response

        result = self.client.list_agent_deployments("wf_1", "proj_abc")
        self.assertEqual(len(result["deployments"]), 1)
        args, kwargs = mock_request.call_args
        self.assertEqual(args[0], "GET")
        self.assertIn("/v1/workflows/wf_1/deploy", args[1])
        self.assertEqual(kwargs.get("params", {}).get("projectId"), "proj_abc")

    @patch("evalguard.client.requests.Session.request")
    def test_deploy_agent_minimal(self, mock_request):
        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.status_code = 201
        mock_response.json.return_value = {"id": "d1", "public_id": "pub_xyz"}
        mock_request.return_value = mock_response

        result = self.client.deploy_agent("wf_1", "proj_abc", "web")
        self.assertEqual(result["public_id"], "pub_xyz")
        args, kwargs = mock_request.call_args
        self.assertEqual(args[0], "POST")
        self.assertIn("/v1/workflows/wf_1/deploy", args[1])
        body = kwargs.get("json", {})
        self.assertEqual(body["projectId"], "proj_abc")
        self.assertEqual(body["channel"], "web")
        self.assertNotIn("allowedOrigins", body)
        self.assertNotIn("greeting", body)

    @patch("evalguard.client.requests.Session.request")
    def test_deploy_agent_full(self, mock_request):
        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.status_code = 201
        mock_response.json.return_value = {"id": "d1", "public_id": "pub_xyz"}
        mock_request.return_value = mock_response

        self.client.deploy_agent(
            "wf_1",
            "proj_abc",
            "web",
            allowed_origins=["https://app.example.com"],
            greeting="Hi there!",
        )
        _, kwargs = mock_request.call_args
        body = kwargs.get("json", {})
        self.assertEqual(body["allowedOrigins"], ["https://app.example.com"])
        self.assertEqual(body["greeting"], "Hi there!")

    @patch("evalguard.client.requests.Session.request")
    def test_update_agent_deployment(self, mock_request):
        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.status_code = 200
        mock_response.json.return_value = {"id": "d1", "status": "paused"}
        mock_request.return_value = mock_response

        result = self.client.update_agent_deployment(
            "d1", "proj_abc", status="paused"
        )
        self.assertEqual(result["status"], "paused")
        args, kwargs = mock_request.call_args
        self.assertEqual(args[0], "PATCH")
        self.assertIn("/v1/deployments/d1", args[1])
        body = kwargs.get("json", {})
        self.assertEqual(body["projectId"], "proj_abc")
        self.assertEqual(body["status"], "paused")
        self.assertNotIn("greeting", body)
        self.assertNotIn("allowedOrigins", body)

    @patch("evalguard.client.requests.Session.request")
    def test_delete_agent_deployment(self, mock_request):
        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.status_code = 200
        mock_response.json.return_value = {"id": "d1", "deleted": True}
        mock_request.return_value = mock_response

        result = self.client.delete_agent_deployment("d1", "proj_abc")
        self.assertTrue(result["deleted"])
        args, _ = mock_request.call_args
        self.assertEqual(args[0], "DELETE")
        self.assertIn("/v1/deployments/d1", args[1])
        self.assertIn("projectId=proj_abc", args[1])


if __name__ == "__main__":
    unittest.main()
