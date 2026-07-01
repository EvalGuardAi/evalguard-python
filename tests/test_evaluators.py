"""Tests for the Evaluator Hub + CLHF calibration methods."""

import unittest
from unittest.mock import patch, MagicMock

from evalguard import EvalGuardClient


def _ok(body):
    r = MagicMock()
    r.ok = True
    r.status_code = 200
    r.json.return_value = body
    return r


class TestEvaluatorHub(unittest.TestCase):
    def setUp(self):
        self.client = EvalGuardClient(api_key="eg_test", base_url="https://evalguard.ai/api")

    @patch("evalguard.client.requests.Session.request")
    def test_list_evaluators(self, mock_request):
        mock_request.return_value = _ok([{"name": "faithfulness", "version": 2}])
        self.client.list_evaluators("proj-1", name="faithfulness")
        args, kwargs = mock_request.call_args
        self.assertEqual(args[0], "GET")
        self.assertIn("/v1/evaluators", args[1])
        self.assertEqual(kwargs["params"]["projectId"], "proj-1")
        self.assertEqual(kwargs["params"]["name"], "faithfulness")

    def test_list_evaluators_requires_project(self):
        with self.assertRaises(ValueError):
            self.client.list_evaluators("")

    @patch("evalguard.client.requests.Session.request")
    def test_create_evaluator(self, mock_request):
        mock_request.return_value = _ok({"id": "v1", "version": 1})
        self.client.create_evaluator("p", "faithfulness", {"kind": "llm-judge", "threshold": 0.7})
        args, kwargs = mock_request.call_args
        self.assertEqual(args[0], "POST")
        self.assertIn("/v1/evaluators", args[1])
        self.assertEqual(kwargs["json"]["definition"]["kind"], "llm-judge")
        self.assertEqual(kwargs["json"]["name"], "faithfulness")

    @patch("evalguard.client.requests.Session.request")
    def test_diff_evaluator_versions(self, mock_request):
        mock_request.return_value = _ok({"diff": {"changed": True}})
        self.client.diff_evaluator_versions("p", "faithfulness", 1, 2)
        args, kwargs = mock_request.call_args
        self.assertEqual(args[0], "POST")
        self.assertIn("/v1/evaluators/diff", args[1])
        self.assertEqual(kwargs["json"]["fromVersion"], 1)
        self.assertEqual(kwargs["json"]["toVersion"], 2)

    @patch("evalguard.client.requests.Session.request")
    def test_calibrate_scorer(self, mock_request):
        mock_request.return_value = _ok({"scorerId": None, "agreement": {"kappa": 0.8}})
        self.client.calibrate_scorer(pairs=[{"human": True, "machine": True}])
        args, kwargs = mock_request.call_args
        self.assertEqual(args[0], "POST")
        self.assertIn("/v1/scorers/calibrate", args[1])
        self.assertEqual(kwargs["json"]["pairs"][0]["human"], True)

    def test_calibrate_scorer_requires_data(self):
        with self.assertRaises(ValueError):
            self.client.calibrate_scorer()


if __name__ == "__main__":
    unittest.main()
