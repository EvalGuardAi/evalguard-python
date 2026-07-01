"""Tests for the agent-memory + voice-ML SDK methods (2026-06-12)."""

import unittest
from unittest.mock import patch, MagicMock

from evalguard import EvalGuardClient


def _ok(json_body, status=200):
    resp = MagicMock()
    resp.ok = True
    resp.status_code = status
    resp.json.return_value = json_body
    return resp


class TestAgentMemory(unittest.TestCase):
    def setUp(self):
        self.client = EvalGuardClient(api_key="eg_test_key123", base_url="https://evalguard.ai/api")

    @patch("evalguard.client.requests.Session.request")
    def test_remember_memory_facts(self, mock_request):
        mock_request.return_value = _ok({"written": ["likes tea"], "skipped": []}, 201)
        result = self.client.remember_memory("proj", "user1", facts=["likes tea"])
        self.assertEqual(result["written"], ["likes tea"])
        args, kwargs = mock_request.call_args
        self.assertEqual(args[0], "POST")
        self.assertIn("/v1/agent-memory", args[1])
        self.assertEqual(kwargs.get("json", {})["sessionKey"], "user1")
        self.assertEqual(kwargs.get("json", {})["facts"], ["likes tea"])

    def test_remember_memory_requires_facts_or_turns(self):
        with self.assertRaises(ValueError):
            self.client.remember_memory("proj", "user1")

    @patch("evalguard.client.requests.Session.request")
    def test_recall_memory(self, mock_request):
        mock_request.return_value = _ok({"semantic": [{"content": "likes tea", "score": 0.9}]})
        result = self.client.recall_memory("proj", "user1", query="tea", limit=3)
        self.assertEqual(result["semantic"][0]["content"], "likes tea")
        args, kwargs = mock_request.call_args
        self.assertEqual(args[0], "GET")
        params = kwargs.get("params", {})
        self.assertEqual(params["query"], "tea")
        self.assertEqual(params["limit"], 3)

    @patch("evalguard.client.requests.Session.request")
    def test_forget_memory(self, mock_request):
        mock_request.return_value = _ok({"forgotten": 2})
        result = self.client.forget_memory("proj", "user1")
        self.assertEqual(result["forgotten"], 2)
        args, _ = mock_request.call_args
        self.assertEqual(args[0], "DELETE")
        self.assertIn("/v1/agent-memory", args[1])


class TestVoiceMl(unittest.TestCase):
    def setUp(self):
        self.client = EvalGuardClient(api_key="eg_test_key123", base_url="https://evalguard.ai/api")

    @patch("evalguard.client.requests.Session.request")
    def test_transcribe_voice(self, mock_request):
        mock_request.return_value = _ok({"language": "en", "text": "hi", "words": [{"word": "hi"}]})
        result = self.client.transcribe_voice("proj", "d2F2", language="en")
        self.assertEqual(result["words"][0]["word"], "hi")
        args, kwargs = mock_request.call_args
        self.assertEqual(args[0], "POST")
        self.assertIn("/v1/voice/transcribe", args[1])
        self.assertEqual(kwargs.get("json", {})["audioBase64"], "d2F2")

    @patch("evalguard.client.requests.Session.request")
    def test_score_voice_deepfake(self, mock_request):
        mock_request.return_value = _ok({"probability": 0.97, "model": "x"})
        result = self.client.score_voice_deepfake("proj", "d2F2")
        self.assertAlmostEqual(result["probability"], 0.97)
        args, _ = mock_request.call_args
        self.assertIn("/v1/voice/deepfake-score", args[1])


class TestLanguage(unittest.TestCase):
    def setUp(self):
        self.client = EvalGuardClient(api_key="eg_test_key123", base_url="https://evalguard.ai/api")

    @patch("evalguard.client.requests.Session.request")
    def test_detect_language(self, mock_request):
        mock_request.return_value = _ok(
            {"iso6393": "fra", "iso6391": "fr", "name": "French", "confidence": 0.4, "reliable": True}
        )
        result = self.client.detect_language("proj", "Bonjour tout le monde")
        self.assertEqual(result["iso6391"], "fr")
        self.assertTrue(result["reliable"])
        args, _ = mock_request.call_args
        self.assertIn("/v1/language/detect", args[1])

    def test_detect_language_requires_text(self):
        with self.assertRaises(ValueError):
            self.client.detect_language("proj", "")


if __name__ == "__main__":
    unittest.main()
