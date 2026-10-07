import os
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

import api


class AssistantAPITests(unittest.TestCase):
    def setUp(self):
        api._request_times.clear()
        api._metrics.update(
            chat_requests=0,
            chat_errors=0,
            audio_transcriptions=0,
            audio_errors=0,
            chat_latency_total_ms=0.0,
            chat_latency_last_ms=0.0,
            transcription_latency_total_ms=0.0,
            transcription_latency_last_ms=0.0,
        )
        self.environment = patch.dict(
            os.environ,
            {
                "APP_ACCESS_PASSWORD": "test-passphrase",
                "AI_PROVIDER": "groq",
                "GROQ_API_KEY": "test-key",
                "GROQ_MODEL": "test-groq-model",
                "GOOGLE_API_KEY": "test-google-key",
                "GOOGLE_MODEL": "test-google-model",
            },
            clear=True,
        )
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.client = TestClient(api.app)

    def test_health_lists_providers_without_disclosing_secrets(self):
        response = self.client.get("/api/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["provider"], "groq")
        self.assertEqual(response.json()["providers"]["google"]["model"], "test-google-model")
        self.assertNotIn("test-key", response.text)
        self.assertNotIn("test-google-key", response.text)

    def test_chat_requires_password_and_returns_timing_metrics(self):
        payload = {
            "message": "hello",
            "provider": "google",
            "conversation": [],
        }
        self.assertEqual(self.client.post("/api/chat", json=payload).status_code, 401)

        with patch.object(api, "_provider_reply", return_value=("Hi!", {"intent": "chat"})) as reply:
            response = self.client.post(
                "/api/chat",
                json=payload,
                headers={"X-App-Password": "test-passphrase"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["reply"], "Hi!")
        self.assertEqual(response.json()["model"], "test-google-model")
        self.assertGreaterEqual(response.json()["latency_ms"], 0)
        self.assertEqual(reply.call_args.args[1], "google")

    def test_chat_rejects_invalid_provider_and_oversized_messages(self):
        headers = {"X-App-Password": "test-passphrase"}
        self.assertEqual(
            self.client.post(
                "/api/chat",
                json={"message": "hello", "provider": "unsupported"},
                headers=headers,
            ).status_code,
            422,
        )
        self.assertEqual(
            self.client.post(
                "/api/chat",
                json={"message": "x" * 4001},
                headers=headers,
            ).status_code,
            422,
        )

    def test_transcribe_rejects_unsupported_media_before_loading_model(self):
        response = self.client.post(
            "/api/transcribe",
            headers={"X-App-Password": "test-passphrase"},
            files={"file": ("notes.txt", b"not audio", "text/plain")},
        )

        self.assertEqual(response.status_code, 415)
        self.assertEqual(api._metrics["audio_transcriptions"], 0)

    def test_audio_transcription_returns_transcript_and_updates_metrics(self):
        with patch.object(api, "transcribe_audio", return_value="hello from audio"):
            response = self.client.post(
                "/api/transcribe",
                headers={"X-App-Password": "test-passphrase"},
                files={"file": ("clip.wav", b"sample", "audio/wav")},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["transcript"], "hello from audio")
        self.assertEqual(api._metrics["audio_transcriptions"], 1)

    def test_performance_endpoint_requires_password_and_reports_scope(self):
        self.assertEqual(self.client.get("/api/performance").status_code, 401)
        response = self.client.get(
            "/api/performance",
            headers={"X-App-Password": "test-passphrase"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn("resets", response.json()["scope"])

    def test_api_rate_limits_repeated_requests(self):
        with patch.object(api, "MAX_REQUESTS_PER_MINUTE", 1):
            headers = {"X-App-Password": "test-passphrase"}
            self.assertEqual(
                self.client.get("/api/performance", headers=headers).status_code,
                200,
            )
            response = self.client.get("/api/performance", headers=headers)

        self.assertEqual(response.status_code, 429)

    def test_chat_requires_app_password_configuration(self):
        with patch.dict(os.environ, {"APP_ACCESS_PASSWORD": ""}):
            response = self.client.get(
                "/api/performance",
                headers={"X-App-Password": "test-passphrase"},
            )

        self.assertEqual(response.status_code, 503)
        self.assertIn("APP_ACCESS_PASSWORD", response.json()["detail"])


if __name__ == "__main__":
    unittest.main()
