import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from assistant import (
    AssistantError,
    ConversationalAssistant,
    HostedChatClient,
    record_and_transcribe,
    speak_text,
)


def intent(**overrides):
    data = {
        "intent": "chat",
        "action": None,
        "asset": None,
        "amount": None,
        "expiration_seconds": None,
        "issue": None,
        "query": None,
        "clarification": None,
    }
    data.update(overrides)
    return data


class FakeResponse:
    def __init__(self, data):
        self.data = json.dumps(data).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def read(self):
        return self.data


class AssistantTests(unittest.TestCase):
    def test_groq_client_sends_structured_intent_request(self):
        opener = Mock(
            return_value=FakeResponse(
                {"choices": [{"message": {"content": "ok"}}]}
            )
        )
        client = HostedChatClient(
            provider="groq",
            model="test-model",
            api_key="test-key",
            opener=opener,
        )

        self.assertEqual(
            client.chat(
                [{"role": "user", "content": "hello"}],
                output_format={"type": "object"},
            ),
            "ok",
        )
        request = opener.call_args.args[0]
        body = json.loads(request.data.decode("utf-8"))
        self.assertEqual(
            request.full_url,
            "https://api.groq.com/openai/v1/chat/completions",
        )
        self.assertEqual(request.get_header("Authorization"), "Bearer test-key")
        self.assertEqual(body["model"], "test-model")
        self.assertEqual(body["response_format"], {"type": "json_object"})
        self.assertFalse(body["stream"])

    def test_google_ai_studio_client_maps_chat_to_generate_content(self):
        opener = Mock(
            return_value=FakeResponse(
                {
                    "candidates": [
                        {
                            "content": {
                                "parts": [{"text": '{"intent":"chat"}'}]
                            }
                        }
                    ]
                }
            )
        )
        client = HostedChatClient(
            provider="google",
            model="gemini-test",
            api_key="google-key",
            opener=opener,
        )

        content = client.chat(
            [
                {"role": "system", "content": "Be helpful."},
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "Hi."},
                {"role": "user", "content": "one more thing"},
            ],
            output_format={"type": "object"},
        )

        self.assertEqual(content, '{"intent":"chat"}')
        request = opener.call_args.args[0]
        self.assertIn("models/gemini-test:generateContent", request.full_url)
        self.assertNotIn("key=", request.full_url)
        self.assertEqual(request.headers.get("X-goog-api-key"), "google-key")
        body = json.loads(request.data.decode("utf-8"))
        self.assertEqual(body["systemInstruction"]["parts"][0]["text"], "Be helpful.")
        self.assertEqual(body["contents"][1]["role"], "model")
        self.assertEqual(
            body["generationConfig"]["responseMimeType"],
            "application/json",
        )

    def test_missing_api_key_has_provider_specific_actionable_error(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(AssistantError, "GROQ_API_KEY"):
                HostedChatClient(provider="groq")
            with self.assertRaisesRegex(AssistantError, "GOOGLE_API_KEY or GEMINI_API_KEY"):
                HostedChatClient(provider="google")

    def test_intent_parser_validates_and_preserves_structured_trade_draft(self):
        response = intent(
            intent="trade_request",
            action="buy",
            asset="EUR/USD",
            amount=0.01,
        )
        client = SimpleNamespace(chat=Mock(return_value=json.dumps(response)))
        assistant = ConversationalAssistant(client=client)

        parsed = assistant.parse_intent("Buy EUR/USD with 0.01 lots")

        self.assertEqual(parsed, response)
        self.assertEqual(client.chat.call_args.kwargs["output_format"]["required"][0], "intent")

    def test_rejects_malformed_or_unsafe_model_output(self):
        assistant = ConversationalAssistant(
            client=SimpleNamespace(chat=Mock(return_value='{"intent":"trade_request"}'))
        )
        with self.assertRaisesRegex(AssistantError, "malformed JSON|response shape"):
            assistant.parse_intent("buy")

        bad = intent(intent="trade_request", action="buy", amount=-1)
        assistant = ConversationalAssistant(
            client=SimpleNamespace(chat=Mock(return_value=json.dumps(bad)))
        )
        with self.assertRaisesRegex(AssistantError, "invalid amount"):
            assistant.parse_intent("buy")

    def test_chat_keeps_bounded_conversation_history(self):
        client = SimpleNamespace(chat=Mock(return_value="hello back"))
        assistant = ConversationalAssistant(client=client)
        for index in range(15):
            self.assertEqual(assistant.respond(f"message {index}"), "hello back")

        self.assertLessEqual(len(assistant._history), 21)
        self.assertEqual(assistant._history[0]["role"], "system")

    def test_record_and_transcribe_uses_cpu_int8_and_returns_transcript(self):
        recorder = Mock()
        audio = Mock()
        audio.reshape.return_value = "flattened-audio"
        recorder.rec.return_value = audio
        model = Mock()
        model.transcribe.return_value = ([SimpleNamespace(text=" turn left ")], None)
        factory = Mock(return_value=model)

        transcript = record_and_transcribe(
            1.0,
            model_factory=factory,
            recorder=recorder,
        )

        self.assertEqual(transcript, "turn left")
        factory.assert_called_once_with("small", device="cpu", compute_type="int8")
        recorder.wait.assert_called_once()
        model.transcribe.assert_called_once_with("flattened-audio", vad_filter=True)

    def test_speak_text_uses_local_speech_engine(self):
        engine = Mock()
        factory = Mock(return_value=engine)

        speak_text("Hello there.", engine_factory=factory)

        factory.assert_called_once_with()
        engine.say.assert_called_once_with("Hello there.")
        engine.runAndWait.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
