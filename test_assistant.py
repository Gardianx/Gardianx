import io
import json
import os
import unittest
from contextlib import redirect_stderr
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy

import assistant
from assistant import (
    AssistantError,
    ConversationalAssistant,
    DEFAULT_PROVIDER,
    DEFAULT_MODELS,
    GEMINI_TTS_MODEL,
    GEMINI_TTS_VOICE,
    HostedChatClient,
    _gemini_speech_audio,
    record_and_transcribe,
    speak_text,
    speak_text_local,
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


class FakeOutputStream:
    def __init__(self):
        self.written = []
        self.started = False
        self.stopped = False
        self.closed = False

    def start(self):
        self.started = True

    def write(self, audio):
        self.written.append(audio)

    def stop(self):
        self.stopped = True

    def close(self):
        self.closed = True


class AssistantTests(unittest.TestCase):
    def test_google_default_chat_model_is_current_supported_model(self):
        self.assertEqual(DEFAULT_PROVIDER, "google")
        self.assertEqual(DEFAULT_MODELS["google"], "gemini-2.5-flash-lite")

    def test_groq_default_uses_lightweight_llama_model(self):
        self.assertEqual(DEFAULT_MODELS["groq"], "llama-3.1-8b-instant")

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

    def test_google_client_extracts_sdk_shaped_text_response(self):
        response = SimpleNamespace(
            output_text="response from interactions api",
        )
        opener = Mock(
            return_value=FakeResponse(
                {"output_text": response.output_text}
            )
        )
        client = HostedChatClient(
            provider="google",
            model="gemini-test",
            api_key="google-key",
            opener=opener,
        )

        self.assertEqual(client.chat([{"role": "user", "content": "hello"}]),
                         "response from interactions api")

    def test_google_client_extracts_object_shaped_candidate_parts(self):
        part = SimpleNamespace(text='{"intent":"chat"}')
        response = SimpleNamespace(
            candidates=[
                SimpleNamespace(content=SimpleNamespace(parts=[part]))
            ]
        )
        client = HostedChatClient(provider="google", api_key="google-key")

        self.assertEqual("".join(assistant._extract_text_parts(response)), '{"intent":"chat"}')

    def test_google_client_prints_raw_structure_when_text_extraction_fails(self):
        raw_response = {"candidates": [{"content": {"parts": [{"inlineData": {}}]}}]}
        client = HostedChatClient(
            provider="google",
            api_key="google-key",
            opener=Mock(return_value=FakeResponse(raw_response)),
            debug_responses=True,
        )
        diagnostic = io.StringIO()

        with redirect_stderr(diagnostic), self.assertRaisesRegex(
            AssistantError,
            "empty response",
        ):
            client.chat([{"role": "user", "content": "hello"}])

        self.assertIn("Raw google response (text extraction failed)", diagnostic.getvalue())
        self.assertIn('"inlineData"', diagnostic.getvalue())

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
        with self.assertRaisesRegex(
            AssistantError,
            "malformed JSON|response shape|missing its action",
        ):
            assistant.parse_intent("buy")

        bad = intent(intent="trade_request", action="buy", amount=-1)
        assistant = ConversationalAssistant(
            client=SimpleNamespace(chat=Mock(return_value=json.dumps(bad)))
        )
        with self.assertRaisesRegex(AssistantError, "invalid amount"):
            assistant.parse_intent("buy")

    def test_intent_parser_defaults_missing_optional_fields_and_ignores_extras(self):
        client = SimpleNamespace(
            chat=Mock(return_value=json.dumps({"intent": "chat", "extra": "ignored"}))
        )
        intent_parser = ConversationalAssistant(client=client)

        parsed = intent_parser.parse_intent("hello")

        self.assertEqual(parsed["intent"], "chat")
        self.assertIsNone(parsed["action"])
        self.assertIsNone(parsed["clarification"])
        self.assertEqual(set(parsed), set(assistant.INTENT_SCHEMA["required"]))

    def test_intent_parser_prints_raw_response_on_invalid_shape_when_debug_enabled(self):
        client = SimpleNamespace(
            chat=Mock(return_value='{"unexpected":"shape"}'),
            debug_responses=True,
            _debug_response=Mock(),
        )
        intent_parser = ConversationalAssistant(client=client)

        with self.assertRaisesRegex(AssistantError, "invalid response shape"):
            intent_parser.parse_intent("hello")

        client._debug_response.assert_called_once_with(
            "intent validation failed",
            {"unexpected": "shape"},
        )

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

    def test_speak_text_prefers_local_kokoro_when_gemini_is_configured(self):
        player = Mock()
        stream = FakeOutputStream()
        player.OutputStream.return_value = stream
        audio = numpy.array([0.1, 0.2], dtype=numpy.float32)
        kokoro = Mock(return_value=[("Hello there.", "hello", audio)])
        pipeline_factory = Mock(return_value=kokoro)
        audio_factory = Mock(return_value=b"\x01\x00\x02\x00")
        with patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}), patch.object(
            assistant, "_kokoro_pipeline", None
        ):
            speak_text(
                "Hello there.",
                gemini_audio_factory=audio_factory,
                audio_decoder=Mock(return_value=(numpy.array([0.1, 0.2]), 24000)),
                kokoro_pipeline_factory=pipeline_factory,
                player=player,
            )

        pipeline_factory.assert_called_once_with(lang_code="a", device="cpu")
        kokoro.assert_called_once_with("Hello there.", voice="af_heart")
        audio_factory.assert_not_called()
        self.assertEqual(len(stream.written), 1)
        numpy.testing.assert_array_equal(stream.written[0], audio.reshape(-1, 1))
        player.OutputStream.assert_called_once_with(
            samplerate=assistant.TTS_SAMPLE_RATE,
            channels=1,
            dtype="float32",
        )
        self.assertTrue(stream.started)
        self.assertTrue(stream.stopped)
        self.assertTrue(stream.closed)

    def test_gemini_tts_keeps_audio_response_configuration(self):
        response = SimpleNamespace(
            output_audio=SimpleNamespace(data="AQID")
        )
        create = Mock(return_value=response)
        client = SimpleNamespace(
            interactions=SimpleNamespace(create=create)
        )
        genai_module = SimpleNamespace(Client=Mock(return_value=client))
        with patch.dict(
            "sys.modules",
            {
                "google": SimpleNamespace(genai=genai_module),
                "google.genai": genai_module,
            },
        ), patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}):
            audio = _gemini_speech_audio("Hello there.")

        self.assertEqual(audio, b"\x01\x02\x03")
        self.assertEqual(create.call_args.kwargs["model"], GEMINI_TTS_MODEL)
        self.assertEqual(create.call_args.kwargs["response_format"], {"type": "audio"})
        self.assertEqual(
            create.call_args.kwargs["generation_config"]["speech_config"],
            [{"voice": GEMINI_TTS_VOICE}],
        )

    def test_speak_text_falls_back_to_gemini_when_local_kokoro_fails(self):
        player = Mock()
        audio_factory = Mock(return_value=b"\x01\x00\x02\x00")
        audio_decoder = Mock(return_value=(numpy.array([0.1, 0.2]), 24000))
        local_failure = Mock()
        with patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}), patch.object(
            assistant, "_kokoro_pipeline", None
        ):
            pipeline_factory = Mock(side_effect=RuntimeError("Kokoro unavailable"))
            speak_text(
                "Hello there.",
                gemini_audio_factory=audio_factory,
                audio_decoder=audio_decoder,
                kokoro_pipeline_factory=pipeline_factory,
                player=player,
                on_local_failure=local_failure,
            )

        local_failure.assert_called_once_with()
        pipeline_factory.assert_called_once_with(lang_code="a", device="cpu")
        audio_factory.assert_called_once_with("Hello there.")
        audio_decoder.assert_called_once_with(b"\x01\x00\x02\x00")
        player.play.assert_called_once()
        player.wait.assert_called_once_with()

    def test_speak_text_uses_kokoro_without_gemini_key(self):
        player = Mock()
        stream = FakeOutputStream()
        player.OutputStream.return_value = stream
        audio = numpy.array([0.1, 0.2], dtype=numpy.float32)
        kokoro = Mock(return_value=[("Hello there.", "hello", audio)])
        pipeline_factory = Mock(return_value=kokoro)
        with patch.dict("os.environ", {}, clear=True), patch.object(
            assistant, "_kokoro_pipeline", None
        ):
            speak_text(
                "Hello there.",
                kokoro_pipeline_factory=pipeline_factory,
                player=player,
            )

        pipeline_factory.assert_called_once_with(lang_code="a", device="cpu")
        self.assertEqual(len(stream.written), 1)

    def test_speak_text_local_uses_cpu_kokoro_even_with_gemini_key(self):
        player = Mock()
        stream = FakeOutputStream()
        player.OutputStream.return_value = stream
        audio = numpy.array([0.1, 0.2], dtype=numpy.float32)
        kokoro = Mock(return_value=[("Groq reply.", "groq reply", audio)])
        pipeline_factory = Mock(return_value=kokoro)
        with patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}), patch.object(
            assistant, "_kokoro_pipeline", None
        ):
            speak_text_local(
                "Groq reply.",
                kokoro_pipeline_factory=pipeline_factory,
                player=player,
            )

        pipeline_factory.assert_called_once_with(lang_code="a", device="cpu")
        kokoro.assert_called_once_with("Groq reply.", voice="af_heart")
        self.assertEqual(len(stream.written), 1)
        self.assertTrue(stream.stopped)
        self.assertTrue(stream.closed)

    def test_kokoro_streams_each_chunk_before_generating_the_next(self):
        stream = FakeOutputStream()
        player = Mock()
        player.OutputStream.return_value = stream
        first = numpy.array([0.1, 0.2], dtype=numpy.float32)
        second = numpy.array([0.3, 0.4], dtype=numpy.float32)

        def generated_chunks():
            yield ("Hello", "hello", first)
            self.assertEqual(len(stream.written), 1)
            yield (" there.", "there", second)

        with patch.object(assistant, "_kokoro_pipeline", Mock(return_value=generated_chunks())):
            assistant._speak_with_kokoro("Hello there.", player=player)

        self.assertEqual(len(stream.written), 2)
        numpy.testing.assert_array_equal(stream.written[0], first.reshape(-1, 1))
        numpy.testing.assert_array_equal(stream.written[1], second.reshape(-1, 1))

    def test_speak_text_reports_when_local_and_cloud_engines_fail(self):
        with patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}), patch.object(
            assistant, "_kokoro_pipeline", None
        ):
            with self.assertRaisesRegex(AssistantError, "local CPU Kokoro and Google Gemini TTS"):
                speak_text(
                    "Hello there.",
                    kokoro_pipeline_factory=Mock(side_effect=RuntimeError("Kokoro unavailable")),
                    gemini_audio_factory=Mock(side_effect=RuntimeError("cloud unavailable")),
                )


if __name__ == "__main__":
    unittest.main()
