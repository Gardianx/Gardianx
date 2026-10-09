import unittest
from unittest.mock import Mock, patch

import nlp_assistant
from assistant import AssistantError


class NlpAssistantSpeechTests(unittest.TestCase):
    def setUp(self):
        self.assistant = Mock()
        self.assistant.parse_intent.return_value = {
            "intent": "chat",
            "issue": None,
            "query": None,
            "clarification": None,
        }
        self.assistant.respond.return_value = "A spoken reply."

    def test_speech_uses_local_first_with_cloud_fallback_callback(self):
        with patch.object(nlp_assistant, "speak_text") as speech:
            nlp_assistant._handle_text(
                self.assistant,
                "Hello",
                allow_web_search=False,
                speak=True,
                provider="groq",
                report=Mock(),
            )

        speech.assert_called_once()
        self.assertEqual(speech.call_args.args, ("A spoken reply.",))
        self.assertIn("on_local_failure", speech.call_args.kwargs)

    def test_google_speech_uses_same_local_first_priority(self):
        report = Mock()
        with patch.object(nlp_assistant, "speak_text") as speech:
            nlp_assistant._handle_text(
                self.assistant,
                "Hello",
                allow_web_search=False,
                speak=True,
                provider="google",
                report=report,
            )

        speech.assert_called_once()
        self.assertEqual(speech.call_args.args, ("A spoken reply.",))
        speech.call_args.kwargs["on_local_failure"]()
        report.assert_called_with(nlp_assistant.LOCAL_VOICE_FALLBACK_WARNING)

    def test_offline_responses_handle_greeting_and_day_question(self):
        self.assertIn("Hello!", nlp_assistant._offline_response("hello"))
        day_reply = nlp_assistant._offline_response("how is your day going?")
        self.assertIn("thanks for asking", day_reply)
        self.assertIn("How is your day going?", day_reply)

    def test_offline_response_is_honest_for_unrecognized_questions(self):
        response = nlp_assistant._offline_response("What is the weather tomorrow?")

        self.assertIn("I heard you ask: What is the weather tomorrow?", response)
        self.assertIn("I'm offline", response)

    def test_google_intent_failure_answers_and_speaks_user_greeting(self):
        self.assistant.parse_intent.side_effect = AssistantError("HTTP 503")
        report = Mock()

        with patch.object(nlp_assistant, "speak_text") as speech:
            nlp_assistant._handle_text(
                self.assistant,
                "Hello",
                allow_web_search=False,
                speak=True,
                provider="google",
                report=report,
            )

        answer = nlp_assistant._offline_response("Hello")
        report.assert_any_call(f"Assistant: {answer}")
        speech.assert_called_once_with(
            answer,
            on_local_failure=unittest.mock.ANY,
        )
        self.assistant.respond.assert_not_called()

    def test_google_response_failure_answers_and_speaks_user_question(self):
        self.assistant.respond.side_effect = AssistantError("network timeout")
        report = Mock()

        with patch.object(nlp_assistant, "speak_text") as speech:
            nlp_assistant._handle_text(
                self.assistant,
                "Hello",
                allow_web_search=False,
                speak=True,
                provider="google",
                report=report,
            )

        answer = nlp_assistant._offline_response("Hello")
        report.assert_any_call(f"Assistant: {answer}")
        speech.assert_called_once_with(
            answer,
            on_local_failure=unittest.mock.ANY,
        )

    def test_google_failure_without_speech_still_answers_user(self):
        self.assistant.parse_intent.side_effect = AssistantError("HTTP 503")
        report = Mock()

        with patch.object(nlp_assistant, "speak_text") as speech:
            nlp_assistant._handle_text(
                self.assistant,
                "Hello",
                allow_web_search=False,
                provider="google",
                report=report,
            )

        report.assert_any_call(
            f"Assistant: {nlp_assistant._offline_response('Hello')}"
        )
        speech.assert_not_called()

    def test_cli_defaults_to_google(self):
        args = nlp_assistant.build_parser().parse_args([])

        self.assertEqual(args.provider, "google")


if __name__ == "__main__":
    unittest.main()
