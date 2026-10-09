"""Interactive hosted text assistant with local voice transcription."""

import argparse
import json
import math
import re
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

from assistant import (
    DEFAULT_PROVIDER,
    DEFAULT_MODELS,
    AssistantError,
    ConversationalAssistant,
    HostedChatClient,
    record_and_transcribe,
    search_web,
    speak_text,
    transcribe_audio,
)

LOCAL_VOICE_FALLBACK_WARNING = (
    "⚠️ Local engine hitch. Falling back to Google Cloud voice..."
)


def _offline_response(text: str) -> str:
    normalized = re.sub(r"[^\w\s']", "", text.casefold()).strip()
    if normalized in {
        "hello",
        "hi",
        "hey",
        "hello gardian",
        "hi gardian",
        "hey gardian",
        "good morning",
        "good afternoon",
        "good evening",
    }:
        return "Hello! It's good to hear from you. What would you like help with?"
    if any(
        phrase in normalized
        for phrase in (
            "how was your day",
            "how is your day",
            "how's your day",
            "how are you",
            "how are you doing",
        )
    ):
        return (
            "I'm doing well, thanks for asking! I don't have a day like a person, "
            "but I'm here and ready to help. How is your day going?"
        )
    if normalized in {"thank you", "thanks", "thanks gardian"}:
        return "You're welcome! I'm here whenever you need help."
    if normalized in {"bye", "goodbye", "see you"}:
        return "Goodbye! Talk to you later."
    if normalized in {"help", "what can you do", "what can you help me with"}:
        return (
            "I can chat about simple greetings and help explain issues. "
            "I'm offline, so detailed questions need the Google connection."
        )
    return (
        f"I heard you ask: {text.strip()}. I'm offline right now, so I can't "
        "give a reliable detailed answer to that yet. Please try again when "
        "the Google connection is available."
    )


def _positive_float(value: str) -> float:
    try:
        number = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("duration must be a positive number") from error
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("duration must be a positive number")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--provider",
        choices=tuple(DEFAULT_MODELS),
        default=DEFAULT_PROVIDER,
        help="Hosted conversation provider (default: google).",
    )
    parser.add_argument(
        "--model",
        help="Provider model name (defaults to a configured model for the provider).",
    )
    parser.add_argument("--audio", type=Path, help="Transcribe one local audio file.")
    parser.add_argument(
        "--listen-seconds",
        type=_positive_float,
        help="Record and transcribe one utterance from the microphone.",
    )
    parser.add_argument(
        "--allow-web-search",
        action="store_true",
        help="Allow explicit web_search intents to send the query to DuckDuckGo.",
    )
    parser.add_argument(
        "--speak",
        action="store_true",
        help="Speak replies with local CPU Kokoro first and Gemini TTS as backup.",
    )
    return parser


def _handle_text(
    assistant: ConversationalAssistant,
    text: str,
    *,
    allow_web_search: bool,
    speak: bool = False,
    provider: str = DEFAULT_PROVIDER,
    report=print,
) -> None:
    def say(message: str) -> None:
        report(f"Assistant: {message}")
        if speak:
            speak_text(
                message,
                on_local_failure=lambda: report(LOCAL_VOICE_FALLBACK_WARNING),
            )

    def report_google_failure(user_text: str) -> None:
        response = _offline_response(user_text)
        report(f"Assistant: {response}")
        if speak:
            speak_text(
                response,
                on_local_failure=lambda: report(LOCAL_VOICE_FALLBACK_WARNING),
            )

    report(f"You: {text}")
    try:
        intent = assistant.parse_intent(text)
    except AssistantError:
        if provider != "google":
            raise
        report_google_failure(text)
        return

    def respond(prompt: str, *, fallback_input: str = text) -> None:
        try:
            response = assistant.respond(prompt)
        except AssistantError:
            if provider != "google":
                raise
            report_google_failure(fallback_input)
            return
        say(response)

    kind = intent["intent"]
    if kind == "chat":
        respond(text)
    elif kind == "status":
        say("I can help interpret a status, but this chat is not connected to the running monitor.")
    elif kind == "explain_issue":
        issue = intent["issue"] or text
        respond("Help me understand this issue: " + issue, fallback_input=issue)
    elif kind == "trade_request":
        report(
            "Trade request parsed (not executed): "
            + json.dumps(
                {
                    "action": intent["action"],
                    "asset": intent["asset"],
                    "amount": intent["amount"],
                    "expiration_seconds": intent["expiration_seconds"],
                },
                ensure_ascii=False,
            )
        )
        if intent["clarification"]:
            say(intent["clarification"])
        else:
            say("This is only a draft. The assistant will not place trades.")
    elif kind == "web_search":
        if not allow_web_search:
            say(
                "Online lookup is disabled. Restart with --allow-web-search to "
                "explicitly permit sending search queries online."
            )
            return
        results = search_web(intent["query"] or text)
        if not results:
            report("Assistant: The search returned no results.")
        else:
            for result in results:
                report(
                    f"{result['title']}\n{result['url']}\n{result['snippet']}"
                )
    elif intent["clarification"]:
        say(intent["clarification"])
    else:
        say("I couldn't confidently identify what you need. Could you rephrase?")


def run(
    args: argparse.Namespace,
    *,
    assistant_factory=ConversationalAssistant,
    report=print,
    input_fn=input,
) -> int:
    if args.audio is not None and args.listen_seconds is not None:
        raise ValueError("Choose --audio or --listen-seconds, not both.")
    client = HostedChatClient(
        provider=args.provider,
        model=args.model,
        debug_responses=True,
    )
    assistant = assistant_factory(client)
    if args.audio is not None:
        text = transcribe_audio(str(args.audio))
        _handle_text(
            assistant,
            text,
            allow_web_search=args.allow_web_search,
            speak=args.speak,
            provider=args.provider,
            report=report,
        )
        return 0
    if args.listen_seconds is not None:
        text = record_and_transcribe(args.listen_seconds)
        _handle_text(
            assistant,
            text,
            allow_web_search=args.allow_web_search,
            speak=args.speak,
            provider=args.provider,
            report=report,
        )
        return 0

    report(
        f"Assistant ready ({args.provider}/{client.model}). "
        "Type /voice [seconds] to speak or /quit to exit."
    )
    while True:
        try:
            text = input_fn("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            report("")
            return 0
        if text.lower() in {"/quit", "/exit"}:
            return 0
        if not text:
            continue
        if text.lower() == "/voice" or text.lower().startswith("/voice "):
            fields = text.split()
            if len(fields) > 2:
                report("Assistant error: use /voice or /voice <seconds>.")
                continue
            try:
                duration = _positive_float(fields[1]) if len(fields) == 2 else 5.0
                text = record_and_transcribe(duration)
            except (argparse.ArgumentTypeError, AssistantError) as error:
                report(f"Assistant error: {error}")
                continue
        try:
            _handle_text(
                assistant,
                text,
                allow_web_search=args.allow_web_search,
                speak=args.speak,
                provider=args.provider,
                report=report,
            )
        except AssistantError as error:
            report(f"Assistant error: {error}")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.model is not None and not args.model.strip():
        parser.error("--model must not be empty.")
    if args.audio is not None and not args.audio.is_file():
        parser.error(f"audio file not found: {args.audio}")
    try:
        return run(args)
    except AssistantError as error:
        parser.exit(2, f"Assistant error: {error}\n")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
