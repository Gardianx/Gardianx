"""Hosted conversational APIs with local speech transcription."""

import base64
import io
import json
import math
import os
import threading
import urllib.error
import urllib.parse
import urllib.request
from typing import Any


DEFAULT_PROVIDER = "groq"
DEFAULT_MODELS = {
    "groq": "llama-3.3-70b-versatile",
    "google": "gemini-3.8-flash",
}
PROVIDER_ENV_VARS = {
    "groq": ("GROQ_API_KEY",),
    "google": ("GOOGLE_API_KEY", "GEMINI_API_KEY"),
}
MAX_HISTORY_MESSAGES = 20
GEMINI_TTS_MODEL = "gemini-3.8-flash-tts"
GEMINI_TTS_VOICE = "Kore"
TTS_SAMPLE_RATE = 24000
_kokoro_lock = threading.Lock()
_kokoro_pipeline = None

INTENT_SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {
            "type": "string",
            "enum": ["chat", "status", "explain_issue", "trade_request", "web_search", "unknown"],
        },
        "action": {
            "type": ["string", "null"],
            "enum": ["buy", "sell", "call", "put", None],
        },
        "asset": {"type": ["string", "null"]},
        "amount": {"type": ["number", "null"]},
        "expiration_seconds": {"type": ["integer", "null"]},
        "issue": {"type": ["string", "null"]},
        "query": {"type": ["string", "null"]},
        "clarification": {"type": ["string", "null"]},
    },
    "required": [
        "intent",
        "action",
        "asset",
        "amount",
        "expiration_seconds",
        "issue",
        "query",
        "clarification",
    ],
    "additionalProperties": False,
}

INTENT_SYSTEM_PROMPT = (
    "You are a careful intent parser for a desktop assistant. Understand natural "
    "language, typos, and speech transcription errors. Classify the user's latest "
    "message as chat, status, explain_issue, trade_request, web_search, or unknown. "
    "Extract only information actually stated; never invent values. A trade_request "
    "is only a draft for confirmation, never an instruction to execute a trade. "
    "Use explain_issue for a reported error or a request to diagnose an issue and "
    "put the user's description in issue. Use web_search only for an explicit "
    "request for current online information. Ask a concise clarification if a "
    "trade request is missing required details or the intent is unclear. Emit "
    "exactly the required JSON fields."
)

CHAT_SYSTEM_PROMPT = (
    "You are a helpful, honest desktop assistant. Respond conversationally and "
    "concisely. Explain errors in plain language, distinguish confirmed facts "
    "from possibilities, and request missing diagnostic details instead of "
    "claiming to have inspected the computer. You cannot execute trades or "
    "control the desktop."
)


class AssistantError(RuntimeError):
    """Raised when hosted inference or speech recognition fails."""


class HostedChatClient:
    """Small REST client for Groq Cloud or Google AI Studio."""

    def __init__(
        self,
        provider: str = DEFAULT_PROVIDER,
        model: str | None = None,
        api_key: str | None = None,
        *,
        timeout_seconds: float = 120.0,
        opener=urllib.request.urlopen,
    ) -> None:
        provider = provider.strip().lower()
        if provider not in DEFAULT_MODELS:
            raise ValueError(f"provider must be one of: {', '.join(DEFAULT_MODELS)}.")
        selected_model = model or DEFAULT_MODELS[provider]
        if not selected_model.strip():
            raise ValueError("model must not be empty.")
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be finite and positive.")
        self.provider = provider
        self.model = selected_model
        self.api_key = api_key or self._environment_api_key(provider)
        self.timeout_seconds = timeout_seconds
        self._opener = opener

    @staticmethod
    def _environment_api_key(provider: str) -> str:
        for variable in PROVIDER_ENV_VARS[provider]:
            value = os.environ.get(variable)
            if value and value.strip():
                return value.strip()
        variables = " or ".join(PROVIDER_ENV_VARS[provider])
        raise AssistantError(
            f"Set {variables} in your environment to use the {provider} API."
        )

    def _request(
        self,
        messages: list[dict[str, str]],
        output_format: Any,
    ) -> tuple[str, bytes, dict[str, str]]:
        if self.provider == "groq":
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            body: dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "stream": False,
                "temperature": 0.2,
            }
            if output_format is not None:
                body["response_format"] = {"type": "json_object"}
            return (
                "https://api.groq.com/openai/v1/chat/completions",
                json.dumps(body).encode("utf-8"),
                headers,
            )

        system_message = next(
            (message["content"] for message in messages if message["role"] == "system"),
            "",
        )
        contents = [
            {
                "role": "model" if message["role"] == "assistant" else "user",
                "parts": [{"text": message["content"]}],
            }
            for message in messages
            if message["role"] != "system"
        ]
        generation_config: dict[str, Any] = {"temperature": 0.2}
        if output_format is not None:
            generation_config["responseMimeType"] = "application/json"
        body = {
            "systemInstruction": {"parts": [{"text": system_message}]},
            "contents": contents,
            "generationConfig": generation_config,
        }
        model = urllib.parse.quote(self.model, safe="")
        return (
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
            json.dumps(body).encode("utf-8"),
            {
                "Content-Type": "application/json",
                "X-goog-api-key": self.api_key,
            },
        )

    def chat(self, messages: list[dict[str, str]], *, output_format: Any = None) -> str:
        url, data, headers = self._request(messages, output_format)
        request = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            with self._opener(request, timeout=self.timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            try:
                error_payload = json.loads(detail)
            except json.JSONDecodeError:
                error_message = detail[:500]
            else:
                error_data = error_payload.get("error", {})
                error_message = (
                    error_data.get("message", str(error_data))
                    if isinstance(error_data, dict)
                    else str(error_data)
                )
            raise AssistantError(
                f"{self.provider} API request failed (HTTP {error.code}): "
                f"{error_message or error.reason}"
            ) from error
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            raise AssistantError(
                f"Could not reach the {self.provider} API. Check your internet "
                "connection and API service."
            ) from error
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise AssistantError(f"{self.provider} returned an invalid response.") from error

        if not isinstance(payload, dict):
            raise AssistantError(f"{self.provider} returned an invalid response object.")
        if payload.get("error"):
            raise AssistantError(f"{self.provider} API error: {payload['error']}")
        if self.provider == "groq":
            choices = payload.get("choices")
            choice = choices[0] if isinstance(choices, list) and choices else {}
            message = choice.get("message") if isinstance(choice, dict) else None
            content = message.get("content") if isinstance(message, dict) else None
        else:
            candidates = payload.get("candidates")
            candidate = candidates[0] if isinstance(candidates, list) and candidates else {}
            response_content = candidate.get("content") if isinstance(candidate, dict) else None
            parts = response_content.get("parts") if isinstance(response_content, dict) else None
            content = (
                "".join(part.get("text", "") for part in parts if isinstance(part, dict))
                if isinstance(parts, list)
                else None
            )
        if not isinstance(content, str) or not content.strip():
            raise AssistantError(f"{self.provider} returned an empty response.")
        return content.strip()


class ConversationalAssistant:
    """Parse text into safe intent drafts and maintain short-lived chat context."""

    def __init__(
        self,
        client: HostedChatClient,
        conversation: list[dict[str, str]] | None = None,
    ) -> None:
        self.client = client
        self._history: list[dict[str, str]] = [{"role": "system", "content": CHAT_SYSTEM_PROMPT}]
        if conversation:
            self._history.extend(conversation[-MAX_HISTORY_MESSAGES:])

    def parse_intent(self, text: str) -> dict[str, Any]:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must not be empty.")
        content = self.client.chat(
            [
                {"role": "system", "content": INTENT_SYSTEM_PROMPT},
                {"role": "user", "content": text.strip()},
            ],
            output_format=INTENT_SCHEMA,
        )
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as error:
            raise AssistantError("Intent parser returned malformed JSON.") from error
        return self._validate_intent(parsed)

    @staticmethod
    def _validate_intent(value: Any) -> dict[str, Any]:
        if not isinstance(value, dict) or set(value) != set(INTENT_SCHEMA["required"]):
            raise AssistantError("Intent parser returned an invalid response shape.")
        if value["intent"] not in {
            "chat",
            "status",
            "explain_issue",
            "trade_request",
            "web_search",
            "unknown",
        }:
            raise AssistantError("Intent parser returned an unsupported intent.")
        if value["action"] not in {None, "buy", "sell", "call", "put"}:
            raise AssistantError("Intent parser returned an unsupported action.")
        amount = value["amount"]
        if amount is not None and (
            isinstance(amount, bool)
            or not isinstance(amount, (int, float))
            or not math.isfinite(amount)
            or amount <= 0
        ):
            raise AssistantError("Intent parser returned an invalid amount.")
        expiration = value["expiration_seconds"]
        if expiration is not None and (
            isinstance(expiration, bool)
            or not isinstance(expiration, int)
            or expiration <= 0
        ):
            raise AssistantError("Intent parser returned an invalid expiration.")
        for field in ("asset", "issue", "query", "clarification"):
            if value[field] is not None and not isinstance(value[field], str):
                raise AssistantError(f"Intent parser returned an invalid {field}.")
        if value["intent"] == "trade_request" and value["action"] is None:
            raise AssistantError("Trade request is missing its action.")
        return value

    def respond(self, text: str) -> str:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must not be empty.")
        self._history.append({"role": "user", "content": text.strip()})
        reply = self.client.chat(self._history)
        self._history.append({"role": "assistant", "content": reply})
        if len(self._history) > MAX_HISTORY_MESSAGES + 1:
            self._history = [self._history[0], *self._history[-MAX_HISTORY_MESSAGES:]]
        return reply


def transcribe_audio(
    audio_path: str,
    *,
    model_name: str = "small",
    device: str = "cpu",
    compute_type: str = "int8",
    model_factory=None,
) -> str:
    """Transcribe an audio file locally with faster-whisper."""
    if model_factory is None:
        try:
            from faster_whisper import WhisperModel
        except ImportError as error:
            raise AssistantError(
                "Voice input requires faster-whisper. Install "
                "`requirements-assistant.txt`."
            ) from error
        model_factory = WhisperModel
    try:
        model = model_factory(model_name, device=device, compute_type=compute_type)
        segments, _ = model.transcribe(audio_path, vad_filter=True)
        transcript = " ".join(segment.text.strip() for segment in segments).strip()
    except Exception as error:
        raise AssistantError(f"Could not transcribe audio file {audio_path!r}.") from error
    if not transcript:
        raise AssistantError("No speech was recognized in the audio.")
    return transcript


def record_and_transcribe(
    duration_seconds: float,
    *,
    model_name: str = "small",
    model_factory=None,
    recorder=None,
) -> str:
    """Record one microphone utterance and transcribe it locally."""
    if not math.isfinite(duration_seconds) or duration_seconds <= 0:
        raise ValueError("duration_seconds must be finite and positive.")
    if recorder is None:
        try:
            import sounddevice
        except ImportError as error:
            raise AssistantError(
                "Microphone input requires sounddevice and a working audio device. "
                "Install `requirements-assistant.txt`."
            ) from error
        recorder = sounddevice
    if model_factory is None:
        try:
            from faster_whisper import WhisperModel
        except ImportError as error:
            raise AssistantError(
                "Microphone input requires faster-whisper. "
                "Install `requirements-assistant.txt`."
            ) from error
        model_factory = WhisperModel

    try:
        audio = recorder.rec(
            int(duration_seconds * 16000),
            samplerate=16000,
            channels=1,
            dtype="float32",
        )
        recorder.wait()
        model = model_factory(
            model_name,
            device="cpu",
            compute_type="int8",
        )
        segments, _ = model.transcribe(audio.reshape(-1), vad_filter=True)
        transcript = " ".join(segment.text.strip() for segment in segments).strip()
    except Exception as error:
        raise AssistantError("Could not record or transcribe microphone input.") from error
    if not transcript:
        raise AssistantError("No speech was recognized from the microphone.")
    return transcript


def search_web(query: str, *, max_results: int = 5) -> list[dict[str, str]]:
    """Perform an explicit, optional web search; user queries leave the device."""
    if not query.strip():
        raise ValueError("query must not be empty.")
    try:
        from ddgs import DDGS
    except ImportError as error:
        raise AssistantError(
            "Online lookup requires ddgs. Install `requirements-assistant.txt`."
        ) from error
    try:
        results = DDGS().text(query, max_results=max_results)
        return [
            {
                "title": str(result.get("title", "")),
                "url": str(result.get("href", "")),
                "snippet": str(result.get("body", "")),
            }
            for result in results
        ]
    except Exception as error:
        raise AssistantError("Online lookup failed; check the internet connection.") from error


def _get_kokoro_pipeline(pipeline_factory=None):
    global _kokoro_pipeline
    if _kokoro_pipeline is None:
        with _kokoro_lock:
            if _kokoro_pipeline is None:
                if pipeline_factory is None:
                    try:
                        from kokoro import KPipeline
                    except ImportError as error:
                        raise AssistantError(
                            "Local speech requires Kokoro. Install "
                            "`requirements-assistant-voice.txt` with Python 3.12."
                        ) from error
                    pipeline_factory = KPipeline
                _kokoro_pipeline = pipeline_factory(lang_code="a", device="cpu")
    return _kokoro_pipeline


def _gemini_speech_audio(text: str) -> bytes:
    api_key = os.environ.get("GEMINI_API_KEY", "").strip() or os.environ.get(
        "GOOGLE_API_KEY", ""
    ).strip()
    if not api_key:
        raise AssistantError("Google Gemini TTS is not configured.")
    try:
        from google import genai
    except ImportError as error:
        raise AssistantError(
            "Google Gemini TTS requires google-genai. Install "
            "`requirements-assistant-voice.txt`."
        ) from error
    try:
        client = genai.Client(api_key=api_key)
        response = client.interactions.create(
            model=os.environ.get("GEMINI_TTS_MODEL", GEMINI_TTS_MODEL),
            input=[
                {
                    "type": "user_input",
                    "content": [{"type": "text", "text": text}],
                }
            ],
            response_format={"type": "audio"},
            generation_config={
                "speech_config": [{"voice": GEMINI_TTS_VOICE}],
            },
        )
        audio_data = response.output_audio.data
        return base64.b64decode(audio_data, validate=True)
    except Exception as error:
        raise AssistantError("Google Gemini speech generation failed.") from error


def _play_audio(audio, sample_rate: int = TTS_SAMPLE_RATE, *, player=None) -> None:
    if player is None:
        try:
            import sounddevice
        except ImportError as error:
            raise AssistantError(
                "Audio playback requires sounddevice. Install "
                "`requirements-assistant-voice.txt`."
            ) from error
        player = sounddevice
    player.play(audio, sample_rate)
    player.wait()


def _decode_audio(audio_bytes: bytes):
    try:
        import soundfile
    except ImportError as error:
        raise AssistantError(
            "Gemini audio playback requires soundfile. Install "
            "`requirements-assistant-voice.txt`."
        ) from error
    try:
        return soundfile.read(io.BytesIO(audio_bytes), dtype="float32")
    except Exception as error:
        raise AssistantError("Could not decode generated Gemini audio.") from error


def _speak_with_kokoro(text: str, *, pipeline_factory=None, player=None) -> None:
    try:
        import numpy
    except ImportError as error:
        raise AssistantError(
            "Kokoro playback requires numpy. Install `requirements-assistant-voice.txt`."
        ) from error
    pipeline = _get_kokoro_pipeline(pipeline_factory)
    chunks = [audio for _, _, audio in pipeline(text, voice="af_heart")]
    if not chunks:
        raise AssistantError("Kokoro did not generate any speech audio.")
    _play_audio(numpy.concatenate(chunks), player=player)


def speak_text(
    text: str,
    *,
    gemini_audio_factory=None,
    audio_decoder=None,
    kokoro_pipeline_factory=None,
    player=None,
) -> None:
    """Speak with Gemini TTS when configured, falling back to CPU-only Kokoro."""
    if not text.strip():
        return
    errors = []
    gemini_key = os.environ.get("GEMINI_API_KEY", "").strip() or os.environ.get(
        "GOOGLE_API_KEY", ""
    ).strip()
    if gemini_key:
        try:
            audio = (
                gemini_audio_factory(text)
                if gemini_audio_factory is not None
                else _gemini_speech_audio(text)
            )
            audio_data, sample_rate = (
                audio_decoder(audio)
                if audio_decoder is not None
                else _decode_audio(audio)
            )
            _play_audio(audio_data, sample_rate, player=player)
            return
        except Exception as error:
            errors.append(error)
    try:
        _speak_with_kokoro(
            text,
            pipeline_factory=kokoro_pipeline_factory,
            player=player,
        )
    except Exception as fallback_error:
        raise AssistantError(
            "Speech output failed: Gemini TTS and local CPU Kokoro are unavailable."
        ) from fallback_error
