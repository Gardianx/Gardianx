"""FastAPI backend for the browser-based voice and chat assistant."""

import asyncio
from collections import defaultdict, deque
import hmac
import logging
import os
from pathlib import Path
import tempfile
import threading
import time
from typing import Literal

from fastapi import FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from assistant import (
    CHAT_SYSTEM_PROMPT,
    DEFAULT_MODELS,
    DEFAULT_PROVIDER,
    INTENT_SYSTEM_PROMPT,
    AssistantError,
    ConversationalAssistant,
    HostedChatClient,
    PROVIDER_ENV_VARS,
    search_web,
    transcribe_audio,
)


LOGGER = logging.getLogger("gardianx.api")
MAX_AUDIO_BYTES = 15 * 1024 * 1024
MAX_REQUESTS_PER_MINUTE = 30
MAX_HISTORY_MESSAGES = 20
AUDIO_SUFFIXES = {
    "audio/webm": ".webm",
    "audio/ogg": ".ogg",
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
    "audio/mpeg": ".mp3",
    "audio/mp4": ".m4a",
    "audio/aac": ".aac",
    "audio/flac": ".flac",
}

app = FastAPI(title="Gardianx Assistant API", version="1.0.0")
allowed_origins = [
    origin.strip()
    for origin in os.environ.get("CORS_ORIGINS", "").split(",")
    if origin.strip()
]
allowed_origins.extend(["http://localhost:5173", "http://127.0.0.1:5173"])
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(dict.fromkeys(allowed_origins)),
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-App-Password"],
)

_rate_lock = threading.Lock()
_request_times: dict[str, deque[float]] = defaultdict(deque)
_metrics_lock = threading.Lock()
_metrics = {
    "chat_requests": 0,
    "chat_errors": 0,
    "audio_transcriptions": 0,
    "audio_errors": 0,
    "chat_latency_total_ms": 0.0,
    "chat_latency_last_ms": 0.0,
    "transcription_latency_total_ms": 0.0,
    "transcription_latency_last_ms": 0.0,
}
_whisper_lock = threading.Lock()
_whisper_model = None


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=4000)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    provider: Literal["groq", "google"] = DEFAULT_PROVIDER
    conversation: list[ChatTurn] = Field(default_factory=list, max_length=MAX_HISTORY_MESSAGES)


def _check_rate_limit(request: Request) -> None:
    client_host = request.client.host if request.client else "unknown"
    now = time.monotonic()
    with _rate_lock:
        for host, history in list(_request_times.items()):
            while history and history[0] <= now - 60:
                history.popleft()
            if not history:
                _request_times.pop(host, None)
        timestamps = _request_times[client_host]
        if len(timestamps) >= MAX_REQUESTS_PER_MINUTE:
            raise HTTPException(status_code=429, detail="Request limit reached. Try again in a minute.")
        timestamps.append(now)


def _require_app_password(supplied_password: str | None) -> None:
    expected_password = os.environ.get("APP_ACCESS_PASSWORD", "")
    if not expected_password:
        raise HTTPException(
            status_code=503,
            detail="The app is not ready: set APP_ACCESS_PASSWORD in the backend environment.",
        )
    if not supplied_password or not hmac.compare_digest(supplied_password, expected_password):
        raise HTTPException(status_code=401, detail="Incorrect app password.")


def _provider_is_configured(provider: str) -> bool:
    return any(os.environ.get(name, "").strip() for name in PROVIDER_ENV_VARS[provider])


def _provider_model(provider: str) -> str:
    model_variable = "GROQ_MODEL" if provider == "groq" else "GOOGLE_MODEL"
    return os.environ.get(model_variable, "").strip() or DEFAULT_MODELS[provider]


def _get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        with _whisper_lock:
            if _whisper_model is None:
                try:
                    from faster_whisper import WhisperModel
                except ImportError as error:
                    raise AssistantError(
                        "Audio transcription is unavailable: faster-whisper is not installed."
                    ) from error
                model_name = os.environ.get("WHISPER_MODEL", "tiny").strip() or "tiny"
                _whisper_model = WhisperModel(
                    model_name,
                    device="cpu",
                    compute_type="int8",
                )
    return _whisper_model


def _update_metrics(name: str, elapsed_ms: float, *, failed: bool = False) -> None:
    with _metrics_lock:
        if name == "chat":
            _metrics["chat_requests"] += 1
            _metrics["chat_errors"] += int(failed)
            _metrics["chat_latency_total_ms"] += elapsed_ms
            _metrics["chat_latency_last_ms"] = elapsed_ms
        else:
            _metrics["audio_transcriptions"] += 1
            _metrics["audio_errors"] += int(failed)
            _metrics["transcription_latency_total_ms"] += elapsed_ms
            _metrics["transcription_latency_last_ms"] = elapsed_ms


def _provider_reply(
    message: str,
    provider: str,
    conversation: list[ChatTurn],
) -> tuple[str, dict]:
    client = HostedChatClient(provider=provider, model=_provider_model(provider))
    assistant = ConversationalAssistant(
        client,
        conversation=[
            {"role": turn.role, "content": turn.content}
            for turn in conversation[-MAX_HISTORY_MESSAGES:]
        ],
    )
    intent = assistant.parse_intent(message)
    kind = intent["intent"]

    if kind == "status":
        reply = "I can help interpret a status, but this web app is not connected to the live trading monitor."
    elif kind == "trade_request":
        reply = "I understood that as a trade request, but this app never places trades."
        if intent["clarification"]:
            reply += " " + intent["clarification"]
        else:
            reply += (
                f" Draft: {intent['action'] or 'action not specified'}"
                f"; asset: {intent['asset'] or 'not specified'}"
                f"; amount: {intent['amount'] or 'not specified'}."
            )
    elif kind == "web_search":
        if os.environ.get("ALLOW_WEB_SEARCH", "").lower() not in {"1", "true", "yes"}:
            reply = "Online search is disabled on this app."
        else:
            results = search_web(intent["query"] or message)
            context = "\n".join(
                f"- {result['title']}: {result['snippet']} ({result['url']})"
                for result in results
            )
            reply = assistant.respond(
                "Answer the user's request using these search results. Say when they "
                "do not contain enough information.\n\n"
                f"User request: {message}\n\nSearch results:\n{context or 'No results.'}"
            )
    elif kind == "unknown" and intent["clarification"]:
        reply = intent["clarification"]
    elif kind == "explain_issue":
        issue = intent["issue"] or message
        reply = assistant.respond(f"Help explain this reported issue: {issue}")
    else:
        reply = assistant.respond(message)
    return reply, intent


@app.get("/api/health")
async def health() -> dict:
    provider = os.environ.get("AI_PROVIDER", DEFAULT_PROVIDER).strip().lower()
    if provider not in DEFAULT_MODELS:
        provider = DEFAULT_PROVIDER
    return {
        "status": "ok",
        "provider": provider,
        "model": _provider_model(provider),
        "provider_configured": _provider_is_configured(provider),
        "providers": {
            name: {
                "configured": _provider_is_configured(name),
                "model": _provider_model(name),
            }
            for name in DEFAULT_MODELS
        },
        "password_required": bool(os.environ.get("APP_ACCESS_PASSWORD")),
        "audio_transcription": True,
    }


@app.get("/api/performance")
async def performance(
    request: Request,
    x_app_password: str | None = Header(default=None),
) -> dict:
    _check_rate_limit(request)
    _require_app_password(x_app_password)
    with _metrics_lock:
        snapshot = dict(_metrics)
    chat_count = snapshot["chat_requests"]
    audio_count = snapshot["audio_transcriptions"]
    return {
        "chat_requests": chat_count,
        "chat_errors": snapshot["chat_errors"],
        "average_chat_latency_ms": (
            round(snapshot["chat_latency_total_ms"] / chat_count, 1)
            if chat_count
            else 0
        ),
        "last_chat_latency_ms": round(snapshot["chat_latency_last_ms"], 1),
        "audio_transcriptions": audio_count,
        "audio_errors": snapshot["audio_errors"],
        "average_transcription_latency_ms": (
            round(snapshot["transcription_latency_total_ms"] / audio_count, 1)
            if audio_count
            else 0
        ),
        "last_transcription_latency_ms": round(
            snapshot["transcription_latency_last_ms"], 1
        ),
        "scope": "This process only; resets when the backend restarts.",
    }


@app.post("/api/chat")
async def chat(
    payload: ChatRequest,
    request: Request,
    x_app_password: str | None = Header(default=None),
) -> dict:
    _check_rate_limit(request)
    _require_app_password(x_app_password)
    provider = payload.provider
    client_model = _provider_model(provider)
    started = time.perf_counter()
    failed = False
    try:
        reply, intent = await asyncio.to_thread(
            _provider_reply,
            payload.message,
            provider,
            payload.conversation,
        )
    except AssistantError as error:
        failed = True
        LOGGER.warning("Assistant request failed for provider %s: %s", provider, error)
        raise HTTPException(status_code=502, detail=str(error)) from error
    except Exception as error:
        failed = True
        LOGGER.exception("Unexpected assistant request failure.")
        raise HTTPException(status_code=500, detail="Assistant request failed unexpectedly.") from error
    finally:
        _update_metrics("chat", (time.perf_counter() - started) * 1000, failed=failed)

    elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
    return {
        "reply": reply,
        "intent": intent,
        "provider": provider,
        "model": client_model,
        "latency_ms": elapsed_ms,
        "estimated_tokens": {
            "input": max(
                1,
                (
                    len(INTENT_SYSTEM_PROMPT)
                    + len(CHAT_SYSTEM_PROMPT)
                    + (2 * len(payload.message))
                    + sum(len(turn.content) for turn in payload.conversation)
                    + 3
                )
                // 4,
            ),
            "output": max(1, (len(reply) + 3) // 4),
        },
    }


@app.post("/api/transcribe")
async def transcribe(
    request: Request,
    file: UploadFile = File(...),
    x_app_password: str | None = Header(default=None),
) -> dict:
    _check_rate_limit(request)
    _require_app_password(x_app_password)
    content_type = (file.content_type or "").split(";", maxsplit=1)[0].lower()
    suffix = AUDIO_SUFFIXES.get(content_type)
    if suffix is None:
        raise HTTPException(
            status_code=415,
            detail="Unsupported audio type. Use WAV, MP3, M4A, AAC, FLAC, OGG, or WebM.",
        )

    started = time.perf_counter()
    failed = False
    temp_path: Path | None = None
    try:
        audio_data = await file.read(MAX_AUDIO_BYTES + 1)
        if len(audio_data) > MAX_AUDIO_BYTES:
            raise HTTPException(status_code=413, detail="Audio upload exceeds the 15 MB limit.")
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
            temp_file.write(audio_data)
            temp_path = Path(temp_file.name)
        transcript = await asyncio.to_thread(
            transcribe_audio,
            str(temp_path),
            model_name=os.environ.get("WHISPER_MODEL", "tiny"),
            model_factory=lambda *_args, **_kwargs: _get_whisper_model(),
        )
        return {
            "transcript": transcript,
            "latency_ms": round((time.perf_counter() - started) * 1000, 1),
        }
    except HTTPException:
        failed = True
        raise
    except AssistantError as error:
        failed = True
        LOGGER.warning("Audio transcription failed: %s", error)
        raise HTTPException(status_code=502, detail=str(error)) from error
    except Exception as error:
        failed = True
        LOGGER.exception("Unexpected audio transcription failure.")
        raise HTTPException(status_code=500, detail="Audio transcription failed unexpectedly.") from error
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        await file.close()
        _update_metrics(
            "audio",
            (time.perf_counter() - started) * 1000,
            failed=failed,
        )
