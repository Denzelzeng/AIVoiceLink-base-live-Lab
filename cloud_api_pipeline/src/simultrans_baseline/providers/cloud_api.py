from __future__ import annotations

import asyncio
import base64
import io
import json
import uuid
import wave
from collections.abc import AsyncIterator, Sequence
from typing import Any
from urllib.parse import urlencode, urlsplit, urlunsplit

import httpx
import websockets

from ..audio import pcm_to_wav
from ..config import ServiceConfig, TTSConfig
from ..contracts import TextDeltaHandler
from ..events import AudioChunk, AudioWindow, VoiceProfile


class ProviderError(RuntimeError):
    """A remote model API returned an invalid or failed response."""


_LANGUAGE_CODES = {
    "chinese": "zh",
    "mandarin": "zh",
    "中文": "zh",
    "普通话": "zh",
    "cantonese": "yue",
    "粤语": "yue",
    "english": "en",
    "英语": "en",
    "japanese": "ja",
    "日语": "ja",
    "german": "de",
    "德语": "de",
    "korean": "ko",
    "韩语": "ko",
    "russian": "ru",
    "俄语": "ru",
    "french": "fr",
    "法语": "fr",
    "portuguese": "pt",
    "葡萄牙语": "pt",
    "italian": "it",
    "意大利语": "it",
    "spanish": "es",
    "西班牙语": "es",
}


def _headers(api_key: str | None, *, sse: bool = False) -> dict[str, str]:
    result = {"Accept": "text/event-stream" if sse else "application/json"}
    if api_key:
        result["Authorization"] = f"Bearer {api_key}"
    if sse:
        result["X-DashScope-SSE"] = "enable"
    return result


def _timeout(seconds: float) -> httpx.Timeout:
    return httpx.Timeout(
        connect=min(10.0, seconds), read=seconds, write=seconds, pool=10.0
    )


def _language_code(language: str) -> str | None:
    value = language.strip().casefold()
    if value in {"", "auto", "automatic", "auto-detect"}:
        return None
    return _LANGUAGE_CODES.get(value, value if len(value) in {2, 3} else None)


def _raise(response: httpx.Response, label: str) -> None:
    if response.status_code < 400:
        return
    raise ProviderError(
        f"{label} failed with HTTP {response.status_code}: {response.text[:2000]}"
    )


def _choice_text(payload: object, label: str) -> str:
    try:
        if not isinstance(payload, dict):
            raise TypeError
        choices = payload["choices"]
        message = choices[0]["message"]
        value = message["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ProviderError(f"invalid {label} response") from exc
    if not isinstance(value, str):
        raise ProviderError(f"invalid {label} text field")
    return value.strip()


async def _call_handler(handler: TextDeltaHandler | None, text: str) -> None:
    if not handler or not text:
        return
    result = handler(text)
    if asyncio.iscoroutine(result):
        await result


class DashScopeASR:
    """Qwen3-ASR cloud API over cumulative acoustic-turn WAV windows."""

    def __init__(
        self,
        config: ServiceConfig,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        self.config = config
        self._http = httpx.AsyncClient(
            headers=_headers(config.api_key()),
            timeout=_timeout(config.timeout_seconds),
            transport=transport,
        )

    async def transcribe(self, window: AudioWindow, *, language: str) -> str:
        wav = pcm_to_wav(window.pcm, sample_rate=window.sample_rate)
        data_uri = "data:audio/wav;base64," + base64.b64encode(wav).decode("ascii")
        options: dict[str, object] = {"enable_itn": True}
        if self.config.send_language and (code := _language_code(language)):
            options["language"] = code
        payload = {
            "model": self.config.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_audio",
                            "input_audio": {"data": data_uri},
                        }
                    ],
                }
            ],
            "stream": False,
            "asr_options": options,
        }
        response = await self._http.post(
            f"{self.config.normalized_base_url}/chat/completions", json=payload
        )
        _raise(response, "ASR")
        try:
            payload = response.json()
        except ValueError as exc:
            raise ProviderError("ASR response is not JSON") from exc
        return _choice_text(payload, "ASR")

    async def health(self) -> dict[str, object]:
        return await _model_health(self._http, self.config)

    async def aclose(self) -> None:
        await self._http.aclose()


class QwenMTTranslator:
    """Qwen-MT cloud API used for black-box incremental re-translation."""

    def __init__(
        self,
        config: ServiceConfig,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        self.config = config
        self._http = httpx.AsyncClient(
            headers=_headers(config.api_key()),
            timeout=_timeout(config.timeout_seconds),
            transport=transport,
        )

    async def translate(
        self,
        source_text: str,
        *,
        source_language: str,
        target_language: str,
        committed_target: str,
        context: Sequence[tuple[str, str]],
        domain: str,
        on_delta: TextDeltaHandler | None = None,
    ) -> str:
        del committed_target, context
        source = source_language.strip()
        if source.casefold() in {"auto", "automatic", "auto-detect"}:
            source = "auto"
        options: dict[str, object] = {
            "source_lang": source,
            "target_lang": target_language.strip(),
        }
        if domain.strip():
            options["domains"] = domain.strip()
        payload = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": source_text}],
            "translation_options": options,
            "stream": True,
        }
        url = f"{self.config.normalized_base_url}/chat/completions"
        async with self._http.stream("POST", url, json=payload) as response:
            if response.status_code >= 400:
                body = (await response.aread()).decode("utf-8", errors="replace")
                raise ProviderError(
                    f"translation failed with HTTP {response.status_code}: {body[:2000]}"
                )
            content_type = response.headers.get("content-type", "").casefold()
            if "text/event-stream" not in content_type:
                try:
                    text = _choice_text(response.json(), "translation")
                except ValueError as exc:
                    raise ProviderError("translation response is not JSON") from exc
                await _call_handler(on_delta, text)
                return text

            cumulative_model = any(
                marker in self.config.model.casefold() for marker in ("plus", "turbo")
            )
            hypothesis = ""
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                value = line[5:].strip()
                if not value or value == "[DONE]":
                    continue
                try:
                    event = json.loads(value)
                    choices = event.get("choices", [])
                    piece = choices[0].get("delta", {}).get("content", "") if choices else ""
                except (ValueError, AttributeError, IndexError) as exc:
                    raise ProviderError(
                        f"invalid translation SSE event: {value[:500]}"
                    ) from exc
                if not isinstance(piece, str) or not piece:
                    continue
                if cumulative_model:
                    delta = piece[len(hypothesis) :] if piece.startswith(hypothesis) else piece
                    hypothesis = piece
                else:
                    delta = piece
                    hypothesis += piece
                await _call_handler(on_delta, delta)
            return hypothesis.strip()

    async def health(self) -> dict[str, object]:
        return await _model_health(self._http, self.config)

    async def aclose(self) -> None:
        await self._http.aclose()


class DashScopeVoiceCloneTTS:
    """Qwen cloud voice enrollment plus streaming cloned-voice TTS."""

    def __init__(
        self,
        config: TTSConfig,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        ws_connect=None,
    ):
        self.config = config
        self._http = httpx.AsyncClient(
            headers=_headers(config.api_key()),
            timeout=_timeout(config.timeout_seconds),
            transport=transport,
        )
        self._ws_connect = ws_connect or websockets.connect
        self._realtime_socket = None
        self._realtime_connection = None
        self._realtime_connect_lock = asyncio.Lock()
        self._realtime_synthesis_lock = asyncio.Lock()
        self._realtime_prepare_lock = asyncio.Lock()
        self._realtime_session_key: tuple[str, str, float] | None = None
        self._prepared_realtime_sessions: dict[
            tuple[str, str, float], tuple[object, object]
        ] = {}
        self._retired_realtime_tasks: set[asyncio.Task[None]] = set()

    async def warmup(self) -> None:
        """Open the reusable realtime socket while microphone capture starts."""
        if "-realtime" not in self.config.model:
            return
        try:
            await self._ensure_realtime_socket()
        except Exception:
            # Synthesis retries the connection and reports a normal TTS failure
            # if the provider is still unavailable.
            await self._close_realtime_socket(send_finish=False)

    async def prepare_voice(self, profile_id: str, *, language: str) -> None:
        """Make a cloned voice session ready before it becomes active."""
        if "-realtime" not in self.config.model:
            return
        session_key = self._session_key(profile_id, language)
        async with self._realtime_prepare_lock:
            if (
                session_key == self._realtime_session_key
                or session_key in self._prepared_realtime_sessions
            ):
                return

            # The startup socket has no active voice and can be configured in
            # place.  A refresh is different: open/configure a second socket
            # without taking the synthesis lock, so the old voice can keep
            # feeding PCM to the FIFO while the replacement handshakes.
            if self._realtime_session_key is None:
                async with self._realtime_synthesis_lock:
                    await self._prepare_realtime_session_locked(session_key)
                return

            prepared = await self._open_configured_realtime_session(session_key)
            previous = self._prepared_realtime_sessions.pop(session_key, None)
            self._prepared_realtime_sessions[session_key] = prepared
            if previous is not None:
                self._retire_realtime_session(*previous, send_finish=True)

    def _url(self, path: str) -> str:
        return f"{self.config.normalized_base_url}{path}"

    async def enroll(
        self,
        reference_pcm: bytes,
        *,
        sample_rate: int,
        transcript: str,
        language: str,
    ) -> VoiceProfile:
        if sample_rate < 24_000:
            raise ProviderError(
                "Qwen voice enrollment requires reference audio sampled at 24 kHz or higher"
            )
        wav = pcm_to_wav(reference_pcm, sample_rate=sample_rate)
        input_data: dict[str, object] = {
            "action": "create",
            "target_model": self.config.model,
            "preferred_name": self.config.preferred_name,
            "audio": {
                "data": "data:audio/wav;base64,"
                + base64.b64encode(wav).decode("ascii")
            },
        }
        if transcript.strip():
            input_data["text"] = transcript.strip()
        if code := _language_code(language):
            input_data["language"] = code
        response = await self._http.post(
            self._url(self.config.clone_path),
            json={"model": self.config.enrollment_model, "input": input_data},
        )
        _raise(response, "voice enrollment")
        try:
            output = response.json().get("output", {})
            profile_id = output.get("voice") or output.get("voice_id")
        except (ValueError, AttributeError) as exc:
            raise ProviderError("voice enrollment response is not JSON") from exc
        if not isinstance(profile_id, str) or not profile_id:
            raise ProviderError("voice enrollment response has no voice ID")
        reference_ms = round(len(reference_pcm) / 2 / sample_rate * 1_000)
        return VoiceProfile(profile_id=profile_id, reference_ms=reference_ms)

    async def synthesize(
        self,
        text: str,
        *,
        language: str,
        profile_id: str | None,
    ) -> AsyncIterator[AudioChunk]:
        voice = profile_id or self.config.fallback_voice
        if not voice:
            raise ProviderError("no cloned or fallback TTS voice is available")
        if "-realtime" in self.config.model:
            async for chunk in self._synthesize_realtime(
                text,
                language=language,
                voice=voice,
            ):
                yield chunk
            return
        input_data: dict[str, object] = {"text": text, "voice": voice}
        if language.strip():
            input_data["language_type"] = language.strip()
        payload = {"model": self.config.model, "input": input_data}
        async with self._http.stream(
            "POST",
            self._url(self.config.speech_path),
            headers=_headers(self.config.api_key(), sse=True),
            json=payload,
        ) as response:
            if response.status_code >= 400:
                body = (await response.aread()).decode("utf-8", errors="replace")
                raise ProviderError(
                    f"cloned TTS failed with HTTP {response.status_code}: {body[:2000]}"
                )
            content_type = response.headers.get("content-type", "").casefold()
            if "text/event-stream" not in content_type:
                body = await response.aread()
                async for chunk in self._decode_complete_tts(body):
                    yield chunk
                return
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                value = line[5:].strip()
                if not value or value == "[DONE]":
                    continue
                try:
                    event = json.loads(value)
                    encoded = event.get("output", {}).get("audio", {}).get("data", "")
                    pcm = base64.b64decode(encoded, validate=True) if encoded else b""
                except (ValueError, TypeError, AttributeError) as exc:
                    raise ProviderError("invalid TTS SSE audio event") from exc
                if len(pcm) % 2:
                    raise ProviderError("TTS returned an odd number of PCM bytes")
                if pcm:
                    yield AudioChunk(data=pcm, sample_rate=self.config.sample_rate)

    async def _synthesize_realtime(
        self,
        text: str,
        *,
        language: str,
        voice: str,
    ) -> AsyncIterator[AudioChunk]:
        emitted_audio = False
        for attempt in range(2):
            try:
                async for chunk in self._synthesize_realtime_once(
                    text,
                    language=language,
                    voice=voice,
                ):
                    emitted_audio = True
                    yield chunk
                return
            except Exception:
                await self._close_realtime_socket(send_finish=False)
                if attempt == 0 and not emitted_audio:
                    continue
                raise

    async def _synthesize_realtime_once(
        self,
        text: str,
        *,
        language: str,
        voice: str,
    ) -> AsyncIterator[AudioChunk]:
        timeout = self.config.timeout_seconds
        async with self._realtime_synthesis_lock:
            session_key = self._session_key(voice, language)
            socket = await self._prepare_realtime_session_locked(session_key)
            await socket.send(
                json.dumps(
                    {
                        "event_id": f"event_{uuid.uuid4().hex}",
                        "type": "input_text_buffer.append",
                        "text": text,
                    }
                )
            )
            await socket.send(
                json.dumps(
                    {
                        "event_id": f"event_{uuid.uuid4().hex}",
                        "type": "input_text_buffer.commit",
                    }
                )
            )
            while True:
                event = await self._receive_ws_event(socket, timeout)
                event_type = event.get("type")
                if event_type == "response.audio.delta":
                    encoded = event.get("delta", "")
                    try:
                        pcm = base64.b64decode(encoded, validate=True)
                    except (ValueError, TypeError) as exc:
                        raise ProviderError("invalid realtime TTS audio event") from exc
                    if len(pcm) % 2:
                        raise ProviderError(
                            "realtime TTS returned an odd number of PCM bytes"
                        )
                    if pcm:
                        yield AudioChunk(data=pcm, sample_rate=self.config.sample_rate)
                elif event_type == "response.done":
                    return

    def _session_key(self, voice: str, language: str) -> tuple[str, str, float]:
        return (voice, language or "Auto", self.config.speech_rate)

    async def _prepare_realtime_session_locked(
        self,
        session_key: tuple[str, str, float],
    ):
        voice, language, speech_rate = session_key
        # Qwen Realtime allows session.update only before synthesis starts.
        # A cloned voice change therefore needs a fresh WebSocket session.
        if (
            self._realtime_session_key is not None
            and session_key != self._realtime_session_key
        ):
            prepared = self._prepared_realtime_sessions.pop(session_key, None)
            if prepared is not None:
                old_socket = self._realtime_socket
                old_connection = self._realtime_connection
                self._realtime_socket, self._realtime_connection = prepared
                self._realtime_session_key = session_key
                if old_socket is not None or old_connection is not None:
                    self._retire_realtime_session(
                        old_socket,
                        old_connection,
                        send_finish=True,
                    )
                return self._realtime_socket
            await self._close_realtime_socket(send_finish=True)
        socket = await self._ensure_realtime_socket()
        if session_key == self._realtime_session_key:
            return socket
        await self._configure_realtime_socket(socket, session_key)
        self._realtime_session_key = session_key
        return socket

    async def _open_configured_realtime_session(
        self,
        session_key: tuple[str, str, float],
    ) -> tuple[object, object]:
        last_error: Exception | None = None
        for _ in range(2):
            socket = None
            connection = None
            try:
                socket, connection = await self._open_realtime_socket()
                await self._configure_realtime_socket(socket, session_key)
                return socket, connection
            except asyncio.CancelledError:
                if socket is not None or connection is not None:
                    await self._close_realtime_session(
                        socket, connection, send_finish=False
                    )
                raise
            except Exception as exc:
                last_error = exc
                if socket is not None or connection is not None:
                    await self._close_realtime_session(
                        socket, connection, send_finish=False
                    )
        assert last_error is not None
        raise last_error

    async def _configure_realtime_socket(
        self,
        socket,
        session_key: tuple[str, str, float],
    ) -> None:
        voice, language, speech_rate = session_key
        await socket.send(
            json.dumps(
                {
                    "event_id": f"event_{uuid.uuid4().hex}",
                    "type": "session.update",
                    "session": {
                        "voice": voice,
                        "mode": "commit",
                        "language_type": language,
                        "response_format": "pcm",
                        "sample_rate": self.config.sample_rate,
                        "speech_rate": speech_rate,
                    },
                }
            )
        )
        await self._wait_for_ws_event(
            socket,
            "session.updated",
            self.config.timeout_seconds,
        )

    async def _ensure_realtime_socket(self):
        if self._realtime_socket is not None:
            return self._realtime_socket
        async with self._realtime_connect_lock:
            if self._realtime_socket is not None:
                return self._realtime_socket
            socket, connection = await self._open_realtime_socket()
            self._realtime_connection = connection
            self._realtime_socket = socket
            return socket

    async def _open_realtime_socket(self) -> tuple[object, object]:
        key = self.config.api_key()
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        timeout = self.config.timeout_seconds
        connection = self._ws_connect(
            self._realtime_url(),
            additional_headers=headers,
            open_timeout=min(10.0, timeout),
            close_timeout=min(10.0, timeout),
        )
        try:
            socket = await connection.__aenter__()
            await self._wait_for_ws_event(socket, "session.created", timeout)
        except BaseException:
            try:
                await connection.__aexit__(None, None, None)
            except Exception:
                pass
            raise
        return socket, connection

    async def _close_realtime_socket(self, *, send_finish: bool) -> None:
        socket = self._realtime_socket
        connection = self._realtime_connection
        self._realtime_socket = None
        self._realtime_connection = None
        self._realtime_session_key = None
        await self._close_realtime_session(socket, connection, send_finish=send_finish)

    async def _close_realtime_session(
        self,
        socket,
        connection,
        *,
        send_finish: bool,
    ) -> None:
        if socket is not None and send_finish:
            try:
                await socket.send(
                    json.dumps(
                        {
                            "event_id": f"event_{uuid.uuid4().hex}",
                            "type": "session.finish",
                        }
                    )
                )
            except Exception:
                pass
        if connection is not None:
            try:
                await connection.__aexit__(None, None, None)
            except Exception:
                pass

    def _retire_realtime_session(
        self,
        socket,
        connection,
        *,
        send_finish: bool,
    ) -> None:
        task = asyncio.create_task(
            self._close_realtime_session(
                socket,
                connection,
                send_finish=send_finish,
            )
        )
        self._retired_realtime_tasks.add(task)
        task.add_done_callback(self._retired_realtime_tasks.discard)

    async def _wait_for_ws_event(self, socket, expected: str, timeout: float) -> dict:
        while True:
            event = await self._receive_ws_event(socket, timeout)
            if event.get("type") == expected:
                return event

    @staticmethod
    async def _receive_ws_event(socket, timeout: float) -> dict:
        try:
            raw = await asyncio.wait_for(socket.recv(), timeout=timeout)
            event = json.loads(raw)
        except asyncio.TimeoutError as exc:
            raise ProviderError("realtime TTS WebSocket timed out") from exc
        except (ValueError, TypeError) as exc:
            raise ProviderError("realtime TTS returned invalid JSON") from exc
        if not isinstance(event, dict):
            raise ProviderError("realtime TTS event must be a JSON object")
        if event.get("type") == "error":
            error = event.get("error", {})
            if isinstance(error, dict):
                detail = error.get("message") or error.get("code") or error
            else:
                detail = error
            raise ProviderError(f"realtime TTS failed: {detail}")
        return event

    def _realtime_url(self) -> str:
        parsed = urlsplit(self.config.normalized_websocket_url)
        query = parsed.query
        model_query = urlencode({"model": self.config.model})
        query = f"{query}&{model_query}" if query else model_query
        return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, query, ""))

    async def _decode_complete_tts(self, body: bytes) -> AsyncIterator[AudioChunk]:
        try:
            payload = json.loads(body)
            audio = payload.get("output", {}).get("audio", {})
        except (ValueError, AttributeError) as exc:
            raise ProviderError("TTS response is not valid JSON") from exc
        encoded = audio.get("data", "")
        if encoded:
            pcm = base64.b64decode(encoded, validate=True)
            yield AudioChunk(data=pcm, sample_rate=self.config.sample_rate)
            return
        url = audio.get("url", "")
        if not isinstance(url, str) or not url:
            raise ProviderError("TTS response has neither audio data nor URL")
        # Do not forward the API Authorization header to the signed object URL.
        async with httpx.AsyncClient(timeout=_timeout(self.config.timeout_seconds)) as client:
            response = await client.get(url)
        _raise(response, "TTS audio download")
        try:
            with wave.open(io.BytesIO(response.content), "rb") as stream:
                if stream.getnchannels() != 1 or stream.getsampwidth() != 2:
                    raise ProviderError("downloaded TTS WAV must be mono PCM16")
                pcm = stream.readframes(stream.getnframes())
                rate = stream.getframerate()
        except wave.Error as exc:
            raise ProviderError("downloaded TTS audio is not PCM WAV") from exc
        yield AudioChunk(data=pcm, sample_rate=rate)

    async def delete_profile(self, profile_id: str) -> None:
        response = await self._http.post(
            self._url(self.config.clone_path),
            json={
                "model": self.config.enrollment_model,
                "input": {"action": "delete", "voice": profile_id},
            },
        )
        _raise(response, "voice profile deletion")

    async def health(self) -> dict[str, object]:
        response = await self._http.post(
            self._url(self.config.clone_path),
            json={
                "model": self.config.enrollment_model,
                "input": {"action": "list", "page_index": 0, "page_size": 1},
            },
        )
        _raise(response, "voice API health")
        result: dict[str, object] = {
            "provider": "dashscope_qwen_voice_clone",
            "configured_model": self.config.model,
            "api_reachable": True,
        }
        if "-realtime" in self.config.model:
            key = self.config.api_key()
            headers = {"Authorization": f"Bearer {key}"} if key else {}
            timeout = min(10.0, self.config.timeout_seconds)
            async with self._ws_connect(
                self._realtime_url(),
                additional_headers=headers,
                open_timeout=timeout,
                close_timeout=timeout,
            ) as socket:
                await self._wait_for_ws_event(socket, "session.created", timeout)
            result["websocket_reachable"] = True
        return result

    async def aclose(self) -> None:
        await self._close_realtime_socket(send_finish=True)
        prepared = tuple(self._prepared_realtime_sessions.values())
        self._prepared_realtime_sessions.clear()
        if prepared:
            await asyncio.gather(
                *(
                    self._close_realtime_session(
                        socket,
                        connection,
                        send_finish=True,
                    )
                    for socket, connection in prepared
                ),
                return_exceptions=True,
            )
        if self._retired_realtime_tasks:
            await asyncio.gather(
                *tuple(self._retired_realtime_tasks),
                return_exceptions=True,
            )
        await self._http.aclose()


async def _model_health(
    http: httpx.AsyncClient, config: ServiceConfig
) -> dict[str, object]:
    response = await http.get(f"{config.normalized_base_url}/models")
    _raise(response, "model catalog")
    try:
        payload = response.json()
        models = [
            entry.get("id")
            for entry in payload.get("data", [])
            if isinstance(entry, dict)
        ]
    except (ValueError, AttributeError) as exc:
        raise ProviderError("/models response is not OpenAI-compatible JSON") from exc
    family = "asr" if "asr" in config.model.casefold() else "mt"
    similar = [
        model_id
        for model_id in models
        if isinstance(model_id, str) and family in model_id.casefold()
    ]
    return {
        "configured_model": config.model,
        "configured_model_available": config.model in models,
        "catalog_size": len(models),
        "similar_models": similar[:20],
    }
