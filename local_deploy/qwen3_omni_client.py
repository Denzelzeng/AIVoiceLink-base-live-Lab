from __future__ import annotations

import asyncio
import base64
import io
import json
import math
import statistics
import time
import wave
from array import array
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import httpx


DeltaCallback = Callable[[str], None]


class Qwen3OmniError(RuntimeError):
    """Base error raised by the local Qwen3-Omni client."""


class Qwen3OmniHTTPError(Qwen3OmniError):
    """The model server rejected a request."""


class Qwen3OmniProtocolError(Qwen3OmniError):
    """The model server returned an unexpected response."""


@dataclass(frozen=True)
class ClientConfig:
    base_url: str = "http://172.26.63.11:8003/v1"
    model: str = "qwen3-omni"
    api_key: str | None = None
    source_language: str = "auto-detect"
    target_language: str = "English"
    max_tokens: int = 256
    temperature: float = 0.1
    request_timeout_seconds: float = 180.0

    def __post_init__(self) -> None:
        if not self.base_url.startswith(("http://", "https://")):
            raise ValueError("base_url must start with http:// or https://")
        if not self.model.strip():
            raise ValueError("model cannot be empty")
        if not self.target_language.strip():
            raise ValueError("target_language cannot be empty")
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be positive")

    @property
    def normalized_base_url(self) -> str:
        return self.base_url.rstrip("/")

    @property
    def server_root(self) -> str:
        base = self.normalized_base_url
        return base[:-3] if base.endswith("/v1") else base


@dataclass(frozen=True)
class ASRConfig:
    base_url: str = "http://172.26.63.11:8004/v1"
    model: str = "qwen3-asr"
    api_key: str | None = None
    max_tokens: int = 512
    request_timeout_seconds: float = 180.0

    def __post_init__(self) -> None:
        if not self.base_url.startswith(("http://", "https://")):
            raise ValueError("ASR base_url must start with http:// or https://")
        if not self.model.strip():
            raise ValueError("ASR model cannot be empty")

    @property
    def normalized_base_url(self) -> str:
        return self.base_url.rstrip("/")

    @property
    def server_root(self) -> str:
        base = self.normalized_base_url
        return base[:-3] if base.endswith("/v1") else base


@dataclass(frozen=True)
class MicrophoneConfig:
    sample_rate: int = 16_000
    channels: int = 1
    sample_width: int = 2
    frame_ms: int = 50
    energy_threshold: int = 350
    calibration_seconds: float = 1.0
    pre_roll_ms: int = 200
    min_speech_ms: int = 400
    end_silence_ms: int = 650
    max_segment_ms: int = 6_000
    max_pending_segments: int = 2

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if self.channels != 1 or self.sample_width != 2:
            raise ValueError("microphone input must be mono 16-bit PCM")
        if self.frame_ms <= 0 or 1000 % self.frame_ms:
            raise ValueError("frame_ms must be a positive divisor of 1000")
        if self.min_speech_ms > self.max_segment_ms:
            raise ValueError("min_speech_ms cannot exceed max_segment_ms")
        if self.max_pending_segments < 1:
            raise ValueError("max_pending_segments must be positive")

    @property
    def frames_per_buffer(self) -> int:
        return self.sample_rate * self.frame_ms // 1000


@dataclass(frozen=True)
class AudioSegment:
    sequence: int
    pcm: bytes


def pcm_rms(pcm: bytes) -> int:
    """Return the RMS amplitude of little-endian signed 16-bit PCM."""
    if not pcm:
        return 0
    if len(pcm) % 2:
        raise ValueError("16-bit PCM must contain an even number of bytes")
    samples = array("h")
    samples.frombytes(pcm)
    if not samples:
        return 0
    return int(math.sqrt(sum(sample * sample for sample in samples) / len(samples)))


def pcm_to_wav(
    pcm: bytes,
    *,
    sample_rate: int = 16_000,
    channels: int = 1,
    sample_width: int = 2,
) -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(sample_width)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)
    return output.getvalue()


def _content_to_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for part in content:
        if isinstance(part, str):
            parts.append(part)
        elif isinstance(part, dict) and isinstance(part.get("text"), str):
            parts.append(part["text"])
    return "".join(parts)


def _server_hint(body: str) -> str:
    lowered = body.lower()
    if "not a multimodal model" in lowered:
        return (
            " The server loaded this checkpoint as text-only. Redeploy a "
            "multimodal-capable Qwen3-Omni build/checkpoint or use the ASR backend."
        )
    if "install vllm[audio]" in lowered:
        return (
            " The ASR container is missing vLLM audio dependencies. Install the "
            "matching vllm[audio]/Qwen audio dependencies and restart the service."
        )
    return ""


async def _stream_chat_completion(
    http: httpx.AsyncClient,
    *,
    url: str,
    payload: dict[str, object],
    on_delta: DeltaCallback | None,
) -> str:
    chunks: list[str] = []
    async with http.stream("POST", url, json=payload) as response:
        if response.status_code >= 400:
            body = (await response.aread()).decode("utf-8", errors="replace")
            raise Qwen3OmniHTTPError(
                f"POST {url} failed with HTTP {response.status_code}: "
                f"{body[:2000]}{_server_hint(body)}"
            )

        content_type = response.headers.get("content-type", "").lower()
        if "text/event-stream" not in content_type:
            body = await response.aread()
            return Qwen3OmniClient._parse_json_response(body, on_delta)

        async for line in response.aiter_lines():
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if not data or data == "[DONE]":
                continue
            try:
                event = json.loads(data)
            except json.JSONDecodeError as exc:
                raise Qwen3OmniProtocolError(
                    f"Invalid SSE JSON from model server: {data[:500]}"
                ) from exc
            choices = event.get("choices", [])
            if not choices:
                continue
            delta = choices[0].get("delta", {})
            text = _content_to_text(delta.get("content"))
            if text:
                chunks.append(text)
                if on_delta:
                    on_delta(text)
    return "".join(chunks).strip()


class SpeechSegmenter:
    """Energy-based endpointing for short, ordered interpretation turns."""

    def __init__(self, config: MicrophoneConfig, *, energy_threshold: int | None = None):
        self.config = config
        self.energy_threshold = energy_threshold or config.energy_threshold
        pre_roll_frames = max(1, config.pre_roll_ms // config.frame_ms)
        self._pre_roll: deque[bytes] = deque(maxlen=pre_roll_frames)
        self._active = False
        self._frames: list[bytes] = []
        self._duration_ms = 0
        self._speech_ms = 0
        self._silence_ms = 0

    def add_frame(self, frame: bytes) -> bytes | None:
        voiced = pcm_rms(frame) >= self.energy_threshold

        if not self._active:
            self._pre_roll.append(frame)
            if voiced:
                self._active = True
                self._frames = list(self._pre_roll)
                self._pre_roll.clear()
                self._duration_ms = len(self._frames) * self.config.frame_ms
                self._speech_ms = self.config.frame_ms
                self._silence_ms = 0
            return None

        self._frames.append(frame)
        self._duration_ms += self.config.frame_ms
        if voiced:
            self._speech_ms += self.config.frame_ms
            self._silence_ms = 0
        else:
            self._silence_ms += self.config.frame_ms

        reached_endpoint = (
            self._silence_ms >= self.config.end_silence_ms
            and self._speech_ms >= self.config.min_speech_ms
        )
        reached_limit = self._duration_ms >= self.config.max_segment_ms
        short_noise = (
            self._silence_ms >= self.config.end_silence_ms
            and self._speech_ms < self.config.min_speech_ms
        )
        if reached_endpoint or reached_limit or short_noise:
            return self._finish(include_short=reached_limit)
        return None

    def flush(self) -> bytes | None:
        if not self._active:
            return None
        return self._finish(include_short=False)

    def _finish(self, *, include_short: bool) -> bytes | None:
        speech_ms = self._speech_ms
        frames = self._frames

        if self._silence_ms:
            keep_silence_ms = min(100, self._silence_ms)
            remove_count = (self._silence_ms - keep_silence_ms) // self.config.frame_ms
            if remove_count:
                frames = frames[:-remove_count]

        self._active = False
        self._frames = []
        self._duration_ms = 0
        self._speech_ms = 0
        self._silence_ms = 0
        self._pre_roll.clear()

        if speech_ms < self.config.min_speech_ms and not include_short:
            return None
        return b"".join(frames)


class Qwen3OmniClient:
    def __init__(
        self,
        config: ClientConfig,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        self.config = config
        headers = {"Accept": "text/event-stream"}
        if config.api_key:
            headers["Authorization"] = f"Bearer {config.api_key}"
        timeout = httpx.Timeout(
            connect=10.0,
            read=config.request_timeout_seconds,
            write=30.0,
            pool=10.0,
        )
        self._http = httpx.AsyncClient(
            headers=headers,
            timeout=timeout,
            transport=transport,
        )

    async def __aenter__(self) -> Qwen3OmniClient:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._http.aclose()

    async def check_server(self) -> dict[str, object]:
        health_response = await self._http.get(f"{self.config.server_root}/health")
        self._raise_for_status(health_response)

        models_response = await self._http.get(
            f"{self.config.normalized_base_url}/models"
        )
        self._raise_for_status(models_response)
        try:
            models_payload = models_response.json()
        except ValueError as exc:
            raise Qwen3OmniProtocolError("/v1/models did not return JSON") from exc

        model_ids = [
            entry.get("id")
            for entry in models_payload.get("data", [])
            if isinstance(entry, dict) and isinstance(entry.get("id"), str)
        ]
        return {
            "health_status": health_response.status_code,
            "models": model_ids,
            "configured_model_available": self.config.model in model_ids,
        }

    async def translate_pcm(
        self,
        pcm: bytes,
        *,
        previous_translation: str = "",
        on_delta: DeltaCallback | None = None,
        sample_rate: int = 16_000,
    ) -> str:
        wav_bytes = pcm_to_wav(pcm, sample_rate=sample_rate)
        return await self.translate_audio(
            wav_bytes,
            audio_format="wav",
            previous_translation=previous_translation,
            on_delta=on_delta,
        )

    async def translate_audio(
        self,
        audio_bytes: bytes,
        *,
        audio_format: str,
        previous_translation: str = "",
        on_delta: DeltaCallback | None = None,
    ) -> str:
        if not audio_bytes:
            raise ValueError("audio_bytes cannot be empty")

        context = previous_translation.strip()
        context_instruction = (
            f"\nPrevious translated context (do not repeat it): {context[-1000:]}"
            if context
            else ""
        )
        instruction = (
            f"Interpret the audible speech from {self.config.source_language} "
            f"into {self.config.target_language}. Output only the translation, "
            "with no transcript, labels, notes, or explanation. Preserve names, "
            "numbers, tone, and intent. The clip may begin or end mid-sentence. "
            "If there is no intelligible speech, output nothing."
            f"{context_instruction}"
        )
        payload = {
            "model": self.config.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a precise simultaneous interpreter. Respond only "
                        "with the requested translated speech."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_audio",
                            "input_audio": {
                                "data": base64.b64encode(audio_bytes).decode("ascii"),
                                "format": audio_format,
                            },
                        },
                        {"type": "text", "text": instruction},
                    ],
                },
            ],
            "stream": True,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
        }
        return await _stream_chat_completion(
            self._http,
            url=f"{self.config.normalized_base_url}/chat/completions",
            payload=payload,
            on_delta=on_delta,
        )

    async def translate_text(
        self,
        transcript: str,
        *,
        previous_translation: str = "",
        on_delta: DeltaCallback | None = None,
    ) -> str:
        transcript = transcript.strip()
        if not transcript:
            return ""
        context = previous_translation.strip()
        context_instruction = (
            f"\nPrevious translated context (do not repeat it): {context[-1000:]}"
            if context
            else ""
        )
        payload = {
            "model": self.config.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a precise simultaneous interpreter. Output only "
                        "the requested translation without notes or labels."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Translate this speech from {self.config.source_language} "
                        f"into {self.config.target_language}. Preserve names, "
                        f"numbers, tone, and intent:\n\n{transcript}"
                        f"{context_instruction}"
                    ),
                },
            ],
            "stream": True,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
        }
        return await _stream_chat_completion(
            self._http,
            url=f"{self.config.normalized_base_url}/chat/completions",
            payload=payload,
            on_delta=on_delta,
        )

    @staticmethod
    def _parse_json_response(body: bytes, on_delta: DeltaCallback | None) -> str:
        try:
            payload = json.loads(body)
            choices = payload.get("choices", [])
            message = choices[0].get("message", {}) if choices else {}
            text = _content_to_text(message.get("content")).strip()
        except (json.JSONDecodeError, AttributeError, IndexError) as exc:
            raise Qwen3OmniProtocolError(
                f"Unexpected model response: {body[:1000]!r}"
            ) from exc
        if text and on_delta:
            on_delta(text)
        return text

    @staticmethod
    def _raise_for_status(response: httpx.Response) -> None:
        if response.status_code < 400:
            return
        raise Qwen3OmniHTTPError(
            f"{response.request.method} {response.request.url} failed with "
            f"HTTP {response.status_code}: {response.text[:2000]}"
        )


class Qwen3ASRClient:
    def __init__(
        self,
        config: ASRConfig,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        self.config = config
        headers = {"Accept": "text/event-stream"}
        if config.api_key:
            headers["Authorization"] = f"Bearer {config.api_key}"
        timeout = httpx.Timeout(
            connect=10.0,
            read=config.request_timeout_seconds,
            write=30.0,
            pool=10.0,
        )
        self._http = httpx.AsyncClient(
            headers=headers,
            timeout=timeout,
            transport=transport,
        )

    async def __aenter__(self) -> Qwen3ASRClient:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._http.aclose()

    async def check_server(self) -> dict[str, object]:
        health_response = await self._http.get(f"{self.config.server_root}/health")
        Qwen3OmniClient._raise_for_status(health_response)
        models_response = await self._http.get(
            f"{self.config.normalized_base_url}/models"
        )
        Qwen3OmniClient._raise_for_status(models_response)
        try:
            payload = models_response.json()
        except ValueError as exc:
            raise Qwen3OmniProtocolError("ASR /v1/models did not return JSON") from exc
        model_ids = [
            entry.get("id")
            for entry in payload.get("data", [])
            if isinstance(entry, dict) and isinstance(entry.get("id"), str)
        ]
        return {
            "health_status": health_response.status_code,
            "models": model_ids,
            "configured_model_available": self.config.model in model_ids,
        }

    async def transcribe_pcm(
        self,
        pcm: bytes,
        *,
        sample_rate: int = 16_000,
    ) -> str:
        return await self.transcribe_audio(
            pcm_to_wav(pcm, sample_rate=sample_rate),
            audio_format="wav",
        )

    async def transcribe_audio(
        self,
        audio_bytes: bytes,
        *,
        audio_format: str,
    ) -> str:
        if not audio_bytes:
            raise ValueError("audio_bytes cannot be empty")
        mime_types = {
            "wav": "audio/wav",
            "mp3": "audio/mpeg",
            "flac": "audio/flac",
            "ogg": "audio/ogg",
            "m4a": "audio/mp4",
        }
        url = f"{self.config.normalized_base_url}/audio/transcriptions"
        response = await self._http.post(
            url,
            data={
                "model": self.config.model,
                "response_format": "json",
                "temperature": "0",
            },
            files={
                "file": (
                    f"audio.{audio_format}",
                    audio_bytes,
                    mime_types.get(audio_format, "application/octet-stream"),
                )
            },
        )
        if response.status_code in {404, 405}:
            return await self._transcribe_chat_audio(
                audio_bytes,
                audio_format=audio_format,
            )
        if response.status_code >= 400:
            body = response.text
            raise Qwen3OmniHTTPError(
                f"POST {url} failed with HTTP {response.status_code}: "
                f"{body[:2000]}{_server_hint(body)}"
            )
        try:
            payload = response.json()
            transcript = payload.get("text", "")
        except (ValueError, AttributeError) as exc:
            raise Qwen3OmniProtocolError(
                f"Unexpected ASR transcription response: {response.text[:1000]}"
            ) from exc
        if not isinstance(transcript, str):
            raise Qwen3OmniProtocolError("ASR response field 'text' is not a string")
        return transcript.strip()

    async def _transcribe_chat_audio(
        self,
        audio_bytes: bytes,
        *,
        audio_format: str,
    ) -> str:
        payload = {
            "model": self.config.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_audio",
                            "input_audio": {
                                "data": base64.b64encode(audio_bytes).decode("ascii"),
                                "format": audio_format,
                            },
                        },
                        {
                            "type": "text",
                            "text": (
                                "Transcribe the audible speech exactly. Output only "
                                "the transcript, without notes or labels."
                            ),
                        },
                    ],
                }
            ],
            "stream": True,
            "max_tokens": self.config.max_tokens,
            "temperature": 0.0,
        }
        return await _stream_chat_completion(
            self._http,
            url=f"{self.config.normalized_base_url}/chat/completions",
            payload=payload,
            on_delta=None,
        )


class InterpretationPipeline:
    def __init__(
        self,
        omni: Qwen3OmniClient,
        *,
        asr: Qwen3ASRClient | None = None,
        direct_audio: bool = False,
    ):
        if not direct_audio and asr is None:
            raise ValueError("ASR client is required unless direct_audio is enabled")
        self.omni = omni
        self.asr = asr
        self.direct_audio = direct_audio

    async def interpret_pcm(
        self,
        pcm: bytes,
        *,
        sample_rate: int,
        previous_translation: str = "",
        on_transcript: DeltaCallback | None = None,
        on_delta: DeltaCallback | None = None,
    ) -> str:
        if self.direct_audio:
            return await self.omni.translate_pcm(
                pcm,
                sample_rate=sample_rate,
                previous_translation=previous_translation,
                on_delta=on_delta,
            )
        assert self.asr is not None
        transcript = await self.asr.transcribe_pcm(pcm, sample_rate=sample_rate)
        if transcript and on_transcript:
            on_transcript(transcript)
        return await self.omni.translate_text(
            transcript,
            previous_translation=previous_translation,
            on_delta=on_delta,
        )

    async def interpret_audio(
        self,
        audio_bytes: bytes,
        *,
        audio_format: str,
        previous_translation: str = "",
        on_transcript: DeltaCallback | None = None,
        on_delta: DeltaCallback | None = None,
    ) -> str:
        if self.direct_audio:
            return await self.omni.translate_audio(
                audio_bytes,
                audio_format=audio_format,
                previous_translation=previous_translation,
                on_delta=on_delta,
            )
        assert self.asr is not None
        transcript = await self.asr.transcribe_audio(
            audio_bytes,
            audio_format=audio_format,
        )
        if transcript and on_transcript:
            on_transcript(transcript)
        return await self.omni.translate_text(
            transcript,
            previous_translation=previous_translation,
            on_delta=on_delta,
        )


class MicrophoneInterpreter:
    def __init__(
        self,
        pipeline: InterpretationPipeline,
        microphone: MicrophoneConfig,
        *,
        input_device_index: int | None = None,
    ):
        self.pipeline = pipeline
        self.microphone = microphone
        self.input_device_index = input_device_index
        self._queue: asyncio.Queue[AudioSegment | None] = asyncio.Queue(
            maxsize=microphone.max_pending_segments
        )
        self._segmenter: SpeechSegmenter | None = None
        self._next_sequence = 1
        self._previous_translation = ""

    async def run(self) -> None:
        worker = asyncio.create_task(self._translation_worker())
        try:
            await self._capture_loop()
        finally:
            if self._segmenter:
                remaining = self._segmenter.flush()
                if remaining:
                    await self._queue.put(
                        AudioSegment(self._next_sequence, remaining)
                    )
            await self._queue.put(None)
            await worker

    async def _capture_loop(self) -> None:
        try:
            import pyaudio
        except ImportError as exc:
            raise Qwen3OmniError(
                "PyAudio is required for microphone mode. Install local/requirements.txt."
            ) from exc

        audio = pyaudio.PyAudio()
        stream = None
        try:
            stream = audio.open(
                format=pyaudio.paInt16,
                channels=self.microphone.channels,
                rate=self.microphone.sample_rate,
                input=True,
                input_device_index=self.input_device_index,
                frames_per_buffer=self.microphone.frames_per_buffer,
            )
            threshold = await self._calibrate(stream)
            self._segmenter = SpeechSegmenter(
                self.microphone, energy_threshold=threshold
            )
            print(
                f"麦克风已启动（阈值 {threshold}，片段最长 "
                f"{self.microphone.max_segment_ms / 1000:g}s）。按 Ctrl+C 退出。"
            )

            while True:
                frame = await asyncio.to_thread(
                    stream.read,
                    self.microphone.frames_per_buffer,
                    exception_on_overflow=False,
                )
                segment = self._segmenter.add_frame(frame)
                if segment:
                    self._enqueue(segment)
        finally:
            if stream is not None:
                stream.stop_stream()
                stream.close()
            audio.terminate()

    async def _calibrate(self, stream: object) -> int:
        frame_count = max(
            0,
            round(
                self.microphone.calibration_seconds * 1000
                / self.microphone.frame_ms
            ),
        )
        if not frame_count:
            return self.microphone.energy_threshold

        samples: list[int] = []
        print(f"正在校准环境噪声 {self.microphone.calibration_seconds:g} 秒，请保持安静...")
        for _ in range(frame_count):
            frame = await asyncio.to_thread(
                stream.read,
                self.microphone.frames_per_buffer,
                exception_on_overflow=False,
            )
            samples.append(pcm_rms(frame))
        ambient = statistics.median(samples) if samples else 0
        return max(self.microphone.energy_threshold, int(ambient * 3))

    def _enqueue(self, pcm: bytes) -> None:
        item = AudioSegment(self._next_sequence, pcm)
        self._next_sequence += 1
        try:
            self._queue.put_nowait(item)
        except asyncio.QueueFull:
            print(
                f"\n[警告] 翻译处理落后，已丢弃片段 {item.sequence} 以保持实时性。",
                flush=True,
            )

    async def _translation_worker(self) -> None:
        while True:
            item = await self._queue.get()
            try:
                if item is None:
                    return
                duration = (
                    len(item.pcm)
                    / self.microphone.sample_width
                    / self.microphone.sample_rate
                )
                print(f"\n[片段 {item.sequence} · {duration:.1f}s]", flush=True)
                started = time.monotonic()
                translation_started = False

                def show_transcript(text: str) -> None:
                    print(f"[原文] {text}", flush=True)

                def show_delta(text: str) -> None:
                    nonlocal translation_started
                    if not translation_started:
                        print("[译文] ", end="", flush=True)
                        translation_started = True
                    print(text, end="", flush=True)

                translation = await self.pipeline.interpret_pcm(
                    item.pcm,
                    previous_translation=self._previous_translation,
                    sample_rate=self.microphone.sample_rate,
                    on_transcript=show_transcript,
                    on_delta=show_delta,
                )
                if not translation:
                    print("[译文] （未识别到可翻译语音）", end="")
                print(f"\n[延迟] {time.monotonic() - started:.1f}s", flush=True)
                if translation:
                    self._previous_translation = translation
            except Exception as exc:
                print(f"\n[错误] 片段翻译失败: {exc}", flush=True)
            finally:
                self._queue.task_done()


def list_input_devices() -> list[tuple[int, str]]:
    try:
        import pyaudio
    except ImportError as exc:
        raise Qwen3OmniError("PyAudio is required to list input devices.") from exc

    audio = pyaudio.PyAudio()
    devices: list[tuple[int, str]] = []
    try:
        for index in range(audio.get_device_count()):
            info = audio.get_device_info_by_index(index)
            if int(info.get("maxInputChannels", 0)) > 0:
                devices.append((index, str(info.get("name", f"Device {index}"))))
    finally:
        audio.terminate()
    return devices


def read_audio_file(path: Path) -> tuple[bytes, str]:
    supported = {"wav", "mp3", "flac", "ogg", "m4a"}
    audio_format = path.suffix.lower().lstrip(".")
    if audio_format not in supported:
        raise ValueError(
            f"Unsupported audio extension .{audio_format}; expected one of "
            f"{', '.join(sorted(supported))}"
        )
    return path.read_bytes(), audio_format
