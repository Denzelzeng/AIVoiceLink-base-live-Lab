from __future__ import annotations

import asyncio
import io
import math
import statistics
import time
import wave
from array import array
from collections import deque
from pathlib import Path
from collections.abc import Awaitable, Callable
from typing import AsyncIterator

from .config import AudioConfig
from .events import AudioWindow
from .vad import SpeechDetector


def pcm_rms(pcm: bytes) -> int:
    """Return RMS amplitude for little-endian signed 16-bit PCM."""
    if not pcm:
        return 0
    if len(pcm) % 2:
        raise ValueError("16-bit PCM must contain an even number of bytes")
    samples = array("h")
    samples.frombytes(pcm)
    if not samples:
        return 0
    return int(math.sqrt(sum(value * value for value in samples) / len(samples)))


def pcm_to_wav(
    pcm: bytes,
    *,
    sample_rate: int = 16_000,
    channels: int = 1,
    sample_width: int = 2,
) -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as stream:
        stream.setnchannels(channels)
        stream.setsampwidth(sample_width)
        stream.setframerate(sample_rate)
        stream.writeframes(pcm)
    return output.getvalue()


class EnergyTurnSegmenter:
    """Energy VAD that emits cumulative partial windows and a final window."""

    def __init__(
        self,
        config: AudioConfig,
        *,
        energy_threshold: int | None = None,
        speech_detector: SpeechDetector | None = None,
    ):
        self.config = config
        self.energy_threshold = energy_threshold or config.energy_threshold
        self.speech_detector = speech_detector
        self._pre_roll: deque[bytes] = deque(
            maxlen=max(1, config.pre_roll_ms // config.frame_ms)
        )
        self._active = False
        self._frames: list[bytes] = []
        self._speech_ms = 0
        self._silence_ms = 0
        self._duration_ms = 0
        self._next_partial_ms = config.partial_interval_ms
        self._turn_id = 1
        self._started_at = 0.0
        self._speech_started = False

    def add_frame(self, frame: bytes, *, captured_at: float | None = None) -> list[AudioWindow]:
        captured = time.monotonic() if captured_at is None else captured_at
        voiced = (
            self.speech_detector.is_speech(
                frame,
                sample_rate=self.config.sample_rate,
            )
            if self.speech_detector is not None
            else pcm_rms(frame) >= self.energy_threshold
        )
        if not self._active:
            self._pre_roll.append(frame)
            if not voiced:
                return []
            self._active = True
            self._frames = list(self._pre_roll)
            self._pre_roll.clear()
            self._speech_ms = self.config.frame_ms
            self._silence_ms = 0
            self._duration_ms = len(self._frames) * self.config.frame_ms
            self._next_partial_ms = self.config.partial_interval_ms
            self._started_at = captured - self._duration_ms / 1_000
            self._speech_started = True
            return []

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
        reached_limit = self._duration_ms >= self.config.max_turn_ms
        short_noise = (
            self._silence_ms >= self.config.end_silence_ms
            and self._speech_ms < self.config.min_speech_ms
        )
        if reached_endpoint or reached_limit or short_noise:
            final = self._finish(captured_at=captured, include_short=reached_limit)
            return [final] if final else []

        if self._duration_ms >= self._next_partial_ms:
            self._next_partial_ms += self.config.partial_interval_ms
            return [self._snapshot(captured_at=captured, is_final=False)]
        return []

    def consume_speech_started(self) -> bool:
        value = self._speech_started
        self._speech_started = False
        return value

    def flush(self, *, captured_at: float | None = None) -> AudioWindow | None:
        if not self._active:
            return None
        return self._finish(
            captured_at=time.monotonic() if captured_at is None else captured_at,
            include_short=False,
        )

    def _snapshot(self, *, captured_at: float, is_final: bool) -> AudioWindow:
        return AudioWindow(
            turn_id=self._turn_id,
            pcm=b"".join(self._frames),
            sample_rate=self.config.sample_rate,
            is_final=is_final,
            started_at=self._started_at,
            captured_at=captured_at,
        )

    def _finish(self, *, captured_at: float, include_short: bool) -> AudioWindow | None:
        speech_ms = self._speech_ms
        frames = self._frames
        if self._silence_ms:
            keep_ms = min(100, self._silence_ms)
            remove = (self._silence_ms - keep_ms) // self.config.frame_ms
            if remove:
                frames = frames[:-remove]

        result: AudioWindow | None = None
        if speech_ms >= self.config.min_speech_ms or include_short:
            result = AudioWindow(
                turn_id=self._turn_id,
                pcm=b"".join(frames),
                sample_rate=self.config.sample_rate,
                is_final=True,
                started_at=self._started_at,
                captured_at=captured_at,
            )
            self._turn_id += 1

        self._active = False
        self._frames = []
        self._speech_ms = 0
        self._silence_ms = 0
        self._duration_ms = 0
        self._pre_roll.clear()
        self._started_at = 0.0
        return result


class WavWindowSource:
    def __init__(
        self,
        path: Path,
        config: AudioConfig,
        *,
        real_time: bool = True,
        speech_detector: SpeechDetector | None = None,
        on_speech_started: Callable[[], Awaitable[None] | None] | None = None,
    ):
        self.path = path.resolve()
        self.config = config
        self.real_time = real_time
        self.speech_detector = speech_detector
        self.on_speech_started = on_speech_started

    async def __aiter__(self) -> AsyncIterator[AudioWindow]:
        with wave.open(str(self.path), "rb") as stream:
            actual = (
                stream.getframerate(),
                stream.getnchannels(),
                stream.getsampwidth(),
            )
            expected = (
                self.config.sample_rate,
                self.config.channels,
                self.config.sample_width,
            )
            if actual != expected:
                raise ValueError(
                    f"WAV must be {self.config.sample_rate} Hz, mono, "
                    "signed 16-bit PCM; "
                    f"got rate={actual[0]}, channels={actual[1]}, width={actual[2]}"
                )
            segmenter = EnergyTurnSegmenter(
                self.config,
                speech_detector=self.speech_detector,
            )
            while True:
                frame = stream.readframes(self.config.frames_per_buffer)
                if not frame:
                    break
                expected_bytes = self.config.frames_per_buffer * self.config.sample_width
                if len(frame) < expected_bytes:
                    frame += b"\x00" * (expected_bytes - len(frame))
                now = time.monotonic()
                windows = segmenter.add_frame(frame, captured_at=now)
                if segmenter.consume_speech_started() and self.on_speech_started:
                    result = self.on_speech_started()
                    if asyncio.iscoroutine(result):
                        await result
                for window in windows:
                    yield window
                if self.real_time:
                    await asyncio.sleep(self.config.frame_ms / 1_000)
            final = segmenter.flush()
            if final:
                yield final


class MicrophoneWindowSource:
    def __init__(
        self,
        config: AudioConfig,
        *,
        input_device_index: int | None = None,
        speech_detector: SpeechDetector | None = None,
        on_speech_started: Callable[[], Awaitable[None] | None] | None = None,
        on_ready: Callable[[int], Awaitable[None] | None] | None = None,
    ):
        self.config = config
        self.input_device_index = input_device_index
        self.speech_detector = speech_detector
        self.on_speech_started = on_speech_started
        self.on_ready = on_ready

    async def __aiter__(self) -> AsyncIterator[AudioWindow]:
        try:
            import pyaudio
        except ImportError as exc:
            raise RuntimeError("PyAudio is required for microphone input") from exc

        audio = pyaudio.PyAudio()
        stream = None
        segmenter: EnergyTurnSegmenter | None = None
        try:
            stream = audio.open(
                format=pyaudio.paInt16,
                channels=self.config.channels,
                rate=self.config.sample_rate,
                input=True,
                input_device_index=self.input_device_index,
                frames_per_buffer=self.config.frames_per_buffer,
            )
            threshold = await self._calibrate(stream)
            segmenter = EnergyTurnSegmenter(
                self.config,
                energy_threshold=threshold,
                speech_detector=self.speech_detector,
            )
            if self.on_ready:
                result = self.on_ready(threshold)
                if asyncio.iscoroutine(result):
                    await result
            while True:
                frame = await asyncio.to_thread(
                    stream.read,
                    self.config.frames_per_buffer,
                    exception_on_overflow=False,
                )
                windows = segmenter.add_frame(frame)
                if segmenter.consume_speech_started() and self.on_speech_started:
                    result = self.on_speech_started()
                    if asyncio.iscoroutine(result):
                        await result
                for window in windows:
                    yield window
        finally:
            if stream is not None:
                stream.stop_stream()
                stream.close()
            audio.terminate()

    async def _calibrate(self, stream: object) -> int:
        count = max(
            0,
            round(
                self.config.calibration_seconds * 1_000 / self.config.frame_ms
            ),
        )
        if not count:
            return self.config.energy_threshold
        samples: list[int] = []
        for _ in range(count):
            frame = await asyncio.to_thread(
                stream.read,
                self.config.frames_per_buffer,
                exception_on_overflow=False,
            )
            samples.append(pcm_rms(frame))
        ambient = statistics.median(samples) if samples else 0
        return max(self.config.energy_threshold, int(ambient * 3))


def list_input_devices() -> list[tuple[int, str]]:
    try:
        import pyaudio
    except ImportError as exc:
        raise RuntimeError("PyAudio is required to list audio devices") from exc
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
