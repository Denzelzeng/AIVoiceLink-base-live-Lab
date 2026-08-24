from __future__ import annotations

import asyncio
import re
import wave
from pathlib import Path

from .events import AudioChunk


class NullAudioSink:
    async def write(self, chunk: AudioChunk, *, segment_id: str) -> None:
        return None

    async def interrupt(self) -> None:
        return None

    async def aclose(self) -> None:
        return None


class WavDirectorySink:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir.resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._streams: dict[str, wave.Wave_write] = {}

    async def write(self, chunk: AudioChunk, *, segment_id: str) -> None:
        stream = self._streams.get(segment_id)
        if stream is None:
            safe_id = re.sub(r"[^A-Za-z0-9_.-]", "_", segment_id)
            stream = wave.open(str(self.output_dir / f"{safe_id}.wav"), "wb")
            stream.setnchannels(chunk.channels)
            stream.setsampwidth(chunk.sample_width)
            stream.setframerate(chunk.sample_rate)
            self._streams[segment_id] = stream
        stream.writeframes(chunk.data)

    async def interrupt(self) -> None:
        self._close_all()

    async def aclose(self) -> None:
        self._close_all()

    def _close_all(self) -> None:
        for stream in self._streams.values():
            stream.close()
        self._streams.clear()


class PyAudioSink:
    """Interruptible PCM16 playback using only methods exposed by PyAudio."""

    def __init__(self):
        try:
            import pyaudio
        except ImportError as exc:
            raise RuntimeError("PyAudio is required for speaker output") from exc
        self._pyaudio_module = pyaudio
        self._audio = pyaudio.PyAudio()
        self._stream = None
        self._format: tuple[int, int, int] | None = None
        self._lock = asyncio.Lock()
        self._generation = 0
        self._block_ms = 20

    async def write(self, chunk: AudioChunk, *, segment_id: str) -> None:
        del segment_id
        generation = self._generation
        frame_bytes = chunk.channels * chunk.sample_width
        block_bytes = max(
            frame_bytes,
            chunk.sample_rate * self._block_ms // 1_000 * frame_bytes,
        )
        block_bytes -= block_bytes % frame_bytes
        for offset in range(0, len(chunk.data), block_bytes):
            if generation != self._generation:
                return
            block = chunk.data[offset : offset + block_bytes]
            async with self._lock:
                if generation != self._generation:
                    return
                expected = (chunk.sample_rate, chunk.channels, chunk.sample_width)
                if self._stream is None or self._format != expected:
                    self._close_stream()
                    self._stream = self._audio.open(
                        format=self._audio.get_format_from_width(chunk.sample_width),
                        channels=chunk.channels,
                        rate=chunk.sample_rate,
                        output=True,
                        frames_per_buffer=max(1, len(block) // frame_bytes),
                    )
                    self._format = expected
                await asyncio.to_thread(self._stream.write, block)

    async def interrupt(self) -> None:
        # Invalidate an in-flight chunk before waiting for its current 20 ms block.
        self._generation += 1
        async with self._lock:
            self._close_stream()

    async def aclose(self) -> None:
        async with self._lock:
            self._close_stream()
            self._audio.terminate()

    def _close_stream(self) -> None:
        if self._stream is None:
            return
        stream = self._stream
        self._stream = None
        self._format = None
        try:
            is_active = getattr(stream, "is_active", None)
            if not callable(is_active) or is_active():
                stream.stop_stream()
        finally:
            stream.close()
