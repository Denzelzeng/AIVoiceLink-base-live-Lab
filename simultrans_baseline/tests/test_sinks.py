from __future__ import annotations

import asyncio
import sys
import time
import types
import unittest
from unittest.mock import patch

from simultrans_baseline.events import AudioChunk
from simultrans_baseline.sinks import PyAudioSink


class _FakeStream:
    def __init__(self, *, write_delay: float = 0.0):
        self.write_delay = write_delay
        self.writes = 0
        self.stopped = False
        self.closed = False

    def write(self, data: bytes) -> None:
        del data
        self.writes += 1
        if self.write_delay:
            time.sleep(self.write_delay)

    def is_active(self) -> bool:
        return not self.stopped and not self.closed

    def stop_stream(self) -> None:
        self.stopped = True

    def close(self) -> None:
        self.closed = True


class _FakePyAudio:
    def __init__(self, stream: _FakeStream):
        self.stream = stream
        self.terminated = False

    def get_format_from_width(self, width: int) -> int:
        return width

    def open(self, **kwargs):
        del kwargs
        return self.stream

    def terminate(self) -> None:
        self.terminated = True


class SinkTests(unittest.IsolatedAsyncioTestCase):
    async def test_interrupt_uses_supported_stop_stream_and_cancels_chunk(self) -> None:
        stream = _FakeStream(write_delay=0.03)
        fake_audio = _FakePyAudio(stream)
        module = types.SimpleNamespace(PyAudio=lambda: fake_audio)
        with patch.dict(sys.modules, {"pyaudio": module}):
            sink = PyAudioSink()
        chunk = AudioChunk(b"\x00\x00" * 2_400, sample_rate=24_000)
        writer = asyncio.create_task(sink.write(chunk, segment_id="test"))
        await asyncio.sleep(0.01)
        await sink.interrupt()
        await writer
        self.assertTrue(stream.stopped)
        self.assertTrue(stream.closed)
        self.assertLess(stream.writes, 5)
        await sink.aclose()
        self.assertTrue(fake_audio.terminated)


if __name__ == "__main__":
    unittest.main()
