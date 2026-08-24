from __future__ import annotations

import base64
import json
import struct
import tempfile
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import httpx

from qwen35_omni_client import (
    AudioPlayer,
    ClientConfig,
    MicrophoneConfig,
    Qwen35OmniClient,
    Qwen35OmniRealtimeClient,
    SpeechSegmenter,
    pcm_to_wav,
    resolve_credentials,
)


def pcm_frame(amplitude: int, samples: int = 800) -> bytes:
    return struct.pack("<h", amplitude) * samples


class AudioHelpersTests(unittest.TestCase):
    def test_pcm_to_wav(self) -> None:
        wav_bytes = pcm_to_wav(pcm_frame(1000))
        with wave.open(__import__("io").BytesIO(wav_bytes), "rb") as audio:
            self.assertEqual(audio.getnchannels(), 1)
            self.assertEqual(audio.getsampwidth(), 2)
            self.assertEqual(audio.getframerate(), 16_000)
            self.assertEqual(audio.getnframes(), 800)

    def test_audio_player_reuses_shared_portaudio_instance(self) -> None:
        output_stream = MagicMock()
        shared_audio = MagicMock()
        shared_audio.open.return_value = output_stream
        pyaudio_module = SimpleNamespace(paInt16=8, PyAudio=MagicMock())
        player = AudioPlayer()

        with patch.dict("sys.modules", {"pyaudio": pyaudio_module}):
            player.start(shared_audio)
            player.put(b"\x00\x00")
            player.close()

        pyaudio_module.PyAudio.assert_not_called()
        shared_audio.open.assert_called_once()
        shared_audio.terminate.assert_not_called()
        output_stream.write.assert_called_once_with(b"\x00\x00")


class SegmenterTests(unittest.TestCase):
    def test_emits_after_silence(self) -> None:
        config = MicrophoneConfig(
            frame_ms=50,
            min_speech_ms=200,
            end_silence_ms=150,
            max_segment_ms=1000,
            calibration_seconds=0,
        )
        segmenter = SpeechSegmenter(config)
        for _ in range(4):
            self.assertIsNone(segmenter.add_frame(pcm_frame(1200)))
        result = None
        for _ in range(3):
            result = segmenter.add_frame(pcm_frame(0))
        self.assertIsNotNone(result)

    def test_discards_short_noise(self) -> None:
        config = MicrophoneConfig(
            frame_ms=50,
            min_speech_ms=200,
            end_silence_ms=100,
            max_segment_ms=1000,
        )
        segmenter = SpeechSegmenter(config)
        segmenter.add_frame(pcm_frame(1200))
        self.assertIsNone(segmenter.add_frame(pcm_frame(0)))
        self.assertIsNone(segmenter.add_frame(pcm_frame(0)))


class CredentialTests(unittest.TestCase):
    def test_reads_transposed_workspace_csv(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "Default Workspace-apiKey-1.csv").write_text(
                "id,1\napiKey,test-key\nopenAiCompatible,https://example.test/v1\n",
                encoding="utf-8",
            )
            old_key = __import__("os").environ.pop("DASHSCOPE_API_KEY", None)
            old_url = __import__("os").environ.pop("QWEN35_OMNI_BASE_URL", None)
            try:
                result = resolve_credentials(root)
            finally:
                if old_key is not None:
                    __import__("os").environ["DASHSCOPE_API_KEY"] = old_key
                if old_url is not None:
                    __import__("os").environ["QWEN35_OMNI_BASE_URL"] = old_url
            self.assertEqual(result.api_key, "test-key")
            self.assertEqual(result.base_url, "https://example.test/v1")


class ClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_audio_request_streams_text_and_audio(self) -> None:
        captured_payload: dict[str, object] = {}
        audio = base64.b64encode(b"\x01\x02\x03\x04").decode()

        def handler(request: httpx.Request) -> httpx.Response:
            captured_payload.update(json.loads(request.read()))
            body = (
                'data: {"choices":[{"delta":{"content":"Hello "}}]}\n\n'
                f'data: {{"choices":[{{"delta":{{"content":"world","audio":{{"data":"{audio}"}}}}}}]}}\n\n'
                "data: [DONE]\n\n"
            )
            return httpx.Response(200, content=body.encode(), headers={"content-type": "text/event-stream"})

        config = ClientConfig(
            api_key="test-key",
            base_url="https://example.test/v1",
            audio_output=True,
        )
        async with Qwen35OmniClient(config, transport=httpx.MockTransport(handler)) as client:
            text_deltas: list[str] = []
            audio_deltas: list[bytes] = []
            result = await client.translate_pcm(
                pcm_frame(1200),
                on_text=text_deltas.append,
                on_audio=audio_deltas.append,
            )

        self.assertEqual(result.text, "Hello world")
        self.assertEqual(result.audio, b"\x01\x02\x03\x04")
        self.assertEqual(text_deltas, ["Hello ", "world"])
        self.assertEqual(b"".join(audio_deltas), result.audio)
        self.assertEqual(captured_payload["modalities"], ["text", "audio"])
        self.assertEqual(captured_payload["audio"]["voice"], "Tina")
        content = captured_payload["messages"][1]["content"]
        self.assertTrue(content[0]["input_audio"]["data"].startswith("data:;base64,"))
        self.assertTrue(captured_payload["stream"])

    async def test_text_only_omits_audio_parameter(self) -> None:
        captured_payload: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured_payload.update(json.loads(request.read()))
            body = 'data: {"choices":[{"delta":{"content":"hello"}}]}\n\ndata: [DONE]\n\n'
            return httpx.Response(200, content=body.encode())

        config = ClientConfig(
            api_key="test-key",
            base_url="https://example.test/v1",
            audio_output=False,
        )
        async with Qwen35OmniClient(config, transport=httpx.MockTransport(handler)) as client:
            await client.translate_text("hello")
        self.assertEqual(captured_payload["modalities"], ["text"])
        self.assertNotIn("audio", captured_payload)


class FakeRealtimeWebSocket:
    def __init__(self) -> None:
        audio = base64.b64encode(b"\x01\x02\x03\x04").decode()
        self.events = [
            {
                "type": "session.updated",
                "session": {
                    "audio": {
                        "input": {"format": {"type": "pcm", "sample_rate": 16_000}}
                    }
                },
            },
            {"type": "input_audio_buffer.committed"},
            {
                "type": "conversation.item.input_audio_transcription.delta",
                "text": "你好",
                "stash": "",
                "language": "zh",
            },
            {
                "type": "conversation.item.input_audio_transcription.completed",
                "transcript": "你好",
            },
            {"type": "response.audio_transcript.delta", "delta": "Hello"},
            {"type": "response.audio.delta", "delta": audio},
            {"type": "response.done", "response": {"usage": {"total_tokens": 3}}},
        ]
        self.sent: list[dict[str, object]] = []
        self.closed = False

    async def send(self, message: str) -> None:
        self.sent.append(json.loads(message))

    async def recv(self) -> str:
        return json.dumps(self.events.pop(0))

    async def close(self) -> None:
        self.closed = True


class RealtimeClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_realtime_returns_omni_audio_delta(self) -> None:
        websocket = FakeRealtimeWebSocket()
        config = ClientConfig(
            api_key="test-key",
            base_url="https://workspace.example/v1",
            source_language="Chinese",
            target_language="English",
            audio_output=True,
        )
        client = Qwen35OmniRealtimeClient(config, websocket=websocket)
        self.assertIn(
            "model=qwen3.5-omni-flash-realtime", client.websocket_url
        )
        text_deltas: list[str] = []
        audio_deltas: list[bytes] = []
        await client.connect()
        result = await client.translate_pcm(
            pcm_frame(1000),
            sample_rate=16_000,
            on_text=text_deltas.append,
            on_audio=audio_deltas.append,
        )
        await client.aclose()

        self.assertEqual(result.text, "Hello")
        self.assertEqual(result.audio, b"\x01\x02\x03\x04")
        self.assertEqual(result.input_transcript, "你好")
        self.assertEqual(result.input_language, "zh")
        self.assertEqual(text_deltas, ["Hello"])
        self.assertEqual(audio_deltas, [b"\x01\x02\x03\x04"])
        self.assertEqual(websocket.sent[0]["type"], "session.update")
        session = websocket.sent[0]["session"]
        self.assertEqual(session["modalities"], ["text", "audio"])
        self.assertEqual(session["voice"], "Tina")
        self.assertEqual(session["audio"]["input"]["format"]["sample_rate"], 16_000)
        self.assertEqual(
            session["input_audio_transcription"]["model"],
            "qwen3-asr-flash-realtime",
        )
        self.assertIn(
            "input_audio_buffer.append",
            [event["type"] for event in websocket.sent],
        )
        self.assertTrue(websocket.closed)

    async def test_resamples_native_44100_audio_to_16000(self) -> None:
        source = pcm_frame(1000, samples=44_100)
        converted, rate = Qwen35OmniRealtimeClient._resample_pcm(source, 44_100)
        self.assertEqual(rate, 16_000)
        self.assertAlmostEqual(len(converted) / 2 / rate, 1.0, places=3)


if __name__ == "__main__":
    unittest.main()
