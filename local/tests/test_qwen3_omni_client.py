from __future__ import annotations

import json
import struct
import unittest
import wave

import httpx

from local.qwen3_omni_client import (
    ASRConfig,
    ClientConfig,
    InterpretationPipeline,
    MicrophoneConfig,
    Qwen3ASRClient,
    Qwen3OmniClient,
    SpeechSegmenter,
    pcm_to_wav,
)


def pcm_frame(amplitude: int, samples: int = 800) -> bytes:
    return struct.pack("<h", amplitude) * samples


class AudioHelpersTests(unittest.TestCase):
    def test_pcm_to_wav_has_expected_format(self) -> None:
        wav_bytes = pcm_to_wav(pcm_frame(1000), sample_rate=16_000)
        with wave.open(__import__("io").BytesIO(wav_bytes), "rb") as wav:
            self.assertEqual(wav.getnchannels(), 1)
            self.assertEqual(wav.getsampwidth(), 2)
            self.assertEqual(wav.getframerate(), 16_000)
            self.assertEqual(wav.getnframes(), 800)


class SpeechSegmenterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = MicrophoneConfig(
            frame_ms=50,
            pre_roll_ms=100,
            min_speech_ms=200,
            end_silence_ms=150,
            max_segment_ms=1000,
            energy_threshold=300,
        )

    def test_emits_speech_after_endpoint_silence(self) -> None:
        segmenter = SpeechSegmenter(self.config)
        for _ in range(2):
            self.assertIsNone(segmenter.add_frame(pcm_frame(0)))
        for _ in range(4):
            self.assertIsNone(segmenter.add_frame(pcm_frame(1200)))

        result = None
        for _ in range(3):
            result = segmenter.add_frame(pcm_frame(0))
        self.assertIsNotNone(result)
        self.assertGreater(len(result or b""), 4 * len(pcm_frame(1200)))

    def test_discards_short_noise(self) -> None:
        segmenter = SpeechSegmenter(self.config)
        segmenter.add_frame(pcm_frame(1200))
        result = None
        for _ in range(3):
            result = segmenter.add_frame(pcm_frame(0))
        self.assertIsNone(result)
        self.assertIsNone(segmenter.flush())

    def test_forces_a_segment_at_maximum_duration(self) -> None:
        segmenter = SpeechSegmenter(self.config)
        result = None
        for _ in range(20):
            result = segmenter.add_frame(pcm_frame(1200))
            if result:
                break
        self.assertIsNotNone(result)


class ClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_check_server_and_streaming_audio_request(self) -> None:
        captured_payload: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/health":
                return httpx.Response(200, json={})
            if request.url.path == "/v1/models":
                return httpx.Response(
                    200,
                    json={"data": [{"id": "qwen3-omni"}]},
                )
            if request.url.path == "/v1/chat/completions":
                captured_payload.update(json.loads(request.read()))
                body = (
                    'data: {"choices":[{"delta":{"content":"Hello"}}]}\n\n'
                    'data: {"choices":[{"delta":{"content":" world"}}]}\n\n'
                    "data: [DONE]\n\n"
                )
                return httpx.Response(
                    200,
                    headers={"content-type": "text/event-stream"},
                    content=body.encode(),
                )
            return httpx.Response(404)

        transport = httpx.MockTransport(handler)
        config = ClientConfig(
            base_url="http://model.local:8003/v1",
            target_language="English",
        )
        async with Qwen3OmniClient(config, transport=transport) as client:
            status = await client.check_server()
            self.assertTrue(status["configured_model_available"])
            deltas: list[str] = []
            result = await client.translate_pcm(
                pcm_frame(1200),
                on_delta=deltas.append,
            )

        self.assertEqual(result, "Hello world")
        self.assertEqual(deltas, ["Hello", " world"])
        self.assertTrue(captured_payload["stream"])
        messages = captured_payload["messages"]
        audio_part = messages[1]["content"][0]
        self.assertEqual(audio_part["type"], "input_audio")
        self.assertEqual(audio_part["input_audio"]["format"], "wav")
        self.assertTrue(audio_part["input_audio"]["data"])

    async def test_asr_then_omni_pipeline(self) -> None:
        request_paths: list[str] = []
        omni_payloads: list[dict[str, object]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            request_paths.append(request.url.path)
            if request.url.path == "/v1/audio/transcriptions":
                body = request.read()
                self.assertIn(b'qwen3-asr', body)
                self.assertIn(b'audio.wav', body)
                return httpx.Response(200, json={"text": "你好世界"})
            payload = json.loads(request.read())
            omni_payloads.append(payload)
            text = "Hello world"
            body = f'data: {{"choices":[{{"delta":{{"content":"{text}"}}}}]}}\n\n'
            body += "data: [DONE]\n\n"
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=body.encode(),
            )

        transport = httpx.MockTransport(handler)
        omni_config = ClientConfig(base_url="http://model.local:8003/v1")
        asr_config = ASRConfig(base_url="http://model.local:8004/v1")
        async with (
            Qwen3OmniClient(omni_config, transport=transport) as omni,
            Qwen3ASRClient(asr_config, transport=transport) as asr,
        ):
            pipeline = InterpretationPipeline(omni, asr=asr)
            transcripts: list[str] = []
            translation = await pipeline.interpret_pcm(
                pcm_frame(1200),
                sample_rate=16_000,
                on_transcript=transcripts.append,
            )

        self.assertEqual(transcripts, ["你好世界"])
        self.assertEqual(translation, "Hello world")
        self.assertEqual(
            request_paths,
            ["/v1/audio/transcriptions", "/v1/chat/completions"],
        )
        self.assertIn("你好世界", omni_payloads[0]["messages"][1]["content"])


if __name__ == "__main__":
    unittest.main()
