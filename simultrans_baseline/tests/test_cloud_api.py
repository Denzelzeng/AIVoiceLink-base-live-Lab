from __future__ import annotations

import base64
import json
import unittest

import httpx

from simultrans_baseline.config import ServiceConfig, TTSConfig
from simultrans_baseline.events import AudioWindow
from simultrans_baseline.providers.cloud_api import (
    DashScopeASR,
    DashScopeVoiceCloneTTS,
    ProviderError,
    QwenMTTranslator,
)


class CloudAPITests(unittest.IsolatedAsyncioTestCase):
    async def test_voice_enrollment_rejects_reference_below_24_khz(self) -> None:
        tts = DashScopeVoiceCloneTTS(
            TTSConfig(
                "dashscope_qwen_voice_clone",
                "https://api.test",
                "qwen3-tts-vc-2026-01-22",
            ),
            transport=httpx.MockTransport(lambda request: httpx.Response(500)),
        )
        with self.assertRaises(ProviderError):
            await tts.enroll(
                b"\x00\x00" * 48_000,
                sample_rate=16_000,
                transcript="三秒参考音频",
                language="Chinese",
            )
        await tts.aclose()

    async def test_separate_cloud_asr_mt_and_clone_tts_contracts(self) -> None:
        requests: list[tuple[str, str]] = []
        pcm = b"\x00\x00" * 20

        def handler(request: httpx.Request) -> httpx.Response:
            payload = json.loads(request.read())
            model = str(payload.get("model", ""))
            requests.append((request.url.path, model))
            if request.url.path.endswith("/chat/completions"):
                if model == "qwen3-asr-flash-2026-02-10":
                    audio = payload["messages"][0]["content"][0]["input_audio"]["data"]
                    self.assertTrue(audio.startswith("data:audio/wav;base64,"))
                    self.assertEqual(payload["asr_options"]["language"], "zh")
                    return httpx.Response(
                        200, json={"choices": [{"message": {"content": "你好"}}]}
                    )
                self.assertEqual(model, "qwen-mt-flash")
                self.assertEqual(payload["messages"], [{"role": "user", "content": "你好"}])
                self.assertEqual(payload["translation_options"]["target_lang"], "English")
                body = (
                    'data: {"choices":[{"delta":{"content":"Hel"}}]}\n\n'
                    'data: {"choices":[{"delta":{"content":"lo"}}]}\n\n'
                    "data: [DONE]\n\n"
                )
                return httpx.Response(
                    200,
                    headers={"content-type": "text/event-stream"},
                    content=body.encode(),
                )
            if request.url.path.endswith("/audio/tts/customization"):
                action = payload["input"]["action"]
                if action == "create":
                    self.assertEqual(payload["model"], "qwen-voice-enrollment")
                    self.assertEqual(
                        payload["input"]["target_model"], "qwen3-tts-vc-2026-01-22"
                    )
                    self.assertTrue(
                        payload["input"]["audio"]["data"].startswith(
                            "data:audio/wav;base64,"
                        )
                    )
                    return httpx.Response(200, json={"output": {"voice": "voice-1"}})
                if action == "delete":
                    self.assertEqual(payload["input"]["voice"], "voice-1")
                    return httpx.Response(200, json={"output": {"voice": "voice-1"}})
            if request.url.path.endswith("/multimodal-generation/generation"):
                self.assertEqual(request.headers.get("x-dashscope-sse"), "enable")
                self.assertEqual(payload["input"]["voice"], "voice-1")
                encoded = base64.b64encode(pcm).decode("ascii")
                body = (
                    "data: "
                    + json.dumps({"output": {"audio": {"data": encoded}}})
                    + "\n\n"
                    + 'data: {"output":{"finish_reason":"stop","audio":{"data":""}}}\n\n'
                )
                return httpx.Response(
                    200,
                    headers={"content-type": "text/event-stream"},
                    content=body.encode(),
                )
            return httpx.Response(404)

        transport = httpx.MockTransport(handler)
        asr = DashScopeASR(
            ServiceConfig(
                "dashscope_asr",
                "https://api.test/compatible-mode/v1",
                "qwen3-asr-flash-2026-02-10",
                send_language=True,
            ),
            transport=transport,
        )
        mt = QwenMTTranslator(
            ServiceConfig(
                "qwen_mt",
                "https://api.test/compatible-mode/v1",
                "qwen-mt-flash",
            ),
            transport=transport,
        )
        tts = DashScopeVoiceCloneTTS(
            TTSConfig(
                "dashscope_qwen_voice_clone",
                "https://api.test",
                "qwen3-tts-vc-2026-01-22",
            ),
            transport=transport,
        )
        window = AudioWindow(1, b"\x00\x00" * 72_000, 24_000, True, 0.0, 1.0)
        self.assertEqual(await asr.transcribe(window, language="Chinese"), "你好")
        self.assertEqual(
            await mt.translate(
                "你好",
                source_language="Chinese",
                target_language="English",
                committed_target="",
                context=[],
                domain="IT terminology",
            ),
            "Hello",
        )
        profile = await tts.enroll(
            window.pcm,
            sample_rate=24_000,
            transcript="你好",
            language="Chinese",
        )
        chunks = [
            chunk
            async for chunk in tts.synthesize(
                "Hello", language="English", profile_id=profile.profile_id
            )
        ]
        await tts.delete_profile(profile.profile_id)
        self.assertEqual(profile.profile_id, "voice-1")
        self.assertEqual(b"".join(chunk.data for chunk in chunks), pcm)
        self.assertIn(
            (
                "/compatible-mode/v1/chat/completions",
                "qwen3-asr-flash-2026-02-10",
            ),
            requests,
        )
        self.assertIn(
            (
                "/api/v1/services/audio/tts/customization",
                "qwen-voice-enrollment",
            ),
            requests,
        )
        await asr.aclose()
        await mt.aclose()
        await tts.aclose()


if __name__ == "__main__":
    unittest.main()
