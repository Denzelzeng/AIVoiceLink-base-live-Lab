from __future__ import annotations

import base64
import asyncio
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
    async def test_new_voice_prepares_while_old_voice_is_still_synthesizing(self) -> None:
        pcm = b"\x03\x00" * 10
        old_audio_started = asyncio.Event()
        release_old_response = asyncio.Event()
        sockets = []

        class FakeSocket:
            def __init__(self, index):
                self.index = index
                self.step = 0

            async def send(self, value):
                del value

            async def recv(self):
                self.step += 1
                if self.step == 1:
                    return json.dumps({"type": "session.created"})
                if self.step == 2:
                    return json.dumps({"type": "session.updated"})
                if self.index == 0 and self.step == 3:
                    old_audio_started.set()
                    return json.dumps(
                        {
                            "type": "response.audio.delta",
                            "delta": base64.b64encode(pcm).decode("ascii"),
                        }
                    )
                if self.index == 0 and self.step == 4:
                    await release_old_response.wait()
                    return json.dumps({"type": "response.done"})
                if self.index == 1 and self.step == 3:
                    return json.dumps(
                        {
                            "type": "response.audio.delta",
                            "delta": base64.b64encode(pcm).decode("ascii"),
                        }
                    )
                if self.index == 1 and self.step == 4:
                    return json.dumps({"type": "response.done"})
                raise AssertionError((self.index, self.step))

        class FakeConnection:
            def __init__(self, index):
                self.socket = FakeSocket(index)

            async def __aenter__(self):
                sockets.append(self.socket)
                return self.socket

            async def __aexit__(self, exc_type, exc, tb):
                return False

        def connect(url, **kwargs):
            del url, kwargs
            return FakeConnection(len(sockets))

        tts = DashScopeVoiceCloneTTS(
            TTSConfig(
                "dashscope_qwen_voice_clone",
                "https://api.test",
                "qwen3-tts-vc-realtime-2026-01-15",
                websocket_url="wss://api.test/realtime",
                speech_rate=1.3,
            ),
            transport=httpx.MockTransport(lambda request: httpx.Response(500)),
            ws_connect=connect,
        )
        await tts.warmup()
        await tts.prepare_voice("voice-old", language="English")

        async def consume_old():
            return [
                chunk
                async for chunk in tts.synthesize(
                    "Old sentence", language="English", profile_id="voice-old"
                )
            ]

        old_task = asyncio.create_task(consume_old())
        await asyncio.wait_for(old_audio_started.wait(), timeout=1)
        await asyncio.wait_for(
            tts.prepare_voice("voice-new", language="English"), timeout=0.1
        )
        self.assertFalse(old_task.done())
        self.assertEqual(len(sockets), 2)

        release_old_response.set()
        old_chunks = await old_task
        new_chunks = [
            chunk
            async for chunk in tts.synthesize(
                "New sentence", language="English", profile_id="voice-new"
            )
        ]
        await tts.aclose()
        self.assertEqual(b"".join(chunk.data for chunk in old_chunks), pcm)
        self.assertEqual(b"".join(chunk.data for chunk in new_chunks), pcm)

    async def test_realtime_tts_rebuilds_failed_socket_before_audio(self) -> None:
        pcm = b"\x02\x00" * 10
        connect_calls = 0

        class FakeSocket:
            def __init__(self, fail):
                tail = (
                    [
                        {
                            "type": "error",
                            "error": {"message": "session already failed"},
                        }
                    ]
                    if fail
                    else [
                        {
                            "type": "response.audio.delta",
                            "delta": base64.b64encode(pcm).decode("ascii"),
                        },
                        {"type": "response.done"},
                    ]
                )
                self.events = [
                    {"type": "session.created"},
                    {"type": "session.updated"},
                    *tail,
                ]

            async def send(self, value):
                return None

            async def recv(self):
                return json.dumps(self.events.pop(0))

        class FakeConnection:
            def __init__(self, fail):
                self.fail = fail

            async def __aenter__(self):
                return FakeSocket(self.fail)

            async def __aexit__(self, exc_type, exc, tb):
                return False

        def connect(url, **kwargs):
            nonlocal connect_calls
            connect_calls += 1
            return FakeConnection(fail=connect_calls == 1)

        tts = DashScopeVoiceCloneTTS(
            TTSConfig(
                "dashscope_qwen_voice_clone",
                "https://api.test",
                "qwen3-tts-vc-realtime-2026-01-15",
                websocket_url="wss://api.test/realtime",
                speech_rate=1.3,
            ),
            transport=httpx.MockTransport(lambda request: httpx.Response(500)),
            ws_connect=connect,
        )
        chunks = [
            chunk
            async for chunk in tts.synthesize(
                "Recover", language="English", profile_id="voice-retry"
            )
        ]
        await tts.aclose()
        self.assertEqual(connect_calls, 2)
        self.assertEqual(b"".join(chunk.data for chunk in chunks), pcm)

    async def test_realtime_clone_tts_sends_1_3x_speech_rate(self) -> None:
        pcm = b"\x01\x00" * 20
        sent: list[dict] = []
        connection_args: dict[str, object] = {}
        connect_calls = 0

        class FakeSocket:
            def __init__(self):
                self.events = [
                    {"type": "session.created"},
                    {"type": "session.updated"},
                    {
                        "type": "response.audio.delta",
                        "delta": base64.b64encode(pcm).decode("ascii"),
                    },
                    {"type": "response.done"},
                    {
                        "type": "response.audio.delta",
                        "delta": base64.b64encode(pcm).decode("ascii"),
                    },
                    {"type": "response.done"},
                ]

            async def send(self, value):
                sent.append(json.loads(value))

            async def recv(self):
                return json.dumps(self.events.pop(0))

        class FakeConnection:
            async def __aenter__(self):
                self.socket = FakeSocket()
                return self.socket

            async def __aexit__(self, exc_type, exc, tb):
                return False

        def connect(url, **kwargs):
            nonlocal connect_calls
            connect_calls += 1
            connection_args["url"] = url
            connection_args.update(kwargs)
            return FakeConnection()

        tts = DashScopeVoiceCloneTTS(
            TTSConfig(
                "dashscope_qwen_voice_clone",
                "https://api.test",
                "qwen3-tts-vc-realtime-2026-01-15",
                websocket_url="wss://api.test/realtime",
                speech_rate=1.3,
            ),
            transport=httpx.MockTransport(lambda request: httpx.Response(500)),
            ws_connect=connect,
        )
        await tts.warmup()
        await tts.prepare_voice("voice-fast", language="English")
        chunks = [
            chunk
            async for chunk in tts.synthesize(
                "Hello", language="English", profile_id="voice-fast"
            )
        ]
        second_chunks = [
            chunk
            async for chunk in tts.synthesize(
                "Again", language="English", profile_id="voice-fast"
            )
        ]
        await tts.prepare_voice("voice-new", language="English")
        third_chunks = [
            chunk
            async for chunk in tts.synthesize(
                "New speaker", language="English", profile_id="voice-new"
            )
        ]
        await tts.aclose()

        updates = [item for item in sent if item["type"] == "session.update"]
        update = updates[0]
        self.assertEqual(update["session"]["speech_rate"], 1.3)
        self.assertEqual(update["session"]["mode"], "commit")
        self.assertEqual(update["session"]["voice"], "voice-fast")
        self.assertEqual(b"".join(chunk.data for chunk in chunks), pcm)
        self.assertEqual(b"".join(chunk.data for chunk in second_chunks), pcm)
        self.assertEqual(b"".join(chunk.data for chunk in third_chunks), pcm)
        self.assertEqual(len(updates), 2)
        self.assertEqual(connect_calls, 2)
        self.assertEqual(updates[1]["session"]["voice"], "voice-new")
        self.assertIn(
            "model=qwen3-tts-vc-realtime-2026-01-15",
            str(connection_args["url"]),
        )

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
