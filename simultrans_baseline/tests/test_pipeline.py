from __future__ import annotations

import asyncio
import time
import unittest
from dataclasses import replace
from pathlib import Path

from simultrans_baseline.config import apply_overrides, load_config
from simultrans_baseline.endpoint import HeuristicSemanticEndpoint
from simultrans_baseline.events import AudioChunk, AudioWindow, PipelineEvent
from simultrans_baseline.pipeline import RealtimeInterpretationPipeline
from simultrans_baseline.providers.mock import (
    MockCloningTTS,
    RuleBasedMockTranslator,
    ScriptedASR,
)
from simultrans_baseline.sinks import NullAudioSink


ROOT = Path(__file__).resolve().parents[1]


class Source:
    def __init__(self, windows):
        self.windows = windows

    async def __aiter__(self):
        for window in self.windows:
            yield window
            await asyncio.sleep(0)


def window(segment: int, duration_ms: int, final: bool, started: float) -> AudioWindow:
    return AudioWindow(
        turn_id=segment,
        pcm=b"\x01\x00" * round(16_000 * duration_ms / 1_000),
        sample_rate=16_000,
        is_final=final,
        started_at=started,
        captured_at=time.monotonic(),
    )


class PipelineTests(unittest.IsolatedAsyncioTestCase):
    async def test_independent_audio_output_is_not_cancelled_by_new_speech(self) -> None:
        config = apply_overrides(
            load_config(ROOT / "configs" / "mock.toml"),
            audio_output=True,
            voice_consent=True,
        )
        config = replace(
            config,
            streaming=replace(config.streaming, barge_in_enabled=False),
        )
        events: list[PipelineEvent] = []
        tts_started = asyncio.Event()

        async def collect(event: PipelineEvent) -> None:
            events.append(event)
            if event.kind == "tts.started" and event.turn_id == 1:
                tts_started.set()

        class SlowTTS(MockCloningTTS):
            async def synthesize(self, text, *, language, profile_id):
                self.synthesized.append((text, profile_id))
                yield AudioChunk(b"\x00\x00" * 100, self.sample_rate)
                await asyncio.sleep(0.03)
                yield AudioChunk(b"\x00\x00" * 100, self.sample_rate)

        tts = SlowTTS()
        pipeline = RealtimeInterpretationPipeline(
            config,
            asr=ScriptedASR(["第一句。", "第二句。"]),
            translator=RuleBasedMockTranslator(),
            endpoint=HeuristicSemanticEndpoint(),
            tts=tts,
            audio_sink=NullAudioSink(),
            event_handler=collect,
        )
        started = time.monotonic()

        class ContinuousSource:
            async def __aiter__(self):
                await pipeline.on_speech_started()
                yield window(1, 3000, True, started)
                await asyncio.wait_for(tts_started.wait(), timeout=1)
                await pipeline.on_speech_started()
                yield window(2, 3000, True, started)

        await pipeline.run(ContinuousSource())
        self.assertFalse(any(event.kind == "tts.cancelled" for event in events))
        self.assertEqual(len(tts.synthesized), 2)
        speech_events = [
            event for event in events if event.kind == "audio.speech_started"
        ]
        self.assertTrue(speech_events)
        self.assertFalse(any(event.data.get("barge_in") for event in speech_events))

    async def test_voice_profile_refreshes_for_each_eligible_new_turn(self) -> None:
        config = apply_overrides(
            load_config(ROOT / "configs" / "mock.toml"),
            audio_output=True,
            voice_consent=True,
        )
        config = replace(
            config,
            streaming=replace(config.streaming, barge_in_enabled=False),
        )
        events: list[PipelineEvent] = []
        tts = MockCloningTTS()
        pipeline = RealtimeInterpretationPipeline(
            config,
            asr=ScriptedASR(["第一位说话者。", "第二位说话者。"]),
            translator=RuleBasedMockTranslator(),
            endpoint=HeuristicSemanticEndpoint(),
            tts=tts,
            audio_sink=NullAudioSink(),
            event_handler=events.append,
        )
        started = time.monotonic()
        await pipeline.run(
            Source(
                [
                    window(1, 3000, True, started),
                    window(2, 3000, True, started),
                ]
            )
        )
        self.assertEqual(tts.enroll_calls, 2)
        self.assertEqual(len(tts.synthesized), 2)
        first_profile = tts.synthesized[0][1]
        second_profile = tts.synthesized[1][1]
        self.assertIsNotNone(first_profile)
        self.assertIsNotNone(second_profile)
        self.assertNotEqual(first_profile, second_profile)
        self.assertEqual(len(tts.deleted), 2)
        ready = [event for event in events if event.kind == "voice.ready"]
        self.assertEqual([event.data.get("refresh") for event in ready], [False, True])

    async def test_tts_failure_does_not_stop_text_pipeline(self) -> None:
        config = apply_overrides(
            load_config(ROOT / "configs" / "mock.toml"),
            audio_output=True,
            voice_consent=True,
        )
        events: list[PipelineEvent] = []

        class BrokenTTS(MockCloningTTS):
            async def synthesize(self, text, *, language, profile_id):
                if False:
                    yield AudioChunk(b"", self.sample_rate)
                raise RuntimeError("speaker unavailable")

        pipeline = RealtimeInterpretationPipeline(
            config,
            asr=ScriptedASR(["测试完成。"]),
            translator=RuleBasedMockTranslator(),
            endpoint=HeuristicSemanticEndpoint(),
            tts=BrokenTTS(),
            audio_sink=NullAudioSink(),
            event_handler=events.append,
        )
        started = time.monotonic()
        await pipeline.run(Source([window(1, 3_000, True, started)]))
        kinds = [event.kind for event in events]
        self.assertIn("tts.failed", kinds)
        self.assertIn("session.finished", kinds)

    async def test_stale_cumulative_asr_partials_are_coalesced(self) -> None:
        config = apply_overrides(
            load_config(ROOT / "configs" / "mock.toml"),
            audio_output=False,
        )

        class SlowASR(ScriptedASR):
            async def transcribe(self, window, *, language):
                await asyncio.sleep(0.03)
                return await super().transcribe(window, language=language)

        asr = SlowASR(["大家好", "大家好，欢迎参加会议。"])
        pipeline = RealtimeInterpretationPipeline(
            config,
            asr=asr,
            translator=RuleBasedMockTranslator(),
            endpoint=HeuristicSemanticEndpoint(),
            tts=None,
            audio_sink=NullAudioSink(),
            event_handler=lambda event: None,
        )
        started = time.monotonic()
        await pipeline.run(
            Source(
                [
                    window(1, 800, False, started),
                    window(1, 1_600, False, started),
                    window(1, 2_400, False, started),
                    window(1, 3_200, True, started),
                ]
            )
        )
        self.assertLess(asr.calls, 4)
        self.assertEqual(asr.calls, 2)

    async def test_first_translation_waits_for_later_voice_enrollment(self) -> None:
        config = apply_overrides(
            load_config(ROOT / "configs" / "mock.toml"),
            audio_output=True,
            voice_consent=True,
        )
        events: list[PipelineEvent] = []

        async def collect(event: PipelineEvent) -> None:
            events.append(event)

        tts = MockCloningTTS()
        pipeline = RealtimeInterpretationPipeline(
            config,
            asr=ScriptedASR(["你好。", "继续测试。"]),
            translator=RuleBasedMockTranslator(),
            endpoint=HeuristicSemanticEndpoint(),
            tts=tts,
            audio_sink=NullAudioSink(),
            event_handler=collect,
        )
        started = time.monotonic()
        await pipeline.run(
            Source(
                [
                    window(1, 1_000, True, started),
                    window(2, 2_000, True, started),
                ]
            )
        )
        self.assertEqual(tts.enroll_calls, 1)
        self.assertEqual(len(tts.synthesized), 2)
        self.assertFalse(any(event.kind == "tts.skipped" for event in events))
        self.assertTrue(
            any(event.kind == "tts.waiting_for_voice" for event in events)
        )

    async def test_full_pipeline_enrolls_once_and_synthesizes_cloned_voice(self) -> None:
        config = apply_overrides(
            load_config(ROOT / "configs" / "mock.toml"),
            audio_output=True,
            voice_consent=True,
        )
        events: list[PipelineEvent] = []

        async def collect(event: PipelineEvent) -> None:
            events.append(event)

        tts = MockCloningTTS()
        pipeline = RealtimeInterpretationPipeline(
            config,
            asr=ScriptedASR(
                ["大家好", "大家好，欢迎", "大家好，欢迎参加今天的会议。"]
            ),
            translator=RuleBasedMockTranslator(),
            endpoint=HeuristicSemanticEndpoint(),
            tts=tts,
            audio_sink=NullAudioSink(),
            event_handler=collect,
        )
        started = time.monotonic()
        await pipeline.run(
            Source(
                [
                    window(1, 1200, False, started),
                    window(1, 2600, False, started),
                    window(1, 3300, True, started),
                ]
            )
        )
        kinds = [event.kind for event in events]
        self.assertIn("voice.ready", kinds)
        self.assertIn("translation.update", kinds)
        self.assertIn("tts.started", kinds)
        self.assertEqual(tts.enroll_calls, 1)
        self.assertEqual(len(tts.synthesized), 1)
        self.assertTrue(tts.synthesized[0][1].startswith("mock-"))
        self.assertEqual(len(tts.deleted), 1)
        metrics = next(event for event in events if event.kind == "turn.metrics")
        self.assertIsNotNone(metrics.data["first_audio_ms"])

    async def test_semantic_incomplete_segments_merge_into_one_logical_turn(self) -> None:
        config = apply_overrides(
            load_config(ROOT / "configs" / "mock.toml"),
            audio_output=False,
        )
        events: list[PipelineEvent] = []

        async def collect(event: PipelineEvent) -> None:
            events.append(event)

        pipeline = RealtimeInterpretationPipeline(
            config,
            asr=ScriptedASR(["因为", "天气很好。"]),
            translator=RuleBasedMockTranslator(),
            endpoint=HeuristicSemanticEndpoint(),
            tts=None,
            audio_sink=NullAudioSink(),
            event_handler=collect,
        )
        started = time.monotonic()
        await pipeline.run(
            Source(
                [
                    window(1, 900, True, started),
                    window(2, 900, True, started),
                ]
            )
        )
        decisions = [
            event for event in events if event.kind == "endpoint.decision"
        ]
        self.assertEqual([event.data["complete"] for event in decisions], [False, True])
        finals = [
            event
            for event in events
            if event.kind == "transcript.update" and event.data.get("is_final")
        ]
        self.assertEqual(len(finals), 1)
        self.assertEqual(finals[0].turn_id, 1)
        self.assertIn("因为", finals[0].data["committed"])
        self.assertIn("天气很好。", finals[0].data["committed"])

    async def test_barge_in_cancels_audio_from_old_playback_epoch(self) -> None:
        config = apply_overrides(
            load_config(ROOT / "configs" / "mock.toml"),
            audio_output=True,
            voice_consent=True,
        )
        config = replace(
            config,
            streaming=replace(config.streaming, barge_in_enabled=True),
        )
        events: list[PipelineEvent] = []
        tts_started = asyncio.Event()

        async def collect(event: PipelineEvent) -> None:
            events.append(event)
            if event.kind == "tts.started" and event.turn_id == 1:
                tts_started.set()

        class SlowTTS(MockCloningTTS):
            async def synthesize(self, text, *, language, profile_id):
                self.synthesized.append((text, profile_id))
                yield AudioChunk(b"\x00\x00" * 100, self.sample_rate)
                await asyncio.sleep(0.05)
                yield AudioChunk(b"\x00\x00" * 100, self.sample_rate)

        tts = SlowTTS()
        pipeline = RealtimeInterpretationPipeline(
            config,
            asr=ScriptedASR(["你好。", "继续。"]),
            translator=RuleBasedMockTranslator(),
            endpoint=HeuristicSemanticEndpoint(),
            tts=tts,
            audio_sink=NullAudioSink(),
            event_handler=collect,
        )
        started = time.monotonic()

        class InterruptingSource:
            async def __aiter__(self):
                await pipeline.on_speech_started()
                yield window(1, 3000, True, started)
                await asyncio.wait_for(tts_started.wait(), timeout=1)
                await pipeline.on_speech_started()
                yield window(2, 800, True, started)

        await pipeline.run(InterruptingSource())
        cancelled = [event for event in events if event.kind == "tts.cancelled"]
        self.assertTrue(cancelled)
        barge_in = [
            event
            for event in events
            if event.kind == "audio.speech_started" and event.data.get("barge_in")
        ]
        self.assertTrue(barge_in)


if __name__ == "__main__":
    unittest.main()
