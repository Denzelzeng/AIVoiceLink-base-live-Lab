from __future__ import annotations

import asyncio
import time
import unittest
from dataclasses import replace
from pathlib import Path

from simultrans_baseline.config import SpeakerChangeConfig, apply_overrides, load_config
from simultrans_baseline.endpoint import HeuristicSemanticEndpoint
from simultrans_baseline.events import AudioChunk, AudioWindow, PipelineEvent
from simultrans_baseline.pipeline import RealtimeInterpretationPipeline
from simultrans_baseline.providers.mock import (
    MockCloningTTS,
    RuleBasedMockTranslator,
    ScriptedASR,
)
from simultrans_baseline.speaker import SpeakerChangeDetector
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
    async def test_tts_prebuffer_smooths_initial_cloud_chunks(self) -> None:
        config = apply_overrides(
            load_config(ROOT / "configs" / "mock.toml"),
            audio_output=True,
            voice_consent=True,
        )
        config = replace(
            config,
            streaming=replace(config.streaming, tts_prebuffer_ms=250),
        )

        class ChunkedTTS(MockCloningTTS):
            def __init__(self):
                super().__init__()
                self.yield_count = 0

            async def synthesize(self, text, *, language, profile_id):
                self.synthesized.append((text, profile_id))
                chunk = AudioChunk(b"\x00\x00" * 2400, 24_000)
                for _ in range(3):
                    self.yield_count += 1
                    yield chunk
                    await asyncio.sleep(0)

        class RecordingSink:
            def __init__(self, tts):
                self.tts = tts
                self.yields_at_first_write = None

            async def write(self, chunk, *, segment_id):
                del chunk, segment_id
                if self.yields_at_first_write is None:
                    self.yields_at_first_write = self.tts.yield_count

            async def interrupt(self):
                return None

            async def aclose(self):
                return None

        tts = ChunkedTTS()
        sink = RecordingSink(tts)
        events: list[PipelineEvent] = []
        pipeline = RealtimeInterpretationPipeline(
            config,
            asr=ScriptedASR(["测试连续播放。"]),
            translator=RuleBasedMockTranslator(),
            endpoint=HeuristicSemanticEndpoint(),
            tts=tts,
            audio_sink=sink,
            event_handler=events.append,
        )
        started = time.monotonic()
        await pipeline.run(Source([window(1, 3000, True, started)]))
        self.assertEqual(sink.yields_at_first_write, 3)
        metrics = next(event for event in events if event.kind == "turn.metrics")
        self.assertIn("first_cloud_audio_ms", metrics.data)
        self.assertIn("playback_queue_ms", metrics.data)

    async def test_next_tts_is_generated_while_previous_audio_is_playing(self) -> None:
        config = apply_overrides(
            load_config(ROOT / "configs" / "mock.toml"),
            audio_output=True,
            voice_consent=True,
        )
        synthesis_started: list[float] = []

        class FastTTS(MockCloningTTS):
            async def synthesize(self, text, *, language, profile_id):
                self.synthesized.append((text, profile_id))
                synthesis_started.append(time.monotonic())
                yield AudioChunk(b"\x00\x00" * 100, self.sample_rate)
                yield AudioChunk(b"\x00\x00" * 100, self.sample_rate)

        class SlowSink:
            def __init__(self):
                self.first_segment: str | None = None
                self.first_writes = 0
                self.first_finished: float | None = None

            async def write(self, chunk, *, segment_id):
                if self.first_segment is None:
                    self.first_segment = segment_id
                await asyncio.sleep(0.05)
                if segment_id == self.first_segment:
                    self.first_writes += 1
                    if self.first_writes == 2:
                        self.first_finished = time.monotonic()

            async def interrupt(self):
                return None

            async def aclose(self):
                return None

        sink = SlowSink()
        pipeline = RealtimeInterpretationPipeline(
            config,
            asr=ScriptedASR(["第一句话。", "第二句话。"]),
            translator=RuleBasedMockTranslator(),
            endpoint=HeuristicSemanticEndpoint(),
            tts=FastTTS(),
            audio_sink=sink,
            event_handler=lambda event: None,
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
        self.assertEqual(len(synthesis_started), 2)
        self.assertIsNotNone(sink.first_finished)
        self.assertLess(synthesis_started[1], sink.first_finished)

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

    async def test_speaker_gate_refreshes_only_after_detected_change(self) -> None:
        config = apply_overrides(
            load_config(ROOT / "configs" / "mock.toml"),
            audio_output=True,
            voice_consent=True,
        )
        events: list[PipelineEvent] = []
        tts = MockCloningTTS()

        class FakeEmbedder:
            def __init__(self):
                self.values = iter(
                    ([1.0, 0.0], [0.99, 0.01], [0.0, 1.0])
                )

            def embed(self, pcm, *, sample_rate):
                del pcm, sample_rate
                return next(self.values)

        speaker = SpeakerChangeDetector(
            FakeEmbedder(),
            SpeakerChangeConfig(enabled=True, model_path="mock"),
        )
        pipeline = RealtimeInterpretationPipeline(
            config,
            asr=ScriptedASR(["第一句。", "还是我。", "换人了。"]),
            translator=RuleBasedMockTranslator(),
            endpoint=HeuristicSemanticEndpoint(),
            tts=tts,
            audio_sink=NullAudioSink(),
            event_handler=events.append,
            speaker_change_detector=speaker,
        )
        started = time.monotonic()
        await pipeline.run(
            Source(
                [
                    window(1, 3_000, True, started),
                    window(2, 3_000, True, started),
                    window(3, 3_000, True, started),
                ]
            )
        )
        self.assertEqual(tts.enroll_calls, 2)
        decisions = [event for event in events if event.kind == "speaker.decision"]
        self.assertEqual(
            [event.data["state"] for event in decisions],
            ["initial", "same", "changed"],
        )

    async def test_bootstrap_does_not_mix_short_turns_from_two_speakers(self) -> None:
        config = apply_overrides(
            load_config(ROOT / "configs" / "mock.toml"),
            audio_output=True,
            voice_consent=True,
        )
        config = replace(
            config,
            voice_clone=replace(config.voice_clone, min_reference_ms=3_000),
        )
        events: list[PipelineEvent] = []

        class FakeEmbedder:
            def __init__(self):
                self.values = iter(([1.0, 0.0], [0.0, 1.0], [0.01, 0.99]))

            def embed(self, pcm, *, sample_rate):
                del pcm, sample_rate
                return next(self.values)

        speaker = SpeakerChangeDetector(
            FakeEmbedder(),
            SpeakerChangeConfig(enabled=True, model_path="mock"),
        )
        tts = MockCloningTTS()
        pipeline = RealtimeInterpretationPipeline(
            config,
            asr=ScriptedASR(["甲。", "乙第一句。", "乙第二句。"]),
            translator=RuleBasedMockTranslator(),
            endpoint=HeuristicSemanticEndpoint(),
            tts=tts,
            audio_sink=NullAudioSink(),
            event_handler=events.append,
            speaker_change_detector=speaker,
        )
        started = time.monotonic()
        await pipeline.run(
            Source(
                [
                    window(1, 2_200, True, started),
                    window(2, 2_200, True, started),
                    window(3, 2_200, True, started),
                ]
            )
        )
        enrollments = [
            event for event in events if event.kind == "voice.enrollment_started"
        ]
        self.assertEqual(tts.enroll_calls, 1)
        self.assertEqual([event.turn_id for event in enrollments], [3])

    async def test_voice_refresh_does_not_block_tts_fifo(self) -> None:
        config = apply_overrides(
            load_config(ROOT / "configs" / "mock.toml"),
            audio_output=True,
            voice_consent=True,
        )

        class SlowRefreshTTS(MockCloningTTS):
            def __init__(self):
                super().__init__()
                self.first_synthesized = asyncio.Event()
                self.refresh_started = asyncio.Event()
                self.second_synthesized = asyncio.Event()
                self.allow_refresh = asyncio.Event()
                self.prepare_calls = 0

            async def enroll(self, *args, **kwargs):
                return await super().enroll(*args, **kwargs)

            async def prepare_voice(self, profile_id, *, language):
                del profile_id, language
                self.prepare_calls += 1
                if self.prepare_calls == 2:
                    self.refresh_started.set()
                    await self.allow_refresh.wait()

            async def synthesize(self, text, *, language, profile_id):
                call_number = len(self.synthesized) + 1
                async for chunk in super().synthesize(
                    text,
                    language=language,
                    profile_id=profile_id,
                ):
                    if call_number == 1:
                        self.first_synthesized.set()
                    elif call_number == 2:
                        self.second_synthesized.set()
                    yield chunk

        tts = SlowRefreshTTS()
        pipeline = RealtimeInterpretationPipeline(
            config,
            asr=ScriptedASR(["第一句话。", "第二句话。"]),
            translator=RuleBasedMockTranslator(),
            endpoint=HeuristicSemanticEndpoint(),
            tts=tts,
            audio_sink=NullAudioSink(),
            event_handler=lambda event: None,
        )
        started = time.monotonic()

        class RefreshingSource:
            async def __aiter__(self):
                yield window(1, 3000, True, started)
                await asyncio.wait_for(tts.first_synthesized.wait(), timeout=1)
                first_profile = tts.synthesized[0][1]
                yield window(2, 3000, True, started)
                await asyncio.wait_for(tts.refresh_started.wait(), timeout=1)
                await asyncio.wait_for(tts.second_synthesized.wait(), timeout=0.1)
                self.second_profile = tts.synthesized[1][1]
                self.first_profile = first_profile
                tts.allow_refresh.set()

        source = RefreshingSource()
        await pipeline.run(source)
        self.assertEqual(source.second_profile, source.first_profile)
        self.assertEqual(tts.enroll_calls, 2)
        self.assertEqual(tts.prepare_calls, 2)

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
        # The pending partials collapse and the in-flight partial is preempted,
        # so only the authoritative final request completes.
        self.assertEqual(asr.calls, 1)

    async def test_slow_asr_never_backpressures_microphone_or_loses_finals(self) -> None:
        config = apply_overrides(
            load_config(ROOT / "configs" / "mock.toml"),
            audio_output=False,
        )
        config = replace(
            config,
            streaming=replace(config.streaming, queue_capacity=1),
        )
        events: list[PipelineEvent] = []

        class GatedASR(ScriptedASR):
            def __init__(self):
                super().__init__([])
                self.started = asyncio.Event()
                self.release = asyncio.Event()

            async def transcribe(self, audio_window, *, language):
                del language
                self.calls += 1
                if self.calls == 1:
                    self.started.set()
                    await self.release.wait()
                return f"第{audio_window.turn_id}句。"

        class BurstSource:
            def __init__(self):
                self.finished = asyncio.Event()

            async def __aiter__(self):
                started = time.monotonic()
                for turn_id in range(1, 13):
                    yield window(turn_id, 600, True, started)
                    await asyncio.sleep(0)
                self.finished.set()

        asr = GatedASR()
        source = BurstSource()
        pipeline = RealtimeInterpretationPipeline(
            config,
            asr=asr,
            translator=RuleBasedMockTranslator(),
            endpoint=HeuristicSemanticEndpoint(),
            tts=None,
            audio_sink=NullAudioSink(),
            event_handler=events.append,
        )
        task = asyncio.create_task(pipeline.run(source))
        await asyncio.wait_for(asr.started.wait(), timeout=1)
        # Even with queue_capacity=1 and ASR stalled, capture drains the burst.
        await asyncio.wait_for(source.finished.wait(), timeout=0.2)
        asr.release.set()
        await asyncio.wait_for(task, timeout=2)

        finals = [
            event
            for event in events
            if event.kind == "transcript.update" and event.data.get("is_final")
        ]
        self.assertEqual(asr.calls, 12)
        self.assertEqual([event.turn_id for event in finals], list(range(1, 13)))

    async def test_final_audio_preempts_a_stuck_partial_asr_request(self) -> None:
        config = apply_overrides(
            load_config(ROOT / "configs" / "mock.toml"),
            audio_output=False,
        )
        events: list[PipelineEvent] = []

        class CancellableASR(ScriptedASR):
            def __init__(self):
                super().__init__([])
                self.partial_started = asyncio.Event()
                self.partial_cancelled = asyncio.Event()

            async def transcribe(self, audio_window, *, language):
                del language
                self.calls += 1
                if not audio_window.is_final:
                    self.partial_started.set()
                    try:
                        await asyncio.Event().wait()
                    except asyncio.CancelledError:
                        self.partial_cancelled.set()
                        raise
                return "最终结果。"

        asr = CancellableASR()

        class FinalizingSource:
            async def __aiter__(self):
                started = time.monotonic()
                yield window(1, 800, False, started)
                await asr.partial_started.wait()
                yield window(1, 1_600, True, started)

        pipeline = RealtimeInterpretationPipeline(
            config,
            asr=asr,
            translator=RuleBasedMockTranslator(),
            endpoint=HeuristicSemanticEndpoint(),
            tts=None,
            audio_sink=NullAudioSink(),
            event_handler=events.append,
        )
        await asyncio.wait_for(pipeline.run(FinalizingSource()), timeout=1)

        self.assertTrue(asr.partial_cancelled.is_set())
        self.assertEqual(asr.calls, 2)
        metrics = next(event for event in events if event.kind == "turn.metrics")
        self.assertEqual(metrics.data["asr_partials_cancelled"], 1.0)

    async def test_stale_translation_partials_collapse_to_latest_final(self) -> None:
        config = apply_overrides(
            load_config(ROOT / "configs" / "mock.toml"),
            audio_output=False,
        )
        config = replace(
            config,
            streaming=replace(
                config.streaming,
                asr_agreement_depth=1,
                min_source_growth_chars=1,
            ),
        )
        events: list[PipelineEvent] = []

        class SignallingASR(ScriptedASR):
            def __init__(self):
                super().__init__([
                    "今天",
                    "今天我们",
                    "今天我们测试",
                    "今天我们测试完成。",
                ])
                self.processed = [asyncio.Event() for _ in range(4)]

            async def transcribe(self, audio_window, *, language):
                result = await super().transcribe(audio_window, language=language)
                self.processed[self.calls - 1].set()
                return result

        class GatedTranslator(RuleBasedMockTranslator):
            def __init__(self):
                super().__init__()
                self.started = asyncio.Event()
                self.cancelled = asyncio.Event()
                self.initiated: list[str] = []
                self._block_first = True

            async def translate(self, source_text, **kwargs):
                self.initiated.append(source_text)
                if self._block_first:
                    self._block_first = False
                    self.started.set()
                    try:
                        await asyncio.Event().wait()
                    except asyncio.CancelledError:
                        self.cancelled.set()
                        raise
                return await super().translate(source_text, **kwargs)

        asr = SignallingASR()
        translator = GatedTranslator()

        class PacedSource:
            async def __aiter__(self):
                started = time.monotonic()
                windows = [
                    window(1, 800, False, started),
                    window(1, 1_600, False, started),
                    window(1, 2_400, False, started),
                    window(1, 3_200, True, started),
                ]
                for index, audio_window in enumerate(windows):
                    yield audio_window
                    await asr.processed[index].wait()
                    if index == 0:
                        await translator.started.wait()

        pipeline = RealtimeInterpretationPipeline(
            config,
            asr=asr,
            translator=translator,
            endpoint=HeuristicSemanticEndpoint(),
            tts=None,
            audio_sink=NullAudioSink(),
            event_handler=events.append,
        )
        task = asyncio.create_task(pipeline.run(PacedSource()))
        await asyncio.wait_for(asr.processed[-1].wait(), timeout=1)
        await asyncio.wait_for(translator.cancelled.wait(), timeout=1)
        await asyncio.wait_for(task, timeout=2)

        self.assertEqual(
            translator.initiated,
            ["今天", "今天我们测试完成。"],
        )
        self.assertEqual(
            translator.calls,
            ["今天我们测试完成。"],
        )
        metrics = next(event for event in events if event.kind == "turn.metrics")
        self.assertEqual(metrics.data["mt_updates_coalesced"], 2.0)
        self.assertEqual(metrics.data["mt_partials_cancelled"], 1.0)
        self.assertIsNotNone(metrics.data["final_mt_queue_ms"])
        self.assertIsNotNone(metrics.data["final_mt_request_ms"])

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
