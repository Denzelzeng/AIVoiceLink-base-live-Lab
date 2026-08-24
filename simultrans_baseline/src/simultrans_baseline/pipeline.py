from __future__ import annotations

import asyncio
import time
import uuid
from collections import deque
from dataclasses import dataclass

from .config import AppConfig
from .contracts import (
    ASRBackend,
    AudioSink,
    AudioWindowSource,
    CloningTTSBackend,
    EventHandler,
    SemanticEndpointBackend,
    TranslationBackend,
)
from .events import AudioWindow, PipelineEvent
from .stability import AgreementCommitter, PhraseBuffer
from .voice import VoiceEnrollmentManager


@dataclass(frozen=True)
class _TranslationWork:
    turn_id: int
    source_text: str
    is_final: bool


@dataclass(frozen=True)
class _TTSWork:
    turn_id: int
    segment_id: str
    text: str
    playback_epoch: int
    is_final: bool = False


def _join_text(prefix: str, suffix: str) -> str:
    prefix = prefix.strip()
    suffix = suffix.strip()
    if not prefix:
        return suffix
    if not suffix:
        return prefix
    separator = "" if suffix[:1] in "，。！？；：,.!?;:" else " "
    return prefix + separator + suffix


class RealtimeInterpretationPipeline:
    """Concurrent ASR -> re-translation -> cloned-TTS orchestration."""

    def __init__(
        self,
        config: AppConfig,
        *,
        asr: ASRBackend,
        translator: TranslationBackend,
        endpoint: SemanticEndpointBackend,
        tts: CloningTTSBackend | None,
        audio_sink: AudioSink,
        event_handler: EventHandler,
    ):
        self.config = config
        self.asr = asr
        self.translator = translator
        self.endpoint = endpoint
        self.tts = tts
        self.audio_sink = audio_sink
        self.event_handler = event_handler
        self.session_id = uuid.uuid4().hex
        self._playback_epoch = 0
        self._playing = False
        self._closed = False
        self._speech_idle = asyncio.Event()
        self._speech_idle.set()
        self._timing: dict[int, dict[str, float]] = {}
        self._voice = (
            VoiceEnrollmentManager(
                tts,
                config.voice_clone,
                session_id=self.session_id,
                source_language=config.session.source_language,
                emit=self._dispatch,
            )
            if tts and config.session.audio_output and config.voice_clone.enabled
            else None
        )

    async def run(self, source: AudioWindowSource) -> None:
        capacity = self.config.streaming.queue_capacity
        window_queue: asyncio.Queue[AudioWindow | None] = asyncio.Queue(capacity)
        mt_queue: asyncio.Queue[_TranslationWork | None] = asyncio.Queue(capacity)
        # Text/audio output is an independent chain.  Do not back-pressure ASR/MT
        # merely because speech is currently longer than synthesized playback.
        tts_queue: asyncio.Queue[_TTSWork | None] = asyncio.Queue()

        await self._emit("session.started", data={
            "source_language": self.config.session.source_language,
            "target_language": self.config.session.target_language,
            "audio_output": self.config.session.audio_output,
            "voice_clone": bool(self._voice),
        })

        producer = asyncio.create_task(self._produce_windows(source, window_queue))
        asr_worker = asyncio.create_task(self._asr_worker(window_queue, mt_queue))
        mt_worker = asyncio.create_task(self._mt_worker(mt_queue, tts_queue))
        tts_worker = asyncio.create_task(self._tts_worker(tts_queue))
        try:
            await asyncio.gather(producer, asr_worker, mt_worker, tts_worker)
            await self._emit("session.finished")
        except Exception as exc:
            for task in (producer, asr_worker, mt_worker, tts_worker):
                if not task.done():
                    task.cancel()
            await asyncio.gather(
                producer, asr_worker, mt_worker, tts_worker, return_exceptions=True
            )
            await self._emit("pipeline.error", data={"error": str(exc)})
            raise
        finally:
            await self.aclose()

    async def on_speech_started(self) -> None:
        """Called by the audio source immediately when VAD opens the gate."""
        if not self.config.streaming.barge_in_enabled:
            await self._emit(
                "audio.speech_started",
                data={"barge_in": False, "playback_epoch": self._playback_epoch},
            )
            return
        self._speech_idle.clear()
        interrupted = self._playing
        if interrupted and self.config.streaming.barge_in_enabled:
            self._playback_epoch += 1
            try:
                await self.audio_sink.interrupt()
            except Exception as exc:
                await self._emit(
                    "audio.interrupt_failed",
                    data={"error": str(exc)},
                )
        await self._emit(
            "audio.speech_started",
            data={
                "barge_in": interrupted and self.config.streaming.barge_in_enabled,
                "playback_epoch": self._playback_epoch,
            },
        )

    async def _produce_windows(
        self,
        source: AudioWindowSource,
        queue: asyncio.Queue[AudioWindow | None],
    ) -> None:
        try:
            async for window in source:
                await queue.put(window)
                if window.is_final:
                    self._speech_idle.set()
        finally:
            self._speech_idle.set()
            await queue.put(None)

    async def _asr_worker(
        self,
        queue: asyncio.Queue[AudioWindow | None],
        mt_queue: asyncio.Queue[_TranslationWork | None],
    ) -> None:
        logical_turn_id = 1
        logical_prefix = ""
        current_segment_id: int | None = None
        segment_committer: AgreementCommitter | None = None
        last_sent = ""
        last_aggregate = ""
        logical_audio = bytearray()
        audio_seen_by_segment: dict[int, int] = {}
        endpoint_generation = 0
        endpoint_task: asyncio.Task[None] | None = None

        async def hard_finalize(
            generation: int,
            turn_id: int,
            source_text: str,
        ) -> None:
            nonlocal logical_turn_id, logical_prefix, current_segment_id
            nonlocal segment_committer, last_sent, endpoint_generation
            nonlocal logical_audio, audio_seen_by_segment, last_aggregate
            await asyncio.sleep(
                self.config.streaming.semantic_hard_timeout_ms / 1_000
            )
            if generation != endpoint_generation or turn_id != logical_turn_id:
                return
            await self._emit(
                "endpoint.hard_timeout",
                turn_id=turn_id,
                data={"source": source_text},
            )
            await self._emit(
                "transcript.update",
                turn_id=turn_id,
                data={
                    "committed": source_text,
                    "unstable": "",
                    "is_final": True,
                    "reason": "semantic hard timeout",
                },
            )
            if source_text:
                await mt_queue.put(_TranslationWork(turn_id, source_text, True))
            logical_turn_id += 1
            logical_prefix = ""
            current_segment_id = None
            segment_committer = None
            last_sent = ""
            last_aggregate = ""
            logical_audio = bytearray()
            audio_seen_by_segment = {}
            endpoint_generation += 1

        try:
            while True:
                window = await queue.get()
                if window is None:
                    break
                input_closed = False
                if not window.is_final:
                    # Cumulative partials become stale quickly when cloud ASR latency
                    # exceeds the capture interval. Keep only the newest queued
                    # snapshot for this acoustic segment.
                    while True:
                        try:
                            newer = queue.get_nowait()
                        except asyncio.QueueEmpty:
                            break
                        if newer is None:
                            input_closed = True
                            break
                        window = newer
                        if window.is_final:
                            break
                if window.turn_id != current_segment_id:
                    endpoint_generation += 1
                    if endpoint_task and not endpoint_task.done():
                        endpoint_task.cancel()
                    current_segment_id = window.turn_id
                    segment_committer = AgreementCommitter(
                        self.config.streaming.asr_agreement_depth
                    )

                assert segment_committer is not None
                already_seen = audio_seen_by_segment.get(window.turn_id, 0)
                if len(window.pcm) > already_seen:
                    logical_audio.extend(window.pcm[already_seen:])
                    audio_seen_by_segment[window.turn_id] = len(window.pcm)
                timing = self._timing.setdefault(
                    logical_turn_id, {"audio_started": window.started_at}
                )
                hypothesis = await self.asr.transcribe(
                    window,
                    language=self.config.session.source_language,
                )
                if self._voice:
                    await self._voice.observe(
                        window,
                        hypothesis,
                        voice_turn_id=logical_turn_id,
                    )
                update = segment_committer.update(
                    hypothesis,
                    is_final=window.is_final,
                )
                aggregate = _join_text(logical_prefix, update.committed)
                last_aggregate = aggregate
                if update.commit_delta and "first_source_commit" not in timing:
                    timing["first_source_commit"] = time.monotonic()

                decision = None
                logical_final = False
                if window.is_final:
                    max_endpoint_bytes = round(
                        self.config.endpoint.max_audio_ms
                        / 1_000
                        * window.sample_rate
                        * 2
                    )
                    endpoint_pcm = bytes(logical_audio[-max_endpoint_bytes:])
                    endpoint_window = AudioWindow(
                        turn_id=logical_turn_id,
                        pcm=endpoint_pcm,
                        sample_rate=window.sample_rate,
                        is_final=True,
                        started_at=window.started_at,
                        captured_at=window.captured_at,
                    )
                    decision = await self.endpoint.classify(
                        endpoint_window,
                        transcript=hypothesis,
                        language=self.config.session.source_language,
                    )
                    logical_final = decision.complete
                    await self._emit(
                        "endpoint.decision",
                        turn_id=logical_turn_id,
                        data={
                            "complete": decision.complete,
                            "probability": decision.probability,
                            "reason": decision.reason,
                            "acoustic_segment_id": window.turn_id,
                        },
                    )

                await self._emit(
                    "transcript.update",
                    turn_id=logical_turn_id,
                    data={
                        "committed": aggregate,
                        "unstable": update.unstable,
                        "hypothesis": hypothesis,
                        "is_final": logical_final,
                        "acoustic_segment_id": window.turn_id,
                    },
                )

                growth = len(aggregate) - len(last_sent)
                if aggregate and (
                    logical_final
                    or growth >= self.config.streaming.min_source_growth_chars
                ):
                    await mt_queue.put(
                        _TranslationWork(logical_turn_id, aggregate, logical_final)
                    )
                    last_sent = aggregate

                if not window.is_final:
                    if input_closed:
                        break
                    continue
                if logical_final:
                    logical_turn_id += 1
                    logical_prefix = ""
                    current_segment_id = None
                    segment_committer = None
                    last_sent = ""
                    last_aggregate = ""
                    endpoint_generation += 1
                    logical_audio = bytearray()
                    audio_seen_by_segment = {}
                    continue

                logical_prefix = aggregate
                current_segment_id = None
                segment_committer = None
                endpoint_generation += 1
                generation = endpoint_generation
                endpoint_task = asyncio.create_task(
                    hard_finalize(generation, logical_turn_id, aggregate)
                )
                if input_closed:
                    break

            if endpoint_task and not endpoint_task.done():
                endpoint_generation += 1
                endpoint_task.cancel()
                await asyncio.gather(endpoint_task, return_exceptions=True)
            if last_aggregate:
                await self._emit(
                    "transcript.update",
                    turn_id=logical_turn_id,
                    data={
                        "committed": last_aggregate,
                        "unstable": "",
                        "is_final": True,
                        "reason": "input closed",
                    },
                )
                await mt_queue.put(
                    _TranslationWork(logical_turn_id, last_aggregate, True)
                )
        finally:
            await mt_queue.put(None)

    async def _mt_worker(
        self,
        queue: asyncio.Queue[_TranslationWork | None],
        tts_queue: asyncio.Queue[_TTSWork | None],
    ) -> None:
        committers: dict[int, AgreementCommitter] = {}
        phrase_buffers: dict[int, PhraseBuffer] = {}
        context: deque[tuple[str, str]] = deque(
            maxlen=self.config.streaming.recent_context_turns
        )
        segment_numbers: dict[int, int] = {}
        try:
            while True:
                work = await queue.get()
                if work is None:
                    break
                committer = committers.setdefault(
                    work.turn_id,
                    AgreementCommitter(self.config.streaming.mt_agreement_depth),
                )
                phrase_buffer = phrase_buffers.setdefault(
                    work.turn_id,
                    PhraseBuffer(self.config.streaming.tts_min_phrase_chars),
                )

                async def on_delta(text: str) -> None:
                    await self._emit(
                        "translation.delta",
                        turn_id=work.turn_id,
                        data={"delta": text},
                    )

                hypothesis = await self.translator.translate(
                    work.source_text,
                    source_language=self.config.session.source_language,
                    target_language=self.config.session.target_language,
                    committed_target=committer.committed,
                    context=list(context),
                    domain=self.config.session.domain,
                    on_delta=on_delta,
                )
                update = committer.update(hypothesis, is_final=work.is_final)
                timing = self._timing.setdefault(
                    work.turn_id, {"audio_started": time.monotonic()}
                )
                if update.commit_delta and "first_target_commit" not in timing:
                    timing["first_target_commit"] = time.monotonic()
                await self._emit(
                    "translation.update",
                    turn_id=work.turn_id,
                    data={
                        "committed": update.committed,
                        "speculative": update.unstable,
                        "hypothesis": hypothesis,
                        "is_final": work.is_final,
                    },
                )

                phrases = phrase_buffer.push(
                    update.commit_delta,
                    is_final=work.is_final,
                )
                if self.config.session.audio_output and self.tts:
                    for index, phrase in enumerate(phrases):
                        number = segment_numbers.get(work.turn_id, 0) + 1
                        segment_numbers[work.turn_id] = number
                        await tts_queue.put(
                            _TTSWork(
                                turn_id=work.turn_id,
                                segment_id=f"turn-{work.turn_id:04d}-part-{number:03d}",
                                text=phrase,
                                playback_epoch=self._playback_epoch,
                                is_final=work.is_final and index == len(phrases) - 1,
                            )
                        )

                if work.is_final:
                    context.append((work.source_text, update.committed))
                    if self.config.session.audio_output and self.tts:
                        if not phrases:
                            await tts_queue.put(
                                _TTSWork(
                                    turn_id=work.turn_id,
                                    segment_id=f"turn-{work.turn_id:04d}-final",
                                    text="",
                                    playback_epoch=self._playback_epoch,
                                    is_final=True,
                                )
                            )
                    else:
                        await self._emit_turn_metrics(work.turn_id)
                    committers.pop(work.turn_id, None)
                    phrase_buffers.pop(work.turn_id, None)
        finally:
            await tts_queue.put(None)

    async def _tts_worker(self, queue: asyncio.Queue[_TTSWork | None]) -> None:
        while True:
            work = await queue.get()
            if work is None:
                return
            if not work.text:
                if work.is_final:
                    await self._emit_turn_metrics(work.turn_id)
                continue
            if work.playback_epoch != self._playback_epoch:
                await self._emit(
                    "tts.cancelled",
                    turn_id=work.turn_id,
                    data={"segment_id": work.segment_id, "reason": "barge-in"},
                )
                if work.is_final:
                    await self._emit_turn_metrics(work.turn_id)
                continue
            assert self.tts is not None
            profile_id: str | None = None
            if self._voice:
                if self._voice.profile is None:
                    await self._emit(
                        "tts.waiting_for_voice",
                        turn_id=work.turn_id,
                        data={
                            "segment_id": work.segment_id,
                            "timeout_ms": self.config.voice_clone.wait_timeout_ms,
                        },
                    )
                profile = await self._voice.wait_for_profile(work.turn_id)
                profile_id = profile.profile_id if profile else None
                if profile_id is None and self.config.voice_clone.fallback_policy == "skip":
                    await self._emit(
                        "tts.skipped",
                        turn_id=work.turn_id,
                        data={
                            "segment_id": work.segment_id,
                            "reason": "cloned voice is not ready",
                        },
                    )
                    if work.is_final:
                        await self._emit_turn_metrics(work.turn_id)
                    continue
            if self.config.streaming.barge_in_enabled:
                await self._speech_idle.wait()
            if work.playback_epoch != self._playback_epoch:
                await self._emit(
                    "tts.cancelled",
                    turn_id=work.turn_id,
                    data={
                        "segment_id": work.segment_id,
                        "reason": "barge-in while waiting",
                    },
                )
                if work.is_final:
                    await self._emit_turn_metrics(work.turn_id)
                continue
            self._playing = True
            first_chunk = True
            cancelled_during_stream = False
            try:
                await self._emit(
                    "tts.started",
                    turn_id=work.turn_id,
                    data={
                        "segment_id": work.segment_id,
                        "text": work.text,
                        "profile_id": profile_id,
                    },
                )
                async for chunk in self.tts.synthesize(
                    work.text,
                    language=self.config.session.target_language,
                    profile_id=profile_id,
                ):
                    if work.playback_epoch != self._playback_epoch:
                        await self._emit(
                            "tts.cancelled",
                            turn_id=work.turn_id,
                            data={
                                "segment_id": work.segment_id,
                                "reason": "barge-in during synthesis",
                            },
                        )
                        cancelled_during_stream = True
                        break
                    if first_chunk:
                        timing = self._timing.setdefault(
                            work.turn_id, {"audio_started": time.monotonic()}
                        )
                        timing.setdefault("first_audio", time.monotonic())
                        first_chunk = False
                    await self.audio_sink.write(chunk, segment_id=work.segment_id)
                else:
                    await self._emit(
                        "tts.finished",
                        turn_id=work.turn_id,
                        data={"segment_id": work.segment_id},
                    )
                    if work.is_final:
                        await self._emit_turn_metrics(work.turn_id)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                await self._emit(
                    "tts.failed",
                    turn_id=work.turn_id,
                    data={
                        "segment_id": work.segment_id,
                        "error": str(exc),
                    },
                )
                if work.is_final:
                    await self._emit_turn_metrics(work.turn_id)
            finally:
                self._playing = False
            if cancelled_during_stream and work.is_final:
                await self._emit_turn_metrics(work.turn_id)

    async def _emit_turn_metrics(self, turn_id: int) -> None:
        timing = self._timing.get(turn_id, {})
        started = timing.get("audio_started")
        values: dict[str, float | None] = {}
        for key in ("first_source_commit", "first_target_commit", "first_audio"):
            value = timing.get(key)
            values[f"{key}_ms"] = (
                round((value - started) * 1_000, 1)
                if value is not None and started is not None
                else None
            )
        await self._emit("turn.metrics", turn_id=turn_id, data=values)

    async def _emit(
        self,
        kind: str,
        *,
        turn_id: int | None = None,
        data: dict[str, object] | None = None,
    ) -> None:
        await self._dispatch(
            PipelineEvent(
                kind=kind,
                session_id=self.session_id,
                turn_id=turn_id,
                data=dict(data or {}),
            )
        )

    async def _dispatch(self, event: PipelineEvent) -> None:
        result = self.event_handler(event)
        if asyncio.iscoroutine(result):
            await result

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._voice:
            await self._voice.aclose()
        await self.asr.aclose()
        await self.translator.aclose()
        await self.endpoint.aclose()
        if self.tts:
            await self.tts.aclose()
        await self.audio_sink.aclose()
