from __future__ import annotations

import asyncio
import time
import uuid
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Generic, TypeVar

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
from .events import AudioChunk, AudioWindow, PipelineEvent
from .speaker import SpeakerChangeDetector
from .stability import AgreementCommitter, PhraseBuffer
from .voice import VoiceEnrollmentManager


@dataclass(frozen=True)
class _TranslationWork:
    turn_id: int
    source_text: str
    is_final: bool
    enqueued_at: float = field(default_factory=time.monotonic)


@dataclass(frozen=True)
class _TTSWork:
    turn_id: int
    segment_id: str
    text: str
    playback_epoch: int
    is_final: bool = False
    enqueued_at: float = field(default_factory=time.monotonic)


@dataclass(frozen=True)
class _PlaybackWork:
    turn_id: int
    segment_id: str
    text: str
    chunk: AudioChunk | None
    playback_epoch: int
    is_first: bool = False
    is_last: bool = False
    is_final: bool = False


_WorkT = TypeVar("_WorkT")


@dataclass(frozen=True)
class _QueuedWork(Generic[_WorkT]):
    value: _WorkT
    enqueued_at: float
    coalesced: int = 0


class _LatestPartialMailbox(Generic[_WorkT]):
    """Lossless finals with at most one pending snapshot per logical key.

    ``put`` is deliberately synchronous: microphone capture must never wait for
    a slow cloud request.  A newer partial replaces the queued partial for the
    same turn, and a final replaces that partial while retaining its position.
    Finals for different turns are never discarded or reordered.
    """

    def __init__(
        self,
        *,
        key: Callable[[_WorkT], int],
        is_final: Callable[[_WorkT], bool],
    ) -> None:
        self._key = key
        self._is_final = is_final
        self._items: deque[_QueuedWork[_WorkT]] = deque()
        self._available = asyncio.Event()
        self._changed = asyncio.Event()
        self._closed = False

    def put(self, value: _WorkT) -> None:
        if self._closed:
            raise RuntimeError("mailbox is closed")
        key = self._key(value)
        queued = _QueuedWork(value=value, enqueued_at=time.monotonic())
        for index, current in enumerate(self._items):
            if self._key(current.value) != key:
                continue
            if self._is_final(current.value):
                # A final is authoritative; a late partial cannot supersede it.
                return
            self._items[index] = _QueuedWork(
                value=value,
                enqueued_at=queued.enqueued_at,
                coalesced=current.coalesced + 1,
            )
            self._available.set()
            self._changed.set()
            return
        self._items.append(queued)
        self._available.set()
        self._changed.set()

    def close(self) -> None:
        self._closed = True
        self._available.set()
        self._changed.set()

    async def get(self) -> _QueuedWork[_WorkT] | None:
        while not self._items:
            if self._closed:
                return None
            self._available.clear()
            # No task switch occurs between clear and this check, avoiding a
            # missed wake-up without placing a lock on the capture path.
            if self._items or self._closed:
                continue
            await self._available.wait()
        item = self._items.popleft()
        if not self._items and not self._closed:
            self._available.clear()
        return item

    async def wait_for_final(self, key: int) -> bool:
        """Wait until a final for ``key`` is pending, or closure makes it impossible."""
        while True:
            if any(
                self._key(item.value) == key and self._is_final(item.value)
                for item in self._items
            ):
                return True
            if self._closed:
                return False
            self._changed.clear()
            if any(
                self._key(item.value) == key and self._is_final(item.value)
                for item in self._items
            ) or self._closed:
                continue
            await self._changed.wait()


_ResultT = TypeVar("_ResultT")


async def _run_until_final(
    request: Awaitable[_ResultT],
    *,
    mailbox: _LatestPartialMailbox[_WorkT],
    key: int,
) -> tuple[bool, _ResultT | None]:
    """Run a partial request, cancelling it when its authoritative final arrives."""
    request_task = asyncio.create_task(request)
    final_task = asyncio.create_task(mailbox.wait_for_final(key))
    done, _ = await asyncio.wait(
        (request_task, final_task),
        return_when=asyncio.FIRST_COMPLETED,
    )
    if request_task in done:
        final_task.cancel()
        await asyncio.gather(final_task, return_exceptions=True)
        return False, await request_task
    if await final_task:
        request_task.cancel()
        await asyncio.gather(request_task, return_exceptions=True)
        return True, None
    return False, await request_task


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
        speaker_change_detector: SpeakerChangeDetector | None = None,
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
                target_language=config.session.target_language,
                emit=self._dispatch,
                speaker_change_detector=speaker_change_detector,
            )
            if tts and config.session.audio_output and config.voice_clone.enabled
            else None
        )

    async def run(self, source: AudioWindowSource) -> None:
        window_queue = _LatestPartialMailbox[AudioWindow](
            key=lambda window: window.turn_id,
            is_final=lambda window: window.is_final,
        )
        mt_queue = _LatestPartialMailbox[_TranslationWork](
            key=lambda work: work.turn_id,
            is_final=lambda work: work.is_final,
        )
        # Text/audio output is an independent chain.  Do not back-pressure ASR/MT
        # merely because speech is currently longer than synthesized playback.
        tts_queue: asyncio.Queue[_TTSWork | None] = asyncio.Queue()
        playback_queue: asyncio.Queue[_PlaybackWork | None] = asyncio.Queue()

        await self._emit("session.started", data={
            "source_language": self.config.session.source_language,
            "target_language": self.config.session.target_language,
            "audio_output": self.config.session.audio_output,
            "voice_clone": bool(self._voice),
            "tts_speech_rate": self.config.tts.speech_rate,
        })

        warmup = getattr(self.tts, "warmup", None) if self.tts else None
        warmup_task = (
            asyncio.create_task(warmup()) if callable(warmup) else None
        )

        producer = asyncio.create_task(self._produce_windows(source, window_queue))
        asr_worker = asyncio.create_task(self._asr_worker(window_queue, mt_queue))
        mt_worker = asyncio.create_task(self._mt_worker(mt_queue, tts_queue))
        tts_worker = asyncio.create_task(
            self._tts_worker(tts_queue, playback_queue)
        )
        playback_worker = asyncio.create_task(
            self._playback_worker(playback_queue)
        )
        try:
            await asyncio.gather(
                producer,
                asr_worker,
                mt_worker,
                tts_worker,
                playback_worker,
            )
            await self._emit("session.finished")
        except Exception as exc:
            for task in (
                producer,
                asr_worker,
                mt_worker,
                tts_worker,
                playback_worker,
            ):
                if not task.done():
                    task.cancel()
            await asyncio.gather(
                producer,
                asr_worker,
                mt_worker,
                tts_worker,
                playback_worker,
                return_exceptions=True,
            )
            await self._emit("pipeline.error", data={"error": str(exc)})
            raise
        finally:
            if warmup_task:
                if not warmup_task.done():
                    warmup_task.cancel()
                await asyncio.gather(warmup_task, return_exceptions=True)
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
        queue: _LatestPartialMailbox[AudioWindow],
    ) -> None:
        try:
            async for window in source:
                queue.put(window)
                if window.is_final:
                    self._speech_idle.set()
        finally:
            self._speech_idle.set()
            queue.close()

    async def _asr_worker(
        self,
        queue: _LatestPartialMailbox[AudioWindow],
        mt_queue: _LatestPartialMailbox[_TranslationWork],
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
                mt_queue.put(_TranslationWork(turn_id, source_text, True))
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
                queued_window = await queue.get()
                if queued_window is None:
                    break
                window = queued_window.value
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
                if queued_window.coalesced:
                    timing["asr_updates_coalesced"] = (
                        timing.get("asr_updates_coalesced", 0.0)
                        + queued_window.coalesced
                    )
                request_started = time.monotonic()
                if window.is_final:
                    timing["audio_ended"] = window.captured_at
                    timing["final_asr_queue_ms"] = max(
                        0.0,
                        (request_started - queued_window.enqueued_at) * 1_000,
                    )
                asr_request = self.asr.transcribe(
                    window,
                    language=self.config.session.source_language,
                )
                if window.is_final:
                    hypothesis = await asr_request
                else:
                    cancelled, partial_hypothesis = await _run_until_final(
                        asr_request,
                        mailbox=queue,
                        key=window.turn_id,
                    )
                    if cancelled:
                        timing["asr_partials_cancelled"] = (
                            timing.get("asr_partials_cancelled", 0.0) + 1
                        )
                        continue
                    assert partial_hypothesis is not None
                    hypothesis = partial_hypothesis
                if window.is_final:
                    timing["final_asr_request_ms"] = (
                        time.monotonic() - request_started
                    ) * 1_000
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
                    endpoint_started = time.monotonic()
                    decision = await self.endpoint.classify(
                        endpoint_window,
                        transcript=hypothesis,
                        language=self.config.session.source_language,
                    )
                    timing["endpoint_request_ms"] = (
                        time.monotonic() - endpoint_started
                    ) * 1_000
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
                    mt_queue.put(
                        _TranslationWork(logical_turn_id, aggregate, logical_final)
                    )
                    last_sent = aggregate

                if not window.is_final:
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
                mt_queue.put(
                    _TranslationWork(logical_turn_id, last_aggregate, True)
                )
        finally:
            mt_queue.close()

    async def _mt_worker(
        self,
        queue: _LatestPartialMailbox[_TranslationWork],
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
                queued_work = await queue.get()
                if queued_work is None:
                    break
                work = queued_work.value
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

                timing = self._timing.setdefault(
                    work.turn_id, {"audio_started": time.monotonic()}
                )
                if queued_work.coalesced:
                    timing["mt_updates_coalesced"] = (
                        timing.get("mt_updates_coalesced", 0.0)
                        + queued_work.coalesced
                    )
                translation_started = time.monotonic()
                queue_ms = max(
                    0.0,
                    (translation_started - queued_work.enqueued_at) * 1_000,
                )
                timing.setdefault("first_mt_queue_ms", queue_ms)
                if work.is_final:
                    timing["final_mt_queue_ms"] = queue_ms
                translation_request = self.translator.translate(
                    work.source_text,
                    source_language=self.config.session.source_language,
                    target_language=self.config.session.target_language,
                    committed_target=committer.committed,
                    context=list(context),
                    domain=self.config.session.domain,
                    on_delta=on_delta,
                )
                if work.is_final:
                    hypothesis = await translation_request
                else:
                    cancelled, partial_hypothesis = await _run_until_final(
                        translation_request,
                        mailbox=queue,
                        key=work.turn_id,
                    )
                    if cancelled:
                        timing["mt_partials_cancelled"] = (
                            timing.get("mt_partials_cancelled", 0.0) + 1
                        )
                        continue
                    assert partial_hypothesis is not None
                    hypothesis = partial_hypothesis
                request_ms = (time.monotonic() - translation_started) * 1_000
                timing.setdefault("first_mt_request_ms", request_ms)
                if work.is_final:
                    timing["final_mt_request_ms"] = request_ms
                update = committer.update(hypothesis, is_final=work.is_final)
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

    async def _tts_worker(
        self,
        queue: asyncio.Queue[_TTSWork | None],
        playback_queue: asyncio.Queue[_PlaybackWork | None],
    ) -> None:
        try:
            while True:
                work = await queue.get()
                if work is None:
                    return
                if not work.text:
                    if work.is_final:
                        await self._emit_turn_metrics(work.turn_id)
                    continue
                timing = self._timing.setdefault(
                    work.turn_id, {"audio_started": time.monotonic()}
                )
                timing.setdefault(
                    "tts_queue_ms",
                    max(0.0, (time.monotonic() - work.enqueued_at) * 1_000),
                )
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
                    if (
                        profile_id is None
                        and self.config.voice_clone.fallback_policy == "skip"
                    ):
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

                first_chunk = True
                cancelled_during_stream = False
                prefetched: list[AudioChunk] = []
                prefetched_ms = 0.0

                async def enqueue_chunk(chunk: AudioChunk) -> None:
                    nonlocal first_chunk
                    await playback_queue.put(
                        _PlaybackWork(
                            turn_id=work.turn_id,
                            segment_id=work.segment_id,
                            text=work.text,
                            chunk=chunk,
                            playback_epoch=work.playback_epoch,
                            is_first=first_chunk,
                        )
                    )
                    first_chunk = False

                try:
                    timing = self._timing.setdefault(
                        work.turn_id, {"audio_started": time.monotonic()}
                    )
                    timing.setdefault("tts_requested", time.monotonic())
                    async for chunk in self.tts.synthesize(
                        work.text,
                        language=self.config.session.target_language,
                        profile_id=profile_id,
                    ):
                        timing.setdefault("first_cloud_audio", time.monotonic())
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
                        if (
                            first_chunk
                            and self.config.streaming.tts_prebuffer_ms > 0
                        ):
                            prefetched.append(chunk)
                            frame_bytes = chunk.channels * chunk.sample_width
                            prefetched_ms += (
                                len(chunk.data)
                                / frame_bytes
                                / chunk.sample_rate
                                * 1_000
                            )
                            if (
                                prefetched_ms
                                < self.config.streaming.tts_prebuffer_ms
                            ):
                                continue
                            for buffered in prefetched:
                                await enqueue_chunk(buffered)
                            prefetched.clear()
                        else:
                            await enqueue_chunk(chunk)
                    if not cancelled_during_stream:
                        for buffered in prefetched:
                            await enqueue_chunk(buffered)
                        await playback_queue.put(
                            _PlaybackWork(
                                turn_id=work.turn_id,
                                segment_id=work.segment_id,
                                text=work.text,
                                chunk=None,
                                playback_epoch=work.playback_epoch,
                                is_last=True,
                                is_final=work.is_final,
                            )
                        )
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
                if cancelled_during_stream and work.is_final:
                    await self._emit_turn_metrics(work.turn_id)
        finally:
            await playback_queue.put(None)

    async def _playback_worker(
        self,
        queue: asyncio.Queue[_PlaybackWork | None],
    ) -> None:
        cancelled_segments: set[str] = set()
        last_audio_ended_at: float | None = None
        while True:
            work = await queue.get()
            if work is None:
                self._playing = False
                return
            if work.playback_epoch != self._playback_epoch:
                if work.segment_id not in cancelled_segments:
                    cancelled_segments.add(work.segment_id)
                    await self._emit(
                        "tts.cancelled",
                        turn_id=work.turn_id,
                        data={
                            "segment_id": work.segment_id,
                            "reason": "barge-in before playback",
                        },
                    )
                if work.is_last and work.is_final:
                    await self._emit_turn_metrics(work.turn_id)
                continue
            if work.is_last:
                self._playing = False
                await self._emit(
                    "tts.finished",
                    turn_id=work.turn_id,
                    data={"segment_id": work.segment_id},
                )
                if work.is_final:
                    await self._emit_turn_metrics(work.turn_id)
                continue
            if work.chunk is None:
                continue
            if work.is_first:
                self._playing = True
                timing = self._timing.setdefault(
                    work.turn_id, {"audio_started": time.monotonic()}
                )
                if last_audio_ended_at is not None:
                    timing.setdefault(
                        "playback_gap_ms",
                        round((time.monotonic() - last_audio_ended_at) * 1_000, 1),
                    )
                timing.setdefault("first_audio", time.monotonic())
                await self._emit(
                    "tts.started",
                    turn_id=work.turn_id,
                    data={
                        "segment_id": work.segment_id,
                        "text": work.text,
                    },
                )
            await self.audio_sink.write(work.chunk, segment_id=work.segment_id)
            last_audio_ended_at = time.monotonic()

    async def _emit_turn_metrics(self, turn_id: int) -> None:
        timing = self._timing.get(turn_id, {})
        started = timing.get("audio_started")
        values: dict[str, float | None] = {}
        for key in (
            "first_source_commit",
            "first_target_commit",
            "tts_requested",
            "first_cloud_audio",
            "first_audio",
        ):
            value = timing.get(key)
            values[f"{key}_ms"] = (
                round((value - started) * 1_000, 1)
                if value is not None and started is not None
                else None
            )
        cloud_audio = timing.get("first_cloud_audio")
        first_audio = timing.get("first_audio")
        values["playback_queue_ms"] = (
            round((first_audio - cloud_audio) * 1_000, 1)
            if first_audio is not None and cloud_audio is not None
            else None
        )
        values["playback_gap_ms"] = timing.get("playback_gap_ms")
        audio_ended = timing.get("audio_ended")
        values["utterance_ms"] = (
            round((audio_ended - started) * 1_000, 1)
            if audio_ended is not None and started is not None
            else None
        )
        for key in (
            "final_asr_queue_ms",
            "final_asr_request_ms",
            "endpoint_request_ms",
            "first_mt_queue_ms",
            "first_mt_request_ms",
            "final_mt_queue_ms",
            "final_mt_request_ms",
            "tts_queue_ms",
            "asr_updates_coalesced",
            "mt_updates_coalesced",
            "asr_partials_cancelled",
            "mt_partials_cancelled",
        ):
            value = timing.get(key)
            values[key] = round(value, 1) if value is not None else None
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
