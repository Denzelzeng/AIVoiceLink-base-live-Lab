from __future__ import annotations

import asyncio
from collections import OrderedDict
from dataclasses import dataclass, field

from .config import VoiceCloneConfig
from .contracts import CloningTTSBackend, EventHandler
from .events import AudioWindow, PipelineEvent, VoiceProfile


@dataclass
class _TurnReference:
    pcm: bytearray = field(default_factory=bytearray)
    seen_by_segment: dict[int, int] = field(default_factory=dict)
    transcripts: OrderedDict[int, str] = field(default_factory=OrderedDict)
    sample_rate: int = 0
    final: bool = False


class VoiceEnrollmentManager:
    """Continuously refresh cloned voices from new, sufficiently long turns.

    The first profile may bootstrap from several short turns so startup TTS is
    not lost. Once ready, every logical turn containing enough clean speech
    receives its own asynchronously enrolled profile. TTS can then wait for
    the profile belonging to that turn instead of reusing one voice forever.
    """

    def __init__(
        self,
        backend: CloningTTSBackend,
        config: VoiceCloneConfig,
        *,
        session_id: str,
        source_language: str,
        emit: EventHandler,
    ):
        self.backend = backend
        self.config = config
        self.session_id = session_id
        self.source_language = source_language
        self.emit = emit
        self.profile: VoiceProfile | None = None
        self.error: Exception | None = None

        self._turns: dict[int, _TurnReference] = {}
        self._turn_tasks: dict[int, asyncio.Task[None]] = {}
        self._turn_profiles: dict[int, VoiceProfile] = {}
        self._turn_decided: dict[int, asyncio.Event] = {}
        self._all_profiles: dict[str, VoiceProfile] = {}
        self._enrolled_turns: set[int] = set()

        self._bootstrap_pcm = bytearray()
        self._bootstrap_transcripts: OrderedDict[int, str] = OrderedDict()
        self._bootstrap_started = False
        self._first_ready = asyncio.Event()
        self._sequence = 0
        self._active_sequence = 0

    async def observe(
        self,
        window: AudioWindow,
        transcript: str,
        *,
        voice_turn_id: int,
    ) -> None:
        if not self.config.enabled:
            return
        turn = self._turns.setdefault(voice_turn_id, _TurnReference())
        turn.sample_rate = window.sample_rate
        turn.final = turn.final or window.is_final
        decision = self._turn_decided.setdefault(voice_turn_id, asyncio.Event())

        already_seen = turn.seen_by_segment.get(window.turn_id, 0)
        delta = b""
        if len(window.pcm) > already_seen:
            delta = window.pcm[already_seen:]
            turn.pcm.extend(delta)
            turn.seen_by_segment[window.turn_id] = len(window.pcm)
        if transcript.strip():
            turn.transcripts[window.turn_id] = transcript.strip()

        # Before any voice exists, several short utterances from the initial
        # speaker may jointly satisfy the provider's hard reference minimum.
        if not self.profile and not self._bootstrap_started:
            if delta:
                self._bootstrap_pcm.extend(delta)
            if transcript.strip():
                self._bootstrap_transcripts[voice_turn_id] = transcript.strip()
            if self._duration_ms(self._bootstrap_pcm, window.sample_rate) >= (
                self.config.min_reference_ms
            ):
                self._bootstrap_started = True
                self._enrolled_turns.add(voice_turn_id)
                await self._start_enrollment(
                    voice_turn_id,
                    self._bootstrap_pcm,
                    window.sample_rate,
                    " ".join(self._bootstrap_transcripts.values()),
                    refresh=False,
                )
                decision.set()
                return

        if (
            (self.profile or self._bootstrap_started)
            and self.config.refresh_enabled
            and voice_turn_id not in self._enrolled_turns
            and self._duration_ms(turn.pcm, window.sample_rate)
            >= self.config.min_reference_ms
        ):
            self._enrolled_turns.add(voice_turn_id)
            await self._start_enrollment(
                voice_turn_id,
                turn.pcm,
                window.sample_rate,
                " ".join(turn.transcripts.values()),
                refresh=True,
            )
            decision.set()
            return

        if window.is_final:
            # A short turn cannot be enrolled by the provider. Mark it decided
            # so TTS may use the latest completed profile without hanging.
            decision.set()

    async def _start_enrollment(
        self,
        voice_turn_id: int,
        pcm: bytearray,
        sample_rate: int,
        transcript: str,
        *,
        refresh: bool,
    ) -> None:
        max_bytes = round(
            self.config.max_reference_ms / 1_000 * sample_rate * 2
        )
        # Prefer recent speech so a changed speaker replaces the previous voice
        # as soon as one full provider-sized reference is available.
        reference = bytes(pcm[-max_bytes:])
        reference_ms = self._duration_ms(reference, sample_rate)
        self._sequence += 1
        sequence = self._sequence
        await self._emit(
            PipelineEvent(
                kind="voice.enrollment_started",
                session_id=self.session_id,
                turn_id=voice_turn_id,
                data={
                    "reference_ms": reference_ms,
                    "refresh": refresh,
                    "sequence": sequence,
                },
            )
        )
        task = asyncio.create_task(
            self._enroll(
                reference,
                sample_rate=sample_rate,
                transcript=transcript,
                reference_ms=reference_ms,
                voice_turn_id=voice_turn_id,
                refresh=refresh,
                sequence=sequence,
            )
        )
        self._turn_tasks[voice_turn_id] = task

    async def _enroll(
        self,
        reference: bytes,
        *,
        sample_rate: int,
        transcript: str,
        reference_ms: int,
        voice_turn_id: int,
        refresh: bool,
        sequence: int,
    ) -> None:
        try:
            profile = await self.backend.enroll(
                reference,
                sample_rate=sample_rate,
                transcript=transcript,
                language=self.source_language,
            )
            enrolled = VoiceProfile(
                profile_id=profile.profile_id,
                reference_ms=reference_ms,
                created_at=profile.created_at,
            )
            self._turn_profiles[voice_turn_id] = enrolled
            self._all_profiles[enrolled.profile_id] = enrolled
            if sequence >= self._active_sequence:
                self._active_sequence = sequence
                self.profile = enrolled
            self.error = None
            await self._emit(
                PipelineEvent(
                    kind="voice.ready",
                    session_id=self.session_id,
                    turn_id=voice_turn_id,
                    data={
                        "profile_id": enrolled.profile_id,
                        "reference_ms": reference_ms,
                        "refresh": refresh,
                        "sequence": sequence,
                    },
                )
            )
        except Exception as exc:
            self.error = exc
            await self._emit(
                PipelineEvent(
                    kind="voice.failed",
                    session_id=self.session_id,
                    turn_id=voice_turn_id,
                    data={"error": str(exc), "refresh": refresh},
                )
            )
        finally:
            self._first_ready.set()
            self._turn_decided.setdefault(voice_turn_id, asyncio.Event()).set()

    async def wait_for_profile(self, voice_turn_id: int) -> VoiceProfile | None:
        timeout = self.config.wait_timeout_ms / 1_000
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout

        decision = self._turn_decided.setdefault(voice_turn_id, asyncio.Event())
        if not decision.is_set():
            try:
                await asyncio.wait_for(decision.wait(), timeout=timeout)
            except asyncio.TimeoutError:
                return self._turn_profiles.get(voice_turn_id) or self.profile

        task = self._turn_tasks.get(voice_turn_id)
        if task and not task.done():
            remaining = max(0.0, deadline - loop.time())
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=remaining)
            except asyncio.TimeoutError:
                return self._turn_profiles.get(voice_turn_id) or self.profile

        exact = self._turn_profiles.get(voice_turn_id)
        if exact:
            return exact
        if self.profile:
            return self.profile

        remaining = max(0.0, deadline - loop.time())
        try:
            await asyncio.wait_for(self._first_ready.wait(), timeout=remaining)
        except asyncio.TimeoutError:
            return None
        return self.profile

    async def aclose(self) -> None:
        pending = [task for task in self._turn_tasks.values() if not task.done()]
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        if self.config.delete_on_close:
            for profile_id in tuple(self._all_profiles):
                try:
                    await self.backend.delete_profile(profile_id)
                    await self._emit(
                        PipelineEvent(
                            kind="voice.deleted",
                            session_id=self.session_id,
                            data={"profile_id": profile_id},
                        )
                    )
                except Exception as exc:
                    await self._emit(
                        PipelineEvent(
                            kind="voice.delete_failed",
                            session_id=self.session_id,
                            data={"profile_id": profile_id, "error": str(exc)},
                        )
                    )

    @staticmethod
    def _duration_ms(pcm: bytes | bytearray, sample_rate: int) -> int:
        return round(len(pcm) / 2 / sample_rate * 1_000)

    async def _emit(self, event: PipelineEvent) -> None:
        result = self.emit(event)
        if asyncio.iscoroutine(result):
            await result
