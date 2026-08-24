from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from typing import Protocol

from .events import (
    AudioChunk,
    AudioWindow,
    EndpointDecision,
    PipelineEvent,
    VoiceProfile,
)


EventHandler = Callable[[PipelineEvent], Awaitable[None] | None]
TextDeltaHandler = Callable[[str], Awaitable[None] | None]


class AudioWindowSource(Protocol):
    def __aiter__(self) -> AsyncIterator[AudioWindow]: ...


class ASRBackend(Protocol):
    async def transcribe(
        self,
        window: AudioWindow,
        *,
        language: str,
    ) -> str: ...

    async def health(self) -> dict[str, object]: ...

    async def aclose(self) -> None: ...


class TranslationBackend(Protocol):
    async def translate(
        self,
        source_text: str,
        *,
        source_language: str,
        target_language: str,
        committed_target: str,
        context: Sequence[tuple[str, str]],
        domain: str,
        on_delta: TextDeltaHandler | None = None,
    ) -> str: ...

    async def health(self) -> dict[str, object]: ...

    async def aclose(self) -> None: ...


class SemanticEndpointBackend(Protocol):
    async def classify(
        self,
        window: AudioWindow,
        *,
        transcript: str,
        language: str,
    ) -> EndpointDecision: ...

    async def health(self) -> dict[str, object]: ...

    async def aclose(self) -> None: ...


class CloningTTSBackend(Protocol):
    async def enroll(
        self,
        reference_pcm: bytes,
        *,
        sample_rate: int,
        transcript: str,
        language: str,
    ) -> VoiceProfile: ...

    def synthesize(
        self,
        text: str,
        *,
        language: str,
        profile_id: str | None,
    ) -> AsyncIterator[AudioChunk]: ...

    async def delete_profile(self, profile_id: str) -> None: ...

    async def health(self) -> dict[str, object]: ...

    async def aclose(self) -> None: ...


class AudioSink(Protocol):
    async def write(self, chunk: AudioChunk, *, segment_id: str) -> None: ...

    async def interrupt(self) -> None: ...

    async def aclose(self) -> None: ...
