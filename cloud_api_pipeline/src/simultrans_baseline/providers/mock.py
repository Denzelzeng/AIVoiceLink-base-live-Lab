from __future__ import annotations

import math
import struct
import uuid
from collections.abc import AsyncIterator, Sequence

from ..contracts import TextDeltaHandler
from ..events import AudioChunk, AudioWindow, VoiceProfile


class ScriptedASR:
    def __init__(self, hypotheses: Sequence[str]):
        self.hypotheses = list(hypotheses)
        self.calls = 0

    async def transcribe(self, window: AudioWindow, *, language: str) -> str:
        if not self.hypotheses:
            return ""
        index = min(self.calls, len(self.hypotheses) - 1)
        self.calls += 1
        return self.hypotheses[index]

    async def health(self) -> dict[str, object]:
        return {"provider": "mock", "ready": True}

    async def aclose(self) -> None:
        return None


class RuleBasedMockTranslator:
    def __init__(self, translations: dict[str, str] | None = None):
        self.translations = translations or {
            "大家好": "Hello everyone",
            "大家好，欢迎": "Hello everyone, welcome",
            "大家好，欢迎参加今天的会议。": "Hello everyone, welcome to today's meeting.",
        }
        self.calls: list[str] = []

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
    ) -> str:
        self.calls.append(source_text)
        value = self.translations.get(source_text, f"[{target_language}] {source_text}")
        if on_delta:
            result = on_delta(value)
            import asyncio

            if asyncio.iscoroutine(result):
                await result
        return value

    async def health(self) -> dict[str, object]:
        return {"provider": "mock", "ready": True}

    async def aclose(self) -> None:
        return None


class MockCloningTTS:
    """Control-flow fake: profile IDs are real, audio is only a test tone."""

    def __init__(self, sample_rate: int = 24_000):
        self.sample_rate = sample_rate
        self.enroll_calls = 0
        self.synthesized: list[tuple[str, str | None]] = []
        self.deleted: list[str] = []

    async def enroll(
        self,
        reference_pcm: bytes,
        *,
        sample_rate: int,
        transcript: str,
        language: str,
    ) -> VoiceProfile:
        self.enroll_calls += 1
        return VoiceProfile(
            profile_id=f"mock-{uuid.uuid4().hex[:8]}",
            reference_ms=round(len(reference_pcm) / 2 / sample_rate * 1_000),
        )

    async def synthesize(
        self,
        text: str,
        *,
        language: str,
        profile_id: str | None,
    ) -> AsyncIterator[AudioChunk]:
        self.synthesized.append((text, profile_id))
        duration = min(0.5, max(0.1, len(text) * 0.02))
        samples = round(duration * self.sample_rate)
        pcm = b"".join(
            struct.pack("<h", round(1800 * math.sin(2 * math.pi * 440 * i / self.sample_rate)))
            for i in range(samples)
        )
        midpoint = len(pcm) // 2
        midpoint -= midpoint % 2
        for data in (pcm[:midpoint], pcm[midpoint:]):
            if data:
                yield AudioChunk(data=data, sample_rate=self.sample_rate)

    async def delete_profile(self, profile_id: str) -> None:
        self.deleted.append(profile_id)

    async def health(self) -> dict[str, object]:
        return {"provider": "mock", "ready": True}

    async def aclose(self) -> None:
        return None

