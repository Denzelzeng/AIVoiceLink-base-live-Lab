from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class AudioWindow:
    """A cumulative PCM snapshot for one active speech turn."""

    turn_id: int
    pcm: bytes
    sample_rate: int
    is_final: bool
    started_at: float
    captured_at: float

    @property
    def duration_ms(self) -> int:
        return round(len(self.pcm) / 2 / self.sample_rate * 1_000)


@dataclass(frozen=True)
class CommitUpdate:
    committed: str
    unstable: str
    commit_delta: str
    is_final: bool
    hypothesis: str


@dataclass(frozen=True)
class AudioChunk:
    data: bytes
    sample_rate: int
    channels: int = 1
    sample_width: int = 2


@dataclass(frozen=True)
class VoiceProfile:
    profile_id: str
    reference_ms: int
    created_at: float = field(default_factory=time.monotonic)


@dataclass(frozen=True)
class EndpointDecision:
    complete: bool
    probability: float | None = None
    reason: str = ""


@dataclass(frozen=True)
class PipelineEvent:
    kind: str
    session_id: str
    turn_id: int | None = None
    timestamp: float = field(default_factory=time.time)
    data: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)
