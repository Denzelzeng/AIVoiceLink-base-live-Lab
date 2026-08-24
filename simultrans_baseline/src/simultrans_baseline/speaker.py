from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

from .config import ConfigurationError, SpeakerChangeConfig
from .vad import pcm16_as_float, resolve_model_path


class SpeakerEmbeddingBackend(Protocol):
    def embed(self, pcm: bytes, *, sample_rate: int) -> Sequence[float]: ...


@dataclass(frozen=True)
class SpeakerDecision:
    state: str
    changed: bool
    similarity: float | None
    confirmations: int = 0


def _normalize(values: Sequence[float]) -> list[float]:
    vector = [float(value) for value in values]
    norm = math.sqrt(sum(value * value for value in vector))
    if not vector or norm <= 1e-12:
        raise RuntimeError("speaker model returned an empty embedding")
    return [value / norm for value in vector]


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("speaker embeddings must have the same non-zero dimension")
    return sum(a * b for a, b in zip(_normalize(left), _normalize(right), strict=True))


def _centroid(vectors: Sequence[Sequence[float]]) -> list[float]:
    if not vectors:
        raise ValueError("at least one embedding is required")
    width = len(vectors[0])
    mean = [0.0] * width
    for vector in vectors:
        if len(vector) != width:
            raise ValueError("speaker embeddings have inconsistent dimensions")
        for index, value in enumerate(vector):
            mean[index] += float(value)
    return _normalize([value / len(vectors) for value in mean])


class SherpaOnnxSpeakerEmbedder:
    """Local 3D-Speaker/WeSpeaker embedding extractor."""

    def __init__(self, config: SpeakerChangeConfig, model_path: Path):
        try:
            import sherpa_onnx
        except ImportError as exc:
            raise ConfigurationError(
                "speaker change detection needs sherpa-onnx; "
                "run scripts/setup-local-audio.ps1"
            ) from exc
        if not model_path.is_file():
            raise ConfigurationError(
                f"speaker embedding model not found: {model_path}; "
                "run scripts/setup-local-audio.ps1"
            )

        native = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=str(model_path),
            num_threads=config.num_threads,
            provider=config.inference_provider,
        )
        if not native.validate():
            raise ConfigurationError(f"invalid speaker embedding config: {native}")
        self._extractor = sherpa_onnx.SpeakerEmbeddingExtractor(native)

    def embed(self, pcm: bytes, *, sample_rate: int) -> Sequence[float]:
        samples = pcm16_as_float(pcm, sample_rate=sample_rate)
        stream = self._extractor.create_stream()
        stream.accept_waveform(sample_rate=16_000, waveform=samples)
        stream.input_finished()
        if not self._extractor.is_ready(stream):
            raise RuntimeError("speaker audio is too short for embedding extraction")
        return self._extractor.compute(stream)


class SpeakerChangeDetector:
    """Hysteretic speaker-change detector with a slowly adapting reference.

    Scores above ``same_threshold`` update the current speaker centroid. Scores
    below ``change_threshold`` form a candidate speaker.  The gap between the
    thresholds is intentionally undecided so channel/noise drift cannot cause
    cloud voice re-enrollment.
    """

    def __init__(
        self,
        backend: SpeakerEmbeddingBackend,
        config: SpeakerChangeConfig,
    ):
        self.backend = backend
        self.config = config
        self._reference: list[float] | None = None
        self._candidate: list[list[float]] = []
        self._lock = asyncio.Lock()

    async def seed(self, pcm: bytes, *, sample_rate: int) -> None:
        embedding = await asyncio.to_thread(
            self.backend.embed,
            pcm,
            sample_rate=sample_rate,
        )
        async with self._lock:
            self._reference = _normalize(embedding)
            self._candidate.clear()

    async def assess(self, pcm: bytes, *, sample_rate: int) -> SpeakerDecision:
        embedding = _normalize(
            await asyncio.to_thread(
                self.backend.embed,
                pcm,
                sample_rate=sample_rate,
            )
        )
        async with self._lock:
            if self._reference is None:
                self._reference = embedding
                return SpeakerDecision("initial", changed=False, similarity=None)

            similarity = cosine_similarity(self._reference, embedding)
            if similarity >= self.config.same_threshold:
                alpha = self.config.reference_update_alpha
                self._reference = _normalize(
                    [
                        (1.0 - alpha) * old + alpha * new
                        for old, new in zip(
                            self._reference,
                            embedding,
                            strict=True,
                        )
                    ]
                )
                self._candidate.clear()
                return SpeakerDecision("same", changed=False, similarity=similarity)

            if similarity > self.config.change_threshold:
                self._candidate.clear()
                return SpeakerDecision(
                    "ambiguous",
                    changed=False,
                    similarity=similarity,
                )

            if self._candidate and cosine_similarity(
                _centroid(self._candidate), embedding
            ) < self.config.same_threshold:
                self._candidate.clear()
            self._candidate.append(embedding)
            count = len(self._candidate)
            if count < self.config.confirmation_windows:
                return SpeakerDecision(
                    "candidate",
                    changed=False,
                    similarity=similarity,
                    confirmations=count,
                )

            self._reference = _centroid(self._candidate)
            self._candidate.clear()
            return SpeakerDecision(
                "changed",
                changed=True,
                similarity=similarity,
                confirmations=count,
            )


def build_speaker_change_detector(
    config: SpeakerChangeConfig,
    *,
    config_path: Path,
) -> SpeakerChangeDetector | None:
    if not config.enabled:
        return None
    model_path = resolve_model_path(config.model_path, config_path=config_path)
    return SpeakerChangeDetector(
        SherpaOnnxSpeakerEmbedder(config, model_path),
        config,
    )
