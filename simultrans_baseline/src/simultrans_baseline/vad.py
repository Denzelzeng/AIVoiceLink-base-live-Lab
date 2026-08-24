from __future__ import annotations

from array import array
from pathlib import Path
from typing import Protocol

from .config import ConfigurationError, VADConfig


TARGET_SAMPLE_RATE = 16_000


class SpeechDetector(Protocol):
    """Stateful frame-level speech detector used by the turn segmenter."""

    def is_speech(self, pcm: bytes, *, sample_rate: int) -> bool: ...


def pcm16_as_float(
    pcm: bytes,
    *,
    sample_rate: int,
    target_sample_rate: int = TARGET_SAMPLE_RATE,
) -> list[float]:
    """Convert mono PCM16 to normalized floats and linearly resample.

    The local VAD and speaker models both consume 16 kHz mono audio.  Keeping
    this conversion dependency-free avoids pulling PyTorch into the API
    baseline merely for audio I/O.
    """

    if len(pcm) % 2:
        raise ValueError("16-bit PCM must contain an even number of bytes")
    values = array("h")
    values.frombytes(pcm)
    if not values:
        return []
    normalized = [value / 32768.0 for value in values]
    if sample_rate == target_sample_rate:
        return normalized
    if sample_rate <= 0 or target_sample_rate <= 0:
        raise ValueError("sample rates must be positive")

    output_length = max(1, round(len(normalized) * target_sample_rate / sample_rate))
    ratio = sample_rate / target_sample_rate
    last = len(normalized) - 1
    output: list[float] = []
    for output_index in range(output_length):
        position = min(output_index * ratio, last)
        left = int(position)
        right = min(left + 1, last)
        fraction = position - left
        output.append(
            normalized[left] * (1.0 - fraction) + normalized[right] * fraction
        )
    return output


class SherpaOnnxSpeechDetector:
    """Silero/TEN VAD adapter using sherpa-onnx's native streaming runtime."""

    def __init__(self, config: VADConfig, model_path: Path):
        try:
            import sherpa_onnx
        except ImportError as exc:
            raise ConfigurationError(
                "neural VAD needs sherpa-onnx; run scripts/setup-local-audio.ps1"
            ) from exc

        if not model_path.is_file():
            raise ConfigurationError(
                f"VAD model not found: {model_path}; "
                "run scripts/setup-local-audio.ps1"
            )

        native = sherpa_onnx.VadModelConfig()
        model = (
            native.silero_vad
            if config.provider == "sherpa_silero"
            else native.ten_vad
        )
        model.model = str(model_path)
        model.threshold = config.threshold
        model.min_speech_duration = config.min_speech_ms / 1_000
        model.min_silence_duration = config.min_silence_ms / 1_000
        native.sample_rate = TARGET_SAMPLE_RATE
        native.num_threads = config.num_threads
        native.provider = config.inference_provider
        if not native.validate():
            raise ConfigurationError(f"invalid local VAD configuration: {native}")

        self._vad = sherpa_onnx.VoiceActivityDetector(
            native,
            buffer_size_in_seconds=60,
        )
        self._window_size = model.window_size
        self._buffer: list[float] = []

    def is_speech(self, pcm: bytes, *, sample_rate: int) -> bool:
        self._buffer.extend(pcm16_as_float(pcm, sample_rate=sample_rate))
        while len(self._buffer) >= self._window_size:
            window = self._buffer[: self._window_size]
            del self._buffer[: self._window_size]
            self._vad.accept_waveform(window)
            # Completed segments are not consumed here: the turn segmenter
            # owns the original-rate PCM and its final boundaries.
            while not self._vad.empty():
                self._vad.pop()
        return bool(self._vad.is_speech_detected())


def resolve_model_path(value: str, *, config_path: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = config_path.parent / path
    return path.resolve()


def build_speech_detector(
    config: VADConfig,
    *,
    config_path: Path,
) -> SpeechDetector | None:
    if config.provider == "energy":
        return None
    return SherpaOnnxSpeechDetector(
        config,
        resolve_model_path(config.model_path, config_path=config_path),
    )
