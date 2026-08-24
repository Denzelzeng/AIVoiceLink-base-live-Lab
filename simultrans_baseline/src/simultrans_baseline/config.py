from __future__ import annotations

import os
import csv
import tomllib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from dotenv import load_dotenv


class ConfigurationError(ValueError):
    """Raised when a configuration cannot form a safe, runnable pipeline."""


@dataclass(frozen=True)
class SessionConfig:
    source_language: str = "Chinese"
    target_language: str = "English"
    audio_output: bool = True
    domain: str = "general meeting"


@dataclass(frozen=True)
class ServiceConfig:
    provider: str
    base_url: str
    model: str
    api_key_env: str = ""
    timeout_seconds: float = 60.0
    max_tokens: int = 256
    temperature: float = 0.0
    send_language: bool = False

    @property
    def normalized_base_url(self) -> str:
        value = self.base_url
        if value.startswith("env:"):
            variable = value[4:].strip()
            value = os.getenv(variable, "")
            if not value:
                raise ConfigurationError(
                    f"environment variable {variable} is required for base_url"
                )
        return value.rstrip("/")

    def api_key(self) -> str | None:
        return os.getenv(self.api_key_env) if self.api_key_env else None


@dataclass(frozen=True)
class TTSConfig(ServiceConfig):
    enrollment_model: str = "qwen-voice-enrollment"
    preferred_name: str = "simultrans"
    clone_path: str = "/api/v1/services/audio/tts/customization"
    speech_path: str = "/api/v1/services/aigc/multimodal-generation/generation"
    sample_rate: int = 24_000
    response_format: str = "pcm"
    fallback_voice: str = ""
    websocket_url: str = "auto"
    speech_rate: float = 1.0

    @property
    def normalized_websocket_url(self) -> str:
        value = self.websocket_url
        if value.strip().casefold() == "auto":
            parsed = urlsplit(self.normalized_base_url)
            return urlunsplit(
                ("wss", parsed.netloc, "/api-ws/v1/realtime", "", "")
            )
        if value.startswith("env:"):
            variable = value[4:].strip()
            value = os.getenv(variable, "")
            if not value:
                raise ConfigurationError(
                    f"environment variable {variable} is required for websocket_url"
                )
        return value.rstrip("/")


@dataclass(frozen=True)
class AudioConfig:
    sample_rate: int = 16_000
    channels: int = 1
    sample_width: int = 2
    frame_ms: int = 50
    energy_threshold: int = 350
    calibration_seconds: float = 1.0
    pre_roll_ms: int = 200
    min_speech_ms: int = 300
    end_silence_ms: int = 550
    partial_interval_ms: int = 1_200
    max_turn_ms: int = 10_000

    @property
    def frames_per_buffer(self) -> int:
        return self.sample_rate * self.frame_ms // 1_000


@dataclass(frozen=True)
class VADConfig:
    """Local acoustic VAD configuration.

    Neural VAD is deliberately separate from semantic endpointing: it only
    decides whether speech is present.  The energy implementation remains a
    dependency-free fallback and is also useful in deterministic tests.
    """

    provider: str = "energy"
    model_path: str = ""
    inference_provider: str = "cpu"
    threshold: float = 0.5
    min_speech_ms: int = 150
    min_silence_ms: int = 100
    num_threads: int = 1


@dataclass(frozen=True)
class SpeakerChangeConfig:
    """Independent local speaker-change gate for cloud voice enrollment."""

    enabled: bool = False
    provider: str = "sherpa_onnx"
    model_path: str = ""
    inference_provider: str = "cpu"
    num_threads: int = 2
    min_compare_ms: int = 2_000
    same_threshold: float = 0.72
    change_threshold: float = 0.55
    confirmation_windows: int = 1
    reference_update_alpha: float = 0.10


@dataclass(frozen=True)
class StreamingConfig:
    asr_agreement_depth: int = 2
    mt_agreement_depth: int = 2
    min_source_growth_chars: int = 2
    tts_min_phrase_chars: int = 8
    tts_prebuffer_ms: int = 400
    queue_capacity: int = 6
    recent_context_turns: int = 3
    semantic_hard_timeout_ms: int = 1_800
    barge_in_enabled: bool = True


@dataclass(frozen=True)
class EndpointConfig:
    provider: str = "heuristic"
    base_url: str = ""
    model: str = "qwen-flash"
    api_key_env: str = ""
    timeout_seconds: float = 3.0
    max_tokens: int = 8
    temperature: float = 0.0
    complete_threshold: float = 0.5
    max_audio_ms: int = 8_000

    @property
    def normalized_base_url(self) -> str:
        value = self.base_url
        if value.startswith("env:"):
            variable = value[4:].strip()
            value = os.getenv(variable, "")
            if not value:
                raise ConfigurationError(
                    f"environment variable {variable} is required for base_url"
                )
        return value.rstrip("/")

    def api_key(self) -> str | None:
        return os.getenv(self.api_key_env) if self.api_key_env else None


@dataclass(frozen=True)
class VoiceCloneConfig:
    enabled: bool = True
    consent_confirmed: bool = False
    min_reference_ms: int = 2_500
    max_reference_ms: int = 4_000
    wait_timeout_ms: int = 3_000
    fallback_policy: str = "skip"
    refresh_enabled: bool = True
    delete_on_close: bool = True


@dataclass(frozen=True)
class AppConfig:
    session: SessionConfig
    asr: ServiceConfig
    mt: ServiceConfig
    tts: TTSConfig
    audio: AudioConfig
    vad: VADConfig
    speaker_change: SpeakerChangeConfig
    streaming: StreamingConfig
    endpoint: EndpointConfig
    voice_clone: VoiceCloneConfig
    source_path: Path

    def validate(self) -> None:
        if not self.session.source_language.strip():
            raise ConfigurationError("session.source_language cannot be empty")
        if not self.session.target_language.strip():
            raise ConfigurationError("session.target_language cannot be empty")
        if (
            self.session.source_language.casefold()
            == self.session.target_language.casefold()
        ):
            raise ConfigurationError("source and target languages must differ")
        if self.audio.sample_rate < 16_000:
            raise ConfigurationError("audio input sample rate must be at least 16 kHz")
        if self.audio.channels != 1 or self.audio.sample_width != 2:
            raise ConfigurationError("audio input must be mono signed 16-bit PCM")
        if self.audio.frame_ms <= 0 or 1_000 % self.audio.frame_ms:
            raise ConfigurationError("audio.frame_ms must be a positive divisor of 1000")
        if self.audio.partial_interval_ms < self.audio.frame_ms:
            raise ConfigurationError("partial_interval_ms must be at least one frame")
        if self.audio.min_speech_ms > self.audio.max_turn_ms:
            raise ConfigurationError("min_speech_ms cannot exceed max_turn_ms")
        if self.vad.provider not in {"energy", "sherpa_silero", "sherpa_ten"}:
            raise ConfigurationError(
                "vad.provider must be energy, sherpa_silero, or sherpa_ten"
            )
        if self.vad.provider != "energy" and not self.vad.model_path.strip():
            raise ConfigurationError("vad.model_path is required for neural VAD")
        if not 0 < self.vad.threshold < 1:
            raise ConfigurationError("vad.threshold must be between 0 and 1")
        if self.vad.min_speech_ms < 0 or self.vad.min_silence_ms < 0:
            raise ConfigurationError("VAD speech/silence durations cannot be negative")
        if self.vad.num_threads < 1:
            raise ConfigurationError("vad.num_threads must be positive")
        if self.speaker_change.provider not in {"sherpa_onnx"}:
            raise ConfigurationError(
                "speaker_change.provider must be sherpa_onnx"
            )
        if self.speaker_change.enabled:
            if not self.speaker_change.model_path.strip():
                raise ConfigurationError(
                    "speaker_change.model_path is required when enabled"
                )
            if self.speaker_change.min_compare_ms < 1_000:
                raise ConfigurationError(
                    "speaker_change.min_compare_ms must be at least 1000 ms"
                )
            if not (
                0 <= self.speaker_change.change_threshold
                < self.speaker_change.same_threshold <= 1
            ):
                raise ConfigurationError(
                    "speaker thresholds must satisfy 0 <= change < same <= 1"
                )
            if self.speaker_change.confirmation_windows < 1:
                raise ConfigurationError(
                    "speaker_change.confirmation_windows must be positive"
                )
            if not 0 <= self.speaker_change.reference_update_alpha <= 1:
                raise ConfigurationError(
                    "speaker_change.reference_update_alpha must be between 0 and 1"
                )
        if self.streaming.asr_agreement_depth < 1:
            raise ConfigurationError("asr_agreement_depth must be positive")
        if self.streaming.mt_agreement_depth < 1:
            raise ConfigurationError("mt_agreement_depth must be positive")
        if self.streaming.queue_capacity < 1:
            raise ConfigurationError("queue_capacity must be positive")
        if self.streaming.tts_prebuffer_ms < 0:
            raise ConfigurationError("tts_prebuffer_ms cannot be negative")
        if self.streaming.semantic_hard_timeout_ms < self.audio.end_silence_ms:
            raise ConfigurationError(
                "semantic_hard_timeout_ms must be >= audio.end_silence_ms"
            )
        if self.endpoint.provider not in {"heuristic", "llm_http", "always_final"}:
            raise ConfigurationError(
                "endpoint.provider must be heuristic, llm_http, or always_final"
            )
        if self.endpoint.provider == "llm_http":
            _validate_cloud_url("endpoint", self.endpoint.normalized_base_url)
            if not self.endpoint.model.strip():
                raise ConfigurationError("endpoint.model cannot be empty")
            if not 0 <= self.endpoint.complete_threshold <= 1:
                raise ConfigurationError(
                    "endpoint.complete_threshold must be between 0 and 1"
                )
        if self.endpoint.max_audio_ms < 1_000:
            raise ConfigurationError("endpoint.max_audio_ms must be at least 1000")
        if self.asr.provider not in {"mock", "dashscope_asr"}:
            raise ConfigurationError("asr.provider must be mock or dashscope_asr")
        if self.mt.provider not in {"mock", "qwen_mt"}:
            raise ConfigurationError("mt.provider must be mock or qwen_mt")
        if self.tts.provider not in {
            "mock",
            "disabled",
            "dashscope_qwen_voice_clone",
        }:
            raise ConfigurationError(
                "tts.provider must be mock, disabled, or dashscope_qwen_voice_clone"
            )
        if self.voice_clone.fallback_policy not in {"skip", "default"}:
            raise ConfigurationError(
                "voice_clone.fallback_policy must be 'skip' or 'default'"
            )
        if self.voice_clone.min_reference_ms < 2_000:
            raise ConfigurationError(
                "voice_clone.min_reference_ms must be at least 2000 ms"
            )
        if self.voice_clone.max_reference_ms < self.voice_clone.min_reference_ms:
            raise ConfigurationError(
                "voice_clone.max_reference_ms cannot be below min_reference_ms"
            )
        for label, service in (("asr", self.asr), ("mt", self.mt)):
            if service.provider != "mock":
                _validate_cloud_service(label, service)
        if self.session.audio_output:
            if self.tts.provider == "disabled":
                raise ConfigurationError(
                    "audio output is enabled but tts.provider is disabled"
                )
            if self.tts.provider != "mock":
                _validate_cloud_service("tts", self.tts)
                if not 0.5 <= self.tts.speech_rate <= 2.0:
                    raise ConfigurationError(
                        "tts.speech_rate must be between 0.5 and 2.0"
                    )
                if "-realtime" in self.tts.model:
                    parsed_ws = urlsplit(self.tts.normalized_websocket_url)
                    if parsed_ws.scheme != "wss" or not parsed_ws.hostname:
                        raise ConfigurationError(
                            "tts.websocket_url must use a remote wss:// endpoint"
                        )
            if self.voice_clone.enabled and not self.voice_clone.consent_confirmed:
                raise ConfigurationError(
                    "voice cloning needs explicit speaker consent; pass "
                    "--voice-consent or set VOICE_CLONE_CONSENT_CONFIRMED=true"
                )


def _validate_cloud_url(label: str, base_url: str) -> None:
    parsed = urlsplit(base_url)
    if parsed.scheme != "https" or not parsed.hostname:
        raise ConfigurationError(f"{label}.base_url must use a remote https:// API")
    if parsed.hostname.casefold() in {"localhost", "127.0.0.1", "::1"}:
        raise ConfigurationError(f"{label}.base_url cannot point to a local service")


def _validate_cloud_service(label: str, service: ServiceConfig) -> None:
    _validate_cloud_url(label, service.normalized_base_url)
    if not service.model.strip():
        raise ConfigurationError(f"{label}.model cannot be empty")
    if service.timeout_seconds <= 0:
        raise ConfigurationError(f"{label}.timeout_seconds must be positive")


def _section(data: dict[str, Any], name: str) -> dict[str, Any]:
    value = data.get(name, {})
    if not isinstance(value, dict):
        raise ConfigurationError(f"[{name}] must be a TOML table")
    return value


def _service(data: dict[str, Any], name: str) -> ServiceConfig:
    section = _section(data, name)
    try:
        return ServiceConfig(**section)
    except TypeError as exc:
        raise ConfigurationError(f"invalid [{name}] configuration: {exc}") from exc


def _tts(data: dict[str, Any]) -> TTSConfig:
    try:
        return TTSConfig(**_section(data, "tts"))
    except TypeError as exc:
        raise ConfigurationError(f"invalid [tts] configuration: {exc}") from exc


def load_environment(config_path: Path) -> None:
    """Load project/repository .env files without overriding the process."""
    resolved = config_path.resolve()
    candidates = [resolved.parent / ".env"]
    candidates.extend(parent / ".env" for parent in resolved.parents)
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.is_file():
            load_dotenv(candidate, override=False)
    for parent in (resolved.parent, *resolved.parents):
        for candidate in sorted(parent.glob("*apiKey*.csv")):
            values = _read_workspace_csv(candidate)
            if not values:
                continue
            mappings = {
                "apiKey": "DASHSCOPE_API_KEY",
                "workspaceId": "WORKSPACE_ID",
                "openAiCompatible": "WORKSPACE_OPENAI_BASE_URL",
            }
            for csv_name, env_name in mappings.items():
                value = values.get(csv_name, "").strip()
                if value:
                    os.environ.setdefault(env_name, value)
            compatible_url = values.get("openAiCompatible", "").strip()
            if compatible_url:
                service_url = _workspace_service_base_url(compatible_url)
                if service_url:
                    os.environ.setdefault("WORKSPACE_API_BASE_URL", service_url)
            return


def _workspace_service_base_url(compatible_url: str) -> str:
    """Return the workspace origin used by DashScope native HTTP APIs."""
    try:
        parsed = urlsplit(compatible_url.strip())
    except ValueError:
        return ""
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return ""
    return urlunsplit((parsed.scheme, parsed.netloc, "", "", "")).rstrip("/")


def _read_workspace_csv(path: Path) -> dict[str, str]:
    """Read the transposed Model Studio CSV without logging secret values."""
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or "id" not in reader.fieldnames:
                return {}
            columns = [name for name in reader.fieldnames if name != "id"]
            if not columns:
                return {}
            value_column = columns[0]
            return {
                str(row.get("id", "")).strip(): str(
                    row.get(value_column, "")
                ).strip()
                for row in reader
                if str(row.get("id", "")).strip()
            }
    except (OSError, csv.Error):
        return {}


def load_config(path: Path) -> AppConfig:
    resolved = path.resolve()
    if not resolved.is_file():
        raise ConfigurationError(f"configuration file not found: {resolved}")
    load_environment(resolved)
    with resolved.open("rb") as handle:
        raw = tomllib.load(handle)
    try:
        config = AppConfig(
            session=SessionConfig(**_section(raw, "session")),
            asr=_service(raw, "asr"),
            mt=_service(raw, "mt"),
            tts=_tts(raw),
            audio=AudioConfig(**_section(raw, "audio")),
            vad=VADConfig(**_section(raw, "vad")),
            speaker_change=SpeakerChangeConfig(**_section(raw, "speaker_change")),
            streaming=StreamingConfig(**_section(raw, "streaming")),
            endpoint=EndpointConfig(**_section(raw, "endpoint")),
            voice_clone=VoiceCloneConfig(**_section(raw, "voice_clone")),
            source_path=resolved,
        )
    except TypeError as exc:
        raise ConfigurationError(f"invalid configuration: {exc}") from exc
    return config


def apply_overrides(
    config: AppConfig,
    *,
    source_language: str | None = None,
    target_language: str | None = None,
    audio_output: bool | None = None,
    voice_consent: bool | None = None,
) -> AppConfig:
    session = replace(
        config.session,
        source_language=source_language or config.session.source_language,
        target_language=target_language or config.session.target_language,
        audio_output=(
            config.session.audio_output if audio_output is None else audio_output
        ),
    )
    env_consent = os.getenv("VOICE_CLONE_CONSENT_CONFIRMED", "").strip().lower()
    consent_from_env = env_consent in {"1", "true", "yes", "on"}
    voice = replace(
        config.voice_clone,
        consent_confirmed=(
            config.voice_clone.consent_confirmed
            or consent_from_env
            or bool(voice_consent)
        ),
    )
    result = replace(config, session=session, voice_clone=voice)
    result.validate()
    return result
