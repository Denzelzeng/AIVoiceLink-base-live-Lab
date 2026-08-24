from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

from .audio import MicrophoneWindowSource, WavWindowSource, list_input_devices
from .config import AppConfig, ConfigurationError, apply_overrides, load_config
from .endpoint import build_endpoint
from .events import AudioWindow
from .pipeline import RealtimeInterpretationPipeline
from .providers.mock import MockCloningTTS, RuleBasedMockTranslator, ScriptedASR
from .providers.cloud_api import (
    DashScopeASR,
    DashScopeVoiceCloneTTS,
    ProviderError,
    QwenMTTranslator,
)
from .render import ConsoleRenderer, EventFanout, JsonlRecorder
from .speaker import build_speaker_change_detector
from .sinks import NullAudioSink, PyAudioSink, WavDirectorySink
from .vad import build_speech_detector


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "cloud_api.toml"
MOCK_CONFIG = PROJECT_ROOT / "configs" / "mock.toml"
LANGUAGES = (
    "Chinese",
    "English",
    "Japanese",
    "German",
    "Korean",
    "Russian",
    "French",
    "Portuguese",
    "Italian",
    "Spanish",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Modular real-time ASR -> incremental MT -> voice-cloned TTS baseline"
        )
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run microphone or WAV interpretation")
    _common_config_arguments(run)
    run.add_argument(
        "--wav",
        type=Path,
        help="Use a mono PCM16 WAV matching the configured capture sample rate",
    )
    run.add_argument("--no-realtime", action="store_true", help="Do not pace WAV input")
    run.add_argument("--device-index", type=int)
    run.add_argument("--save-audio-dir", type=Path)
    run.add_argument("--events", type=Path, help="Append all events and metrics as JSONL")
    run.add_argument(
        "--no-language-prompt",
        action="store_true",
        help="Use config/CLI languages without the interactive startup selector",
    )

    demo = sub.add_parser("demo", help="Run an offline deterministic control-flow demo")
    demo.add_argument("--audio-output", action="store_true")
    demo.add_argument("--save-audio-dir", type=Path, default=PROJECT_ROOT / "output" / "demo_audio")
    demo.add_argument("--events", type=Path)

    doctor = sub.add_parser("doctor", help="Check configured model endpoints")
    _common_config_arguments(doctor)

    sub.add_parser("devices", help="List microphone input devices")
    return parser


def _common_config_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--source-language")
    parser.add_argument("--target-language")
    output = parser.add_mutually_exclusive_group()
    output.add_argument("--audio-output", action="store_true", dest="audio_output")
    output.add_argument("--text-only", action="store_false", dest="audio_output")
    parser.set_defaults(audio_output=None)
    parser.add_argument(
        "--voice-consent",
        action="store_true",
        help="Confirm that the speaker authorized session-scoped voice cloning",
    )


def _load_from_args(args: argparse.Namespace) -> AppConfig:
    return apply_overrides(
        load_config(args.config),
        source_language=args.source_language,
        target_language=args.target_language,
        audio_output=args.audio_output,
        voice_consent=args.voice_consent,
    )


def _prompt_language(
    label: str,
    default: str,
    *,
    allow_auto: bool,
    exclude: set[str] | None = None,
) -> str:
    all_choices = (("Auto",) + LANGUAGES) if allow_auto else LANGUAGES
    excluded = {value.casefold() for value in (exclude or set())}
    choices = tuple(
        language for language in all_choices if language.casefold() not in excluded
    )
    if default.casefold() in excluded:
        default = choices[0]
    print(f"\n{label}（直接回车使用 {default}）：")
    for index, language in enumerate(choices, start=1):
        print(f"  {index:>2}. {language}")
    aliases = {language.casefold(): language for language in choices}
    while True:
        value = input("> ").strip()
        if not value:
            return default
        if value.isdigit() and 1 <= int(value) <= len(choices):
            return choices[int(value) - 1]
        if value.casefold() in aliases:
            return aliases[value.casefold()]
        print("无效选择，请输入序号或语言英文名。")


def _select_languages(args: argparse.Namespace, config: AppConfig) -> tuple[str, str]:
    source = args.source_language
    target = args.target_language
    should_prompt = (
        not args.no_language_prompt
        and sys.stdin.isatty()
        and (source is None or target is None)
    )
    if should_prompt:
        if source is None:
            source = _prompt_language(
                "请选择输入语言", config.session.source_language, allow_auto=True
            )
        if target is None:
            target = _prompt_language(
                "请选择输出语言",
                config.session.target_language,
                allow_auto=False,
                exclude={source} if source.casefold() != "auto" else None,
            )
    return source or config.session.source_language, target or config.session.target_language


def _build_services(config: AppConfig):
    if config.asr.provider == "mock":
        asr = ScriptedASR(
            [
                "大家好",
                "大家好，欢迎",
                "大家好，欢迎参加今天的会议。",
            ]
        )
    elif config.asr.provider == "dashscope_asr":
        asr = DashScopeASR(config.asr)
    else:
        raise ConfigurationError(f"unsupported ASR API provider: {config.asr.provider}")
    if config.mt.provider == "mock":
        translator = RuleBasedMockTranslator()
    elif config.mt.provider == "qwen_mt":
        translator = QwenMTTranslator(config.mt)
    else:
        raise ConfigurationError(f"unsupported MT API provider: {config.mt.provider}")
    if config.tts.provider == "disabled":
        tts = None
    elif config.tts.provider == "mock":
        tts = MockCloningTTS(config.tts.sample_rate)
    elif config.tts.provider == "dashscope_qwen_voice_clone":
        tts = DashScopeVoiceCloneTTS(config.tts)
    else:
        raise ConfigurationError(f"unsupported TTS API provider: {config.tts.provider}")
    return asr, translator, build_endpoint(config.endpoint), tts


def _event_handler(events_path: Path | None):
    renderer = ConsoleRenderer()
    if events_path is None:
        return EventFanout(renderer), None
    recorder = JsonlRecorder(events_path)
    return EventFanout(renderer, recorder), recorder


async def _run_pipeline(args: argparse.Namespace, *, demo: bool = False) -> int:
    if demo:
        config = load_config(MOCK_CONFIG)
        config = apply_overrides(
            config,
            audio_output=args.audio_output,
            voice_consent=True,
        )
    else:
        base_config = load_config(args.config)
        source_language, target_language = _select_languages(args, base_config)
        config = apply_overrides(
            base_config,
            source_language=source_language,
            target_language=target_language,
            audio_output=args.audio_output,
            voice_consent=args.voice_consent,
        )
    asr, translator, endpoint, tts = _build_services(config)
    speech_detector = build_speech_detector(
        config.vad,
        config_path=config.source_path,
    )
    speaker_change_detector = (
        build_speaker_change_detector(
            config.speaker_change,
            config_path=config.source_path,
        )
        if config.session.audio_output and config.voice_clone.enabled
        else None
    )
    if config.session.audio_output:
        if args.save_audio_dir:
            sink = WavDirectorySink(args.save_audio_dir)
        elif demo:
            sink = WavDirectorySink(PROJECT_ROOT / "output" / "demo_audio")
        else:
            sink = PyAudioSink()
    else:
        sink = NullAudioSink()
    handler, recorder = _event_handler(args.events)
    pipeline = RealtimeInterpretationPipeline(
        config,
        asr=asr,
        translator=translator,
        endpoint=endpoint,
        tts=tts,
        audio_sink=sink,
        event_handler=handler,
        speaker_change_detector=speaker_change_detector,
    )
    if demo:
        source = _DemoSource(pipeline.on_speech_started)
    elif args.wav:
        source = WavWindowSource(
            args.wav,
            config.audio,
            real_time=not args.no_realtime,
            speech_detector=speech_detector,
            on_speech_started=pipeline.on_speech_started,
        )
    else:
        print(
            f"麦克风将校准环境噪声 {config.audio.calibration_seconds:g} 秒。"
            "请先保持安静，看到“现在可以开始讲话”后再说；按 Ctrl+C 退出。"
        )
        if config.session.audio_output and config.voice_clone.enabled:
            print(
                "首次克隆语音需累计至少 3 秒清晰人声；首段译文会等待声纹就绪，"
                "不会直接跳过。"
            )
            if not config.streaming.barge_in_enabled:
                print(
                    "译音播放链路独立：继续讲话不会暂停或取消已排队的翻译语音；"
                    "建议使用耳机避免扬声器回声进入麦克风。"
                )
            if speaker_change_detector is not None:
                print("本地说话人比对已启用：只有确认换人后才注册新的克隆音色。")
        source = MicrophoneWindowSource(
            config.audio,
            input_device_index=args.device_index,
            speech_detector=speech_detector,
            on_speech_started=pipeline.on_speech_started,
            on_ready=lambda threshold: print(
                f"[麦克风] 校准完成（噪声基线 {threshold}；"
                f"VAD={config.vad.provider}），现在可以开始讲话。"
            ),
        )
    try:
        await pipeline.run(source)
    finally:
        if recorder:
            recorder.close()
    return 0


class _DemoSource:
    def __init__(self, on_speech_started):
        self.on_speech_started = on_speech_started

    async def __aiter__(self):
        await self.on_speech_started()
        started = time.monotonic()
        for duration_ms, final in ((1_200, False), (2_600, False), (3_300, True)):
            pcm = b"\x01\x00" * round(16_000 * duration_ms / 1_000)
            yield AudioWindow(
                turn_id=1,
                pcm=pcm,
                sample_rate=16_000,
                is_final=final,
                started_at=started,
                captured_at=time.monotonic(),
            )
            await asyncio.sleep(0.03)


async def _doctor(args: argparse.Namespace) -> int:
    config = _load_from_args(args)
    local_failed = False
    try:
        detector = build_speech_detector(
            config.vad,
            config_path=config.source_path,
        )
        label = type(detector).__name__ if detector is not None else "energy"
        print(f"[ OK ] Local VAD: {label}")
    except Exception as exc:
        local_failed = True
        print(f"[FAIL] Local VAD: {exc}")
    if config.speaker_change.enabled and config.session.audio_output:
        try:
            speaker = build_speaker_change_detector(
                config.speaker_change,
                config_path=config.source_path,
            )
            print(f"[ OK ] Speaker change: {type(speaker).__name__}")
        except Exception as exc:
            local_failed = True
            print(f"[FAIL] Speaker change: {exc}")
    asr, translator, endpoint, tts = _build_services(config)
    checks = [
        ("ASR", asr.health()),
        ("MT", translator.health()),
        ("Endpoint", endpoint.health()),
    ]
    if tts and config.session.audio_output:
        checks.append(("Clone TTS", tts.health()))
    results = await asyncio.gather(
        *(check for _, check in checks), return_exceptions=True
    )
    failed = local_failed
    for (label, _), result in zip(checks, results, strict=True):
        if isinstance(result, Exception):
            failed = True
            print(f"[FAIL] {label}: {result}")
        elif result.get("configured_model_available") is False:
            failed = True
            print(f"[FAIL] {label}: {result}")
        else:
            print(f"[ OK ] {label}: {result}")
    await asr.aclose()
    await translator.aclose()
    await endpoint.aclose()
    if tts:
        await tts.aclose()
    return 1 if failed else 0


def main() -> int:
    args = build_parser().parse_args()
    try:
        if args.command == "devices":
            for index, name in list_input_devices():
                print(f"{index}: {name}")
            return 0
        if args.command == "doctor":
            return asyncio.run(_doctor(args))
        if args.command == "demo":
            return asyncio.run(_run_pipeline(args, demo=True))
        return asyncio.run(_run_pipeline(args))
    except KeyboardInterrupt:
        print("\n已停止。")
        return 130
    except (ConfigurationError, ProviderError, OSError, RuntimeError, ValueError) as exc:
        print(f"[错误] {exc}")
        return 1
