from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv

try:
    from .qwen3_omni_client import (
        ASRConfig,
        ClientConfig,
        InterpretationPipeline,
        MicrophoneConfig,
        MicrophoneInterpreter,
        Qwen3ASRClient,
        Qwen3OmniClient,
        Qwen3OmniError,
        list_input_devices,
        read_audio_file,
    )
except ImportError:
    from qwen3_omni_client import (  # type: ignore[no-redef]
        ASRConfig,
        ClientConfig,
        InterpretationPipeline,
        MicrophoneConfig,
        MicrophoneInterpreter,
        Qwen3ASRClient,
        Qwen3OmniClient,
        Qwen3OmniError,
        list_input_devices,
        read_audio_file,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Qwen3-Omni low-latency speech interpretation client"
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("QWEN_OMNI_BASE_URL", "http://172.26.63.11:8003/v1"),
        help="OpenAI-compatible API base URL",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("QWEN_OMNI_MODEL", "qwen3-omni"),
    )
    parser.add_argument(
        "--asr-base-url",
        default=os.getenv("QWEN_ASR_BASE_URL", "http://172.26.63.11:8004/v1"),
    )
    parser.add_argument(
        "--asr-model",
        default=os.getenv("QWEN_ASR_MODEL", "qwen3-asr"),
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("QWEN_OMNI_API_KEY") or None,
        help="Optional bearer token; the supplied local deployment needs none",
    )
    parser.add_argument(
        "--source-language",
        default=os.getenv("QWEN_OMNI_SOURCE_LANGUAGE", "auto-detect"),
    )
    parser.add_argument(
        "--target-language",
        default=os.getenv("QWEN_OMNI_TARGET_LANGUAGE", "English"),
    )
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--device-index", type=int)
    parser.add_argument("--energy-threshold", type=int, default=350)
    parser.add_argument("--calibration-seconds", type=float, default=1.0)
    parser.add_argument("--end-silence-ms", type=int, default=650)
    parser.add_argument("--max-segment-ms", type=int, default=6000)
    parser.add_argument(
        "--file",
        type=Path,
        help="Translate one audio file instead of using the microphone",
    )
    parser.add_argument(
        "--stdin",
        action="store_true",
        help="Interpret typed transcript lines (works without an audio backend)",
    )
    parser.add_argument(
        "--text",
        help="Translate one transcript string, then exit",
    )
    parser.add_argument(
        "--audio-backend",
        choices=("asr", "direct"),
        default=os.getenv("QWEN_AUDIO_BACKEND", "asr"),
        help="Use port 8004 ASR (default) or direct Qwen3-Omni audio",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check /health and /v1/models, then exit",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List microphone input devices, then exit",
    )
    return parser


async def run_audio_mode(
    args: argparse.Namespace,
    pipeline: InterpretationPipeline,
) -> int:
    print(
        f"同传配置: {args.source_language} -> {args.target_language}; "
        f"音频后端: {args.audio_backend}; 输出: 流式文本"
    )
    if args.file:
        audio_bytes, audio_format = read_audio_file(args.file)
        translation_started = False

        def show_transcript(text: str) -> None:
            print(f"[原文] {text}")

        def show_delta(text: str) -> None:
            nonlocal translation_started
            if not translation_started:
                print("[译文] ", end="", flush=True)
                translation_started = True
            print(text, end="", flush=True)

        translation = await pipeline.interpret_audio(
            audio_bytes,
            audio_format=audio_format,
            on_transcript=show_transcript,
            on_delta=show_delta,
        )
        if not translation:
            print("[译文] （未识别到可翻译语音）", end="")
        print()
        return 0

    microphone_config = MicrophoneConfig(
        energy_threshold=args.energy_threshold,
        calibration_seconds=args.calibration_seconds,
        end_silence_ms=args.end_silence_ms,
        max_segment_ms=args.max_segment_ms,
    )
    interpreter = MicrophoneInterpreter(
        pipeline,
        microphone_config,
        input_device_index=args.device_index,
    )
    await interpreter.run()
    return 0


async def run(args: argparse.Namespace) -> int:
    if args.list_devices:
        devices = list_input_devices()
        if not devices:
            print("未发现麦克风输入设备。")
            return 1
        for index, name in devices:
            print(f"{index}: {name}")
        return 0

    client_config = ClientConfig(
        base_url=args.base_url,
        model=args.model,
        api_key=args.api_key,
        source_language=args.source_language,
        target_language=args.target_language,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )

    asr_config = ASRConfig(
        base_url=args.asr_base_url,
        model=args.asr_model,
        api_key=args.api_key,
    )

    async with Qwen3OmniClient(client_config) as client:
        status = await client.check_server()
        print(
            f"Qwen 翻译 API 可达（HTTP {status['health_status']}），"
            f"模型: {', '.join(status['models']) or '无'}"
        )
        if not status["configured_model_available"]:
            raise Qwen3OmniError(
                f"Configured model {args.model!r} was not returned by /v1/models"
            )
        if args.stdin:
            print("输入原文并按回车翻译；空行退出。")
            previous_translation = ""
            while True:
                transcript = await asyncio.to_thread(input, "[原文] ")
                if not transcript.strip():
                    return 0
                print("[译文] ", end="", flush=True)
                translation = await client.translate_text(
                    transcript,
                    previous_translation=previous_translation,
                    on_delta=lambda text: print(text, end="", flush=True),
                )
                print()
                if translation:
                    previous_translation = translation

        if args.text:
            print("[译文] ", end="", flush=True)
            translation = await client.translate_text(
                args.text,
                on_delta=lambda text: print(text, end="", flush=True),
            )
            if not translation:
                print("（空响应）", end="")
            print()
            return 0

        if args.audio_backend == "direct":
            if args.check:
                return 0
            pipeline = InterpretationPipeline(client, direct_audio=True)
            return await run_audio_mode(args, pipeline)

        async with Qwen3ASRClient(asr_config) as asr:
            asr_status = await asr.check_server()
            print(
                f"Qwen3-ASR API 可达（HTTP {asr_status['health_status']}），"
                f"模型: {', '.join(asr_status['models']) or '无'}"
            )
            if not asr_status["configured_model_available"]:
                raise Qwen3OmniError(
                    f"Configured ASR model {args.asr_model!r} was not returned "
                    "by the ASR /v1/models endpoint"
                )
            if args.check:
                return 0

            pipeline = InterpretationPipeline(
                client,
                asr=asr,
                direct_audio=False,
            )
            return await run_audio_mode(args, pipeline)


def main() -> int:
    load_dotenv()
    parser = build_parser()
    args = parser.parse_args()
    try:
        return asyncio.run(run(args))
    except KeyboardInterrupt:
        print("\n已停止。")
        return 130
    except (Qwen3OmniError, OSError, ValueError) as exc:
        print(f"[错误] {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
