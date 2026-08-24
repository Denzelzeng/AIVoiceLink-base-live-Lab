from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path

from qwen35_omni_client import (
    AudioPlayer,
    ClientConfig,
    MicrophoneConfig,
    MicrophoneInterpreter,
    Qwen35OmniClient,
    Qwen35OmniRealtimeClient,
    QwenOmniError,
    list_input_devices,
    resolve_credentials,
    save_pcm_wav,
)


APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parent

SOURCE_LANGUAGES = [
    ("auto-detect", "自动识别 / Auto-detect"),
    ("Chinese", "中文 / Chinese"),
    ("English", "英语 / English"),
    ("Cantonese", "粤语 / Cantonese"),
    ("Japanese", "日语 / Japanese"),
    ("Korean", "韩语 / Korean"),
    ("French", "法语 / French"),
    ("German", "德语 / German"),
    ("Spanish", "西班牙语 / Spanish"),
    ("Portuguese", "葡萄牙语 / Portuguese"),
    ("Italian", "意大利语 / Italian"),
    ("Russian", "俄语 / Russian"),
    ("Arabic", "阿拉伯语 / Arabic"),
    ("Thai", "泰语 / Thai"),
    ("Vietnamese", "越南语 / Vietnamese"),
    ("Indonesian", "印尼语 / Indonesian"),
    ("Hindi", "印地语 / Hindi"),
    ("Turkish", "土耳其语 / Turkish"),
]

TARGET_LANGUAGES = [
    ("English", "英语 / English"),
    ("Chinese", "中文 / Chinese"),
    ("Cantonese", "粤语 / Cantonese"),
    ("Japanese", "日语 / Japanese"),
    ("Korean", "韩语 / Korean"),
    ("French", "法语 / French"),
    ("German", "德语 / German"),
    ("Spanish", "西班牙语 / Spanish"),
    ("Portuguese", "葡萄牙语 / Portuguese"),
    ("Italian", "意大利语 / Italian"),
    ("Russian", "俄语 / Russian"),
    ("Thai", "泰语 / Thai"),
    ("Indonesian", "印尼语 / Indonesian"),
    ("Arabic", "阿拉伯语 / Arabic"),
    ("Vietnamese", "越南语 / Vietnamese"),
    ("Turkish", "土耳其语 / Turkish"),
    ("Finnish", "芬兰语 / Finnish"),
    ("Polish", "波兰语 / Polish"),
    ("Hindi", "印地语 / Hindi"),
    ("Dutch", "荷兰语 / Dutch"),
    ("Czech", "捷克语 / Czech"),
    ("Urdu", "乌尔都语 / Urdu"),
    ("Tagalog", "他加禄语 / Tagalog"),
    ("Swedish", "瑞典语 / Swedish"),
    ("Danish", "丹麦语 / Danish"),
    ("Hebrew", "希伯来语 / Hebrew"),
    ("Icelandic", "冰岛语 / Icelandic"),
    ("Malay", "马来语 / Malay"),
    ("Norwegian", "挪威语 / Norwegian"),
    ("Persian", "波斯语 / Persian"),
    ("Sichuan dialect", "四川话 / Sichuan dialect"),
    ("Beijing dialect", "北京话 / Beijing dialect"),
    ("Tianjin dialect", "天津话 / Tianjin dialect"),
    ("Nanjing dialect", "南京话 / Nanjing dialect"),
    ("Shaanxi dialect", "陕西话 / Shaanxi dialect"),
    ("Hokkien", "闽南语 / Hokkien"),
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Qwen3.5-Omni real-time microphone interpreter"
    )
    parser.add_argument("--source-language", help="Input language; any model-supported name")
    parser.add_argument("--target-language", help="Output language; any supported name")
    output = parser.add_mutually_exclusive_group()
    output.add_argument("--audio-output", action="store_true", dest="audio_output")
    output.add_argument("--text-only", action="store_false", dest="audio_output")
    parser.set_defaults(audio_output=None)
    parser.add_argument("--voice", default=os.getenv("QWEN35_OMNI_VOICE", "Tina"))
    parser.add_argument("--model", default=os.getenv("QWEN35_OMNI_MODEL", "qwen3.5-omni-plus"))
    parser.add_argument(
        "--realtime-model",
        default=os.getenv(
            "QWEN35_OMNI_REALTIME_MODEL", "qwen3.5-omni-flash-realtime"
        ),
        help="Omni Realtime model used for microphone interpretation",
    )
    parser.add_argument("--base-url")
    parser.add_argument("--api-key", help=argparse.SUPPRESS)
    parser.add_argument("--device-index", type=int)
    parser.add_argument("--energy-threshold", type=int, default=350)
    parser.add_argument("--calibration-seconds", type=float, default=1.0)
    parser.add_argument("--end-silence-ms", type=int, default=600)
    parser.add_argument("--max-segment-ms", type=int, default=7000)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument("--check", action="store_true", help="List available models and exit")
    parser.add_argument("--text", help="Translate one typed string instead of microphone audio")
    parser.add_argument("--file", type=Path, help="Translate one audio file")
    parser.add_argument(
        "--realtime",
        action="store_true",
        help="Use Qwen3.5-Omni-Realtime for --file (microphone always uses it)",
    )
    parser.add_argument(
        "--save-audio",
        type=Path,
        help="Save speech from --text/--file as a WAV file (enables audio output)",
    )
    return parser


def choose(label: str, choices: list[tuple[str, str]], default_index: int = 0) -> str:
    print(f"\n{label}")
    for index, (_, display) in enumerate(choices, 1):
        marker = " [默认]" if index - 1 == default_index else ""
        print(f"  {index:>2}. {display}{marker}")
    value = input("请输入编号，或直接输入语言名称: ").strip()
    if not value:
        return choices[default_index][0]
    if value.isdigit() and 1 <= int(value) <= len(choices):
        return choices[int(value) - 1][0]
    return value


def interactive_config(args: argparse.Namespace) -> None:
    print("=" * 64)
    print(" Qwen3.5-Omni 实时语音翻译 / Real-time Interpreter")
    print("=" * 64)
    args.source_language = args.source_language or choose("选择输入语言", SOURCE_LANGUAGES)
    args.target_language = args.target_language or choose("选择输出语言", TARGET_LANGUAGES)
    if args.audio_output is None:
        answer = input("\n播放语音译文？[Y/n]: ").strip().lower()
        args.audio_output = answer not in {"n", "no", "0"}


def complete_defaults(args: argparse.Namespace) -> None:
    args.source_language = args.source_language or os.getenv(
        "QWEN35_OMNI_SOURCE_LANGUAGE", "auto-detect"
    )
    args.target_language = args.target_language or os.getenv(
        "QWEN35_OMNI_TARGET_LANGUAGE", "English"
    )
    if args.audio_output is None:
        args.audio_output = args.save_audio is not None


async def translate_once(args: argparse.Namespace, client: Qwen35OmniClient) -> int:
    player = AudioPlayer() if args.audio_output and not args.save_audio else None
    if player:
        player.start()

    def show_text(value: str) -> None:
        print(value, end="", flush=True)

    print("译文: ", end="", flush=True)
    try:
        if args.text is not None:
            result = await client.translate_text(
                args.text,
                on_text=show_text,
                on_audio=player.put if player else None,
            )
        else:
            audio_path = args.file.resolve()
            if not audio_path.is_file():
                raise QwenOmniError(f"Audio file not found: {audio_path}")
            audio_format = audio_path.suffix.lower().lstrip(".")
            if audio_format not in {"wav", "mp3", "flac", "ogg", "m4a", "aac", "amr", "3gp"}:
                raise QwenOmniError(f"Unsupported audio format: .{audio_format}")
            result = await client.translate_audio(
                audio_path.read_bytes(),
                audio_format=audio_format,
                on_text=show_text,
                on_audio=player.put if player else None,
            )
        if not result.text:
            print("（空响应）", end="")
        print()
        if result.input_transcript or result.input_language:
            language = result.input_language or "unknown"
            print(f"识别语言: {language}")
            print(f"输入原文: {result.input_transcript or '（未返回输入转写）'}")
        if args.save_audio:
            if not result.audio:
                print("未返回语音，因此没有写入音频文件。")
            else:
                output_path = args.save_audio.resolve()
                save_pcm_wav(output_path, result.audio)
                print(f"语音已保存: {output_path}")
        elif args.audio_output and not result.audio:
            print("[警告] API 已返回文本，但没有返回可播放的语音数据。")
        return 0
    finally:
        if player:
            player.close()


async def run(args: argparse.Namespace) -> int:
    if args.list_devices:
        devices = list_input_devices()
        if not devices:
            print("未发现麦克风输入设备。")
            return 1
        for index, name in devices:
            print(f"{index}: {name}")
        return 0

    microphone_mode = not any((args.check, args.text is not None, args.file))
    if microphone_mode:
        interactive_config(args)
    else:
        complete_defaults(args)

    credentials = resolve_credentials(
        REPO_ROOT,
        api_key=args.api_key,
        base_url=args.base_url,
    )
    config = ClientConfig(
        api_key=credentials.api_key,
        base_url=credentials.base_url,
        model=args.model,
        source_language=args.source_language,
        target_language=args.target_language,
        audio_output=args.audio_output,
        voice=args.voice,
        max_tokens=args.max_tokens,
    )

    use_realtime = microphone_mode or (args.realtime and args.file is not None)
    if args.realtime and args.text is not None:
        raise QwenOmniError("--realtime currently supports microphone or --file WAV input")
    active_model = args.realtime_model if use_realtime else config.model
    print(
        f"配置: {config.source_language} -> {config.target_language}; "
        f"输出: {'文本 + 语音' if config.audio_output else '仅文本'}; "
        f"模型: {active_model}"
    )
    print(f"凭据来源: {credentials.source}")
    print(f"音色: {config.voice}")

    if use_realtime:
        async with Qwen35OmniRealtimeClient(
            config,
            model=args.realtime_model,
        ) as realtime_client:
            print("Omni Realtime 已连接；语音由模型 response.audio.delta 返回。")
            if args.file is not None:
                return await translate_once(args, realtime_client)
            microphone_config = MicrophoneConfig(
                energy_threshold=args.energy_threshold,
                calibration_seconds=args.calibration_seconds,
                end_silence_ms=args.end_silence_ms,
                max_segment_ms=args.max_segment_ms,
            )
            interpreter = MicrophoneInterpreter(
                realtime_client,
                microphone_config,
                input_device_index=args.device_index,
                play_audio=args.audio_output,
            )
            await interpreter.run()
            return 0

    async with Qwen35OmniClient(config) as client:
        if args.check:
            models = await client.list_models()
            relevant = [model for model in models if "omni" in model.lower()]
            print("API 连接成功。可用 Omni 模型:")
            for model in relevant:
                print(f"  - {model}")
            if config.model not in models:
                raise QwenOmniError(
                    f"Configured model {config.model!r} was not returned by /models"
                )
            print(f"已确认配置模型: {config.model}")
            if args.realtime_model not in models:
                raise QwenOmniError(
                    f"Configured Realtime model {args.realtime_model!r} "
                    "was not returned by /models"
                )
            print(f"已确认实时模型: {args.realtime_model}")
            return 0
        if args.text is not None or args.file:
            return await translate_once(args, client)
        return 0


def main() -> int:
    args = build_parser().parse_args()
    try:
        return asyncio.run(run(args))
    except KeyboardInterrupt:
        print("\n已停止。")
        return 130
    except OSError as exc:
        if getattr(exc, "errno", None) == -9999:
            print(
                "[麦克风错误] Windows 音频宿主拒绝打开该设备（PortAudio -9999）。\n"
                "请关闭旧的 Python 错误弹窗和占用麦克风的程序，确认 Windows 设置 > "
                "隐私和安全性 > 麦克风 > 允许桌面应用访问麦克风，然后重试。\n"
                "可先运行 --list-devices，再用 --device-index N 指定另一条麦克风端点。"
            )
        elif getattr(exc, "errno", None) == -9996:
            print(
                "[麦克风错误] 指定的设备编号无效（PortAudio -9996）。"
                "请重新运行 --list-devices 获取当前编号。"
            )
        else:
            print(f"[错误] {exc}")
        return 1
    except (QwenOmniError, ValueError) as exc:
        print(f"[错误] {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
