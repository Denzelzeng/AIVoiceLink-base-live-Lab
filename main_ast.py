import asyncio
import os
from typing import Optional

from dotenv import load_dotenv

from ast_live_client import AstAuth, AstLiveClient


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def print_banner():
    print("=" * 60)
    print("  基于火山引擎 AST 同传大模型")
    print("=" * 60 + "\n")


def choose_language(title: str, options: dict[str, str], default: str) -> str:
    print(title)
    for key, label in options.items():
        print(f"{key}. {label}")
    choice = input(f"请输入选项 (直接回车选择 {default}): ").strip()
    return options.get(choice, default).split(" ")[0]


def get_user_config():
    print("请选择模式:")
    print("1. 语音+文本 [默认] | 2. 仅文本")
    mode_choice = input("请输入选项 (直接回车选择语音+文本): ").strip()
    audio_enabled = mode_choice != "2"

    source_options = {
        "1": "zh 中文",
        "2": "en 英语",
        "3": "ja 日语",
        "4": "ko 韩语",
        "5": "fr 法语",
        "6": "es 西班牙语",
        "7": "de 德语",
        "8": "ru 俄语",
        "9": "pt 葡萄牙语",
        "10": "it 意大利语",
        "11": "zhen 中英反转互译",
    }
    target_options = {
        "1": "en 英语",
        "2": "zh 中文",
        "3": "ja 日语",
        "4": "ko 韩语",
        "5": "fr 法语",
        "6": "es 西班牙语",
        "7": "de 德语",
        "8": "ru 俄语",
        "9": "pt 葡萄牙语",
        "10": "it 意大利语",
        "11": "zhen 中英反转互译",
    }

    source_language = os.environ.get("AST_SOURCE_LANGUAGE") or choose_language(
        "请选择源语言:",
        source_options,
        "zh",
    )
    target_language = os.environ.get("AST_TARGET_LANGUAGE") or choose_language(
        "请选择目标语言:",
        target_options,
        "en",
    )

    if source_language == "zhen" or target_language == "zhen":
        source_language = "zhen"
        target_language = "zhen"

    return source_language, target_language, audio_enabled


def get_auth() -> Optional[AstAuth]:
    api_key = os.environ.get("AST_API_KEY")
    app_key = os.environ.get("AST_APP_KEY") or os.environ.get("AST_APP_ID")
    access_key = os.environ.get("AST_ACCESS_KEY")
    resource_id = os.environ.get("AST_RESOURCE_ID", "volc.service_type.10053")

    if api_key:
        return AstAuth(api_key=api_key, resource_id=resource_id)
    if app_key and access_key:
        return AstAuth(app_key=app_key, access_key=access_key, resource_id=resource_id)
    return None


class LivePrinter:
    def __init__(self):
        self.last_partial_len = 0

    def print_text(self, label: str, text: str, final: bool):
        if not text:
            return
        line = f"[{label}] {text}"
        if final:
            print("\r" + " " * self.last_partial_len + "\r" + line)
            self.last_partial_len = 0
        else:
            print("\r" + " " * self.last_partial_len + "\r" + line, end="", flush=True)
            self.last_partial_len = len(line)


async def main():
    print_banner()
    load_dotenv()

    auth = get_auth()
    if not auth:
        print("[ERROR] 请在 .env 中设置 AST_API_KEY，或设置 AST_APP_KEY + AST_ACCESS_KEY。")
        return

    source_language, target_language, audio_enabled = get_user_config()
    enable_voice_clone = _env_bool("AST_ENABLE_VOICE_CLONE", True)
    tts_resource_id = os.environ.get("AST_TTS_RESOURCE_ID", "seed-icl-2.0")

    print("\n配置完成:")
    print(f"  - 源语言: {source_language}")
    print(f"  - 目标语言: {target_language}")
    print(f"  - 输出模式: {'语音+文本' if audio_enabled else '仅文本'}")
    if audio_enabled:
        print(f"  - 声音复刻: {'启用' if enable_voice_clone else '关闭'}")

    client = AstLiveClient(
        auth=auth,
        source_language=source_language,
        target_language=target_language,
        audio_enabled=audio_enabled,
        enable_voice_clone=enable_voice_clone,
        tts_resource_id=tts_resource_id,
    )
    printer = LivePrinter()

    message_handler = None
    microphone_streamer = None

    try:
        print("正在连接到 AST 翻译服务...")
        await client.connect()
        client.start_audio_player()

        print("\n" + "-" * 60)
        print("连接成功！请对着麦克风说话。")
        print("程序将实时显示原文/译文，并在语音模式下播放翻译结果。按 Ctrl+C 退出。")
        print("-" * 60 + "\n")

        message_handler = asyncio.create_task(
            client.handle_server_messages(
                lambda text, final: printer.print_text("翻译", text, final),
                lambda text, final: printer.print_text("原文", text, final),
            )
        )
        microphone_streamer = asyncio.create_task(client.start_microphone_streaming())

        await asyncio.gather(message_handler, microphone_streamer)

    except KeyboardInterrupt:
        print("\n\n用户中断，正在退出...")
    except Exception as exc:
        print(f"\n发生严重错误: {exc}")
    finally:
        print("\n正在清理资源...")
        if microphone_streamer and not microphone_streamer.done():
            microphone_streamer.cancel()
            await asyncio.gather(microphone_streamer, return_exceptions=True)

        await client.close()

        if message_handler and not message_handler.done():
            message_handler.cancel()
            await asyncio.gather(message_handler, return_exceptions=True)

        print("程序已退出。")


if __name__ == "__main__":
    asyncio.run(main())
