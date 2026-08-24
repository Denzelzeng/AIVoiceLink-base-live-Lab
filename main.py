import os
import asyncio
from dotenv import load_dotenv
from livetranslate_client import LiveTranslateClient

load_dotenv()

def print_banner():
    print("=" * 60)
    print("  基于千问 qwen3.5-livetranslate-flash-realtime")
    print("=" * 60 + "\n")

def get_user_config():
    """获取用户配置"""
    print("请选择模式:")
    print("1. 语音+文本 [默认] | 2. 仅文本")
    mode_choice = input("请输入选项 (直接回车选择语音+文本): ").strip()
    audio_enabled = (mode_choice != "2")

    if audio_enabled:
        lang_map = {
            "1": "en", "2": "zh", "3": "ru", "4": "fr", "5": "de", "6": "pt",
            "7": "es", "8": "it", "9": "ko", "10": "ja", "11": "yue"
        }
        print("请选择翻译目标语言 (音频+文本 模式):")
        print("1. 英语 | 2. 中文 | 3. 俄语 | 4. 法语 | 5. 德语 | 6. 葡萄牙语 | 7. 西班牙语 | 8. 意大利语 | 9. 韩语 | 10. 日语 | 11. 粤语")
    else:
        lang_map = {
            "1": "en", "2": "zh", "3": "ru", "4": "fr", "5": "de", "6": "pt", "7": "es", "8": "it",
            "9": "id", "10": "ko", "11": "ja", "12": "vi", "13": "th", "14": "ar",
            "15": "yue", "16": "hi", "17": "el", "18": "tr"
        }
        print("请选择翻译目标语言 (仅文本 模式):")
        print("1. 英语 | 2. 中文 | 3. 俄语 | 4. 法语 | 5. 德语 | 6. 葡萄牙语 | 7. 西班牙语 | 8. 意大利语 | 9. 印尼语 | 10. 韩语 | 11. 日语 | 12. 越南语 | 13. 泰语 | 14. 阿拉伯语 | 15. 粤语 | 16. 印地语 | 17. 希腊语 | 18. 土耳其语")

    choice = input("请输入选项 (默认取第一个): ").strip()
    target_language = lang_map.get(choice, next(iter(lang_map.values())))

    return target_language, audio_enabled

async def main():
    """主程序入口"""
    print_banner()
    
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        print("[ERROR] 请设置环境变量 DASHSCOPE_API_KEY")
        print("  例如: export DASHSCOPE_API_KEY='your_api_key_here'")
        return
        
    target_language, audio_enabled = get_user_config()
    print("\n配置完成:")
    print(f"  - 目标语言: {target_language}")
    if not audio_enabled:
        print("  - 输出模式: 仅文本")

    client = LiveTranslateClient(api_key=api_key, target_language=target_language, audio_enabled=audio_enabled,
                                 enable_voice_clone=True, voice_clone_frequency="always")
    
    # 定义回调函数
    def on_translation_text(text):
        print(text, end="", flush=True)

    try:
        print("正在连接到翻译服务...")
        await client.connect()
        
        # 根据模式启动音频播放
        client.start_audio_player()
        
        print("\n" + "-" * 60)
        print("连接成功！请对着麦克风说话。")
        print("程序将实时翻译您的语音并播放结果。按 Ctrl+C 退出。")
        print("-" * 60 + "\n")

        # 并发运行消息处理和麦克风录音
        message_handler = asyncio.create_task(client.handle_server_messages(on_translation_text))
        tasks = [message_handler]
        # 无论是否启用音频输出，都需要从麦克风捕获音频进行翻译
        microphone_streamer = asyncio.create_task(client.start_microphone_streaming())
        tasks.append(microphone_streamer)

        await asyncio.gather(*tasks)

    except KeyboardInterrupt:
        print("\n\n用户中断，正在退出...")
    except Exception as e:
        print(f"\n发生严重错误: {e}")
    finally:
        print("\n正在清理资源...")
        await client.close()
        print("程序已退出。")

if __name__ == "__main__":
    asyncio.run(main())