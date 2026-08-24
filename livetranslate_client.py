import os
import time
import base64
import asyncio
import json
import websockets
import pyaudio
import queue
import threading
import traceback

class LiveTranslateClient:
    def __init__(self, api_key: str, target_language: str = "en", *, 
                 audio_enabled: bool = True, 
                 enable_voice_clone: bool = False, 
                 voice_clone_frequency: str = "once"):
        if not api_key:
            raise ValueError("API key cannot be empty.")
            
        self.api_key = api_key
        self.target_language = target_language
        self.audio_enabled = audio_enabled
        self.ws = None
        
        # Voice clone configurations
        self.enable_voice_clone = enable_voice_clone
        self.voice_clone_frequency = voice_clone_frequency
        
        # Public International Endpoint
        self.api_url = "wss://dashscope-intl.aliyuncs.com/api-ws/v1/realtime?model=qwen3.5-livetranslate-flash-realtime"
        # self.api_url = f"wss://llm-46cu06h95cdluabg.ap-southeast-1.maas.aliyuncs.com/api-ws/v1/realtime?model=qwen3.5-livetranslate-flash-realtime"
        
        # 音频输入配置 (来自麦克风)
        self.input_rate = 16000
        self.input_chunk = 1600
        self.input_format = pyaudio.paInt16
        self.input_channels = 1
        
        # 音频输出配置 (用于播放)
        self.output_rate = 24000
        self.output_chunk = 2400
        self.output_format = pyaudio.paInt16
        self.output_channels = 1
        
        # 状态管理
        self.is_connected = False
        self.audio_player_thread = None
        self.audio_playback_queue = queue.Queue()
        self.pyaudio_instance = pyaudio.PyAudio()
        self.session_finished_event = asyncio.Event()

    async def connect(self):
        """建立到翻译服务的 WebSocket 连接。"""
        headers = {"Authorization": f"Bearer {self.api_key}"}
        try:
            self.ws = await websockets.connect(self.api_url, additional_headers=headers)
            self.is_connected = True
            print(f"成功连接到服务端: {self.api_url}")
            await self.configure_session()
        except Exception as e:
            print(f"连接失败: {e}")
            self.is_connected = False
            raise

    async def configure_session(self):
        """配置翻译会话，设置目标语言、声音等。"""
        config = {
            "event_id": f"event_{int(time.time() * 1000)}",
            "type": "session.update",
            "session": {
                "modalities": ["text", "audio"] if self.audio_enabled else ["text"],
                "input_audio_format": "pcm",
                "output_audio_format": "pcm",
                "translation": {
                    "language": self.target_language,
                }
            }
        }
        
        # Inject voice cloning options if enabled and audio is active
        if self.audio_enabled and self.enable_voice_clone:
            config["session"]["enable_voice_clone"] = True
            config["session"]["voice_clone_options"] = {
                "frequency": self.voice_clone_frequency
            }
            # Note: When using 'once' or 'always', voice must be set to 'default'
            config["session"]["voice"] = "default"
            
        print(f"发送会话配置: {json.dumps(config, indent=2, ensure_ascii=False)}")
        await self.ws.send(json.dumps(config))

    async def send_audio_chunk(self, audio_data: bytes):
        """将音频数据块编码并发送到服务端。"""
        if not self.is_connected:
            return

        event = {
            "event_id": f"event_{int(time.time() * 1000)}",
            "type": "input_audio_buffer.append",
            "audio": base64.b64encode(audio_data).decode()
        }
        await self.ws.send(json.dumps(event))

    async def send_image_frame(self, image_bytes: bytes, *, event_id: str | None = None):
        #将图像数据发送到服务端
        if not self.is_connected:
            return

        if not image_bytes:
            raise ValueError("image_bytes 不能为空")

        # 编码为 Base64
        image_b64 = base64.b64encode(image_bytes).decode()

        event = {
            "event_id": event_id or f"event_{int(time.time() * 1000)}",
            "type": "input_image_buffer.append",
            "image": image_b64,
        }

        await self.ws.send(json.dumps(event))

    def _audio_player_task(self):
        stream = self.pyaudio_instance.open(
            format=self.output_format,
            channels=self.output_channels,
            rate=self.output_rate,
            output=True,
            frames_per_buffer=self.output_chunk,
        )
        try:
            while self.is_connected or not self.audio_playback_queue.empty():
                try:
                    audio_chunk = self.audio_playback_queue.get(timeout=0.1)
                    if audio_chunk is None: # 结束信号
                        break
                    stream.write(audio_chunk)
                    self.audio_playback_queue.task_done()
                except queue.Empty:
                    continue
        finally:
            stream.stop_stream()
            stream.close()

    def start_audio_player(self):
        """启动音频播放线程（仅当启用音频输出时）。"""
        if not self.audio_enabled:
            return
        if self.audio_player_thread is None or not self.audio_player_thread.is_alive():
            self.audio_player_thread = threading.Thread(target=self._audio_player_task, daemon=True)
            self.audio_player_thread.start()

    async def handle_server_messages(self, on_text_received):
        """循环处理来自服务端的消息。"""
        try:
            async for message in self.ws:
                event = json.loads(message)
                event_type = event.get("type")
                if event_type == "response.audio.delta" and self.audio_enabled:
                    audio_b64 = event.get("delta", "")
                    if audio_b64:
                        audio_data = base64.b64decode(audio_b64)
                        self.audio_playback_queue.put(audio_data)

                elif event_type == "response.done":
                    print("\n[INFO] 一轮响应完成。")
                    usage = event.get("response", {}).get("usage", {})
                    if usage:
                        print(f"[INFO] Token 使用情况: {json.dumps(usage, indent=2, ensure_ascii=False)}")
                elif event_type == "session.finished":
                    print("[INFO] 会话已结束。")
                    self.session_finished_event.set()
                # 处理源语言识别结果（需启用 input_audio_transcription.model）
                # elif event_type == "conversation.item.input_audio_transcription.text":
                #     stash = event.get("stash", "")  # 待确认的识别文本
                #     print(f"[识别中] {stash}")
                # elif event_type == "conversation.item.input_audio_transcription.completed":
                #     transcript = event.get("transcript", "")  # 完整识别结果
                #     print(f"[源语言] {transcript}")
                elif event_type == "response.audio_transcript.done":
                    print("\n[INFO] 翻译文本完成。")
                    text = event.get("transcript", "")
                    if text:
                        print(f"[INFO] 翻译文本: {text}")
                elif event_type == "response.text.done":
                    print("\n[INFO] 翻译文本完成。")
                    text = event.get("text", "")
                    if text:
                        print(f"[INFO] 翻译文本: {text}")

        except websockets.exceptions.ConnectionClosed as e:
            print(f"[WARNING] 连接已关闭: {e}")
            self.is_connected = False
        except Exception as e:
            print(f"[ERROR] 消息处理时发生未知错误: {e}")
            traceback.print_exc()
            self.is_connected = False

    async def start_microphone_streaming(self):
        """从麦克风捕获音频并流式传输到服务端。"""
        stream = self.pyaudio_instance.open(
            format=self.input_format,
            channels=self.input_channels,
            rate=self.input_rate,
            input=True,
            frames_per_buffer=self.input_chunk
        )
        print("麦克风已启动，请开始说话...")
        try:
            while self.is_connected:
                audio_chunk = await asyncio.get_event_loop().run_in_executor(
                    None, stream.read, self.input_chunk
                )
                await self.send_audio_chunk(audio_chunk)
        finally:
            stream.stop_stream()
            stream.close()

    async def close(self):
        """优雅地关闭连接和资源。"""
        # 发送 session.finish，确保服务端完成最后的语音翻译
        if self.is_connected and self.ws:
            finish_event = {
                "event_id": f"event_{int(time.time() * 1000)}",
                "type": "session.finish",
            }
            await self.ws.send(json.dumps(finish_event))
            print("已发送 session.finish，等待服务端完成处理...")
            try:
                await asyncio.wait_for(self.session_finished_event.wait(), timeout=15)
                print("服务端已完成处理。")
            except asyncio.TimeoutError:
                print("等待 session.finished 超时。")

        self.is_connected = False
        if self.ws:
            await self.ws.close()
            print("WebSocket 连接已关闭。")

        if self.audio_player_thread:
            self.audio_playback_queue.put(None) # 发送结束信号
            self.audio_player_thread.join(timeout=1)
            print("音频播放线程已停止。")

        self.pyaudio_instance.terminate()
        print("PyAudio 实例已释放。")