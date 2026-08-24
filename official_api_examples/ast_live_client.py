import asyncio
import json
import queue
import sys
import threading
import time
import traceback
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import pyaudio
import websockets
from google.protobuf.json_format import MessageToDict


AST_PYTHON_DIR = Path(__file__).resolve().parents[1] / "ast_python"
if str(AST_PYTHON_DIR) not in sys.path:
    sys.path.append(str(AST_PYTHON_DIR))

from python_protogen.common.events_pb2 import Type
from python_protogen.products.understanding.ast.ast_service_pb2 import TranslateRequest, TranslateResponse


AST_WS_URL = "wss://openspeech.bytedance.com/api/v4/ast/v2/translate"
AST_RESOURCE_ID = "volc.service_type.10053"

INPUT_RATE = 16000
INPUT_BITS = 16
INPUT_CHANNELS = 1
INPUT_FORMAT = pyaudio.paInt16
INPUT_CHUNK_MS = 80
INPUT_FRAMES_PER_BUFFER = int(INPUT_RATE * INPUT_CHUNK_MS / 1000)

OUTPUT_RATE = 24000
OUTPUT_BITS = 32
OUTPUT_CHANNELS = 1
OUTPUT_FORMAT = pyaudio.paFloat32
OUTPUT_FRAMES_PER_BUFFER = int(OUTPUT_RATE * 100 / 1000)


@dataclass
class AstAuth:
    api_key: Optional[str] = None
    app_key: Optional[str] = None
    access_key: Optional[str] = None
    resource_id: str = AST_RESOURCE_ID


@dataclass
class AstResponse:
    event: int
    session_id: str
    sequence: int
    text: str
    data: bytes
    start_time: int
    end_time: int
    spk_chg: bool
    message: str


def _set_proto_field(message, field_name: str, value) -> None:
    if hasattr(message, "DESCRIPTOR") and field_name in message.DESCRIPTOR.fields_by_name:
        setattr(message, field_name, value)


def _event_name(event_value: int) -> str:
    try:
        return Type.Name(event_value)
    except ValueError:
        return str(event_value)


class AstLiveClient:
    def __init__(
        self,
        auth: AstAuth,
        *,
        source_language: str = "zh",
        target_language: str = "en",
        audio_enabled: bool = True,
        enable_voice_clone: bool = True,
        tts_resource_id: str = "seed-icl-2.0",
        speech_rate: int = 0,
        ws_url: str = AST_WS_URL,
    ):
        if not auth.api_key and not (auth.app_key and auth.access_key):
            raise ValueError("AST auth requires AST_API_KEY or AST_APP_KEY + AST_ACCESS_KEY.")

        self.auth = auth
        self.source_language = source_language
        self.target_language = target_language
        self.audio_enabled = audio_enabled
        self.mode = "s2s" if audio_enabled else "s2t"
        self.enable_voice_clone = enable_voice_clone and audio_enabled
        self.tts_resource_id = tts_resource_id
        self.speech_rate = speech_rate
        self.ws_url = ws_url

        self.ws = None
        self.session_id = str(uuid.uuid4())
        self.connect_id = str(uuid.uuid4())
        self.is_connected = False
        self.finish_sent = False

        self.audio_player_thread = None
        self.audio_playback_queue = queue.Queue()
        self.pyaudio_instance = pyaudio.PyAudio()
        self.session_finished_event = asyncio.Event()

    def _build_headers(self) -> dict[str, str]:
        headers = {
            "X-Api-Resource-Id": self.auth.resource_id,
            "X-Api-Connect-Id": self.connect_id,
        }
        if self.auth.api_key:
            headers["X-Api-Key"] = self.auth.api_key
        else:
            headers["X-Api-App-Key"] = self.auth.app_key or ""
            headers["X-Api-Access-Key"] = self.auth.access_key or ""
        return headers

    async def _connect_ws(self):
        headers = self._build_headers()
        try:
            return await websockets.connect(
                self.ws_url,
                additional_headers=headers,
                max_size=1_000_000_000,
                ping_interval=None,
            )
        except TypeError as exc:
            if "additional_headers" not in str(exc):
                raise
            return await websockets.connect(
                self.ws_url,
                extra_headers=headers,
                max_size=1_000_000_000,
                ping_interval=None,
            )

    async def connect(self):
        self.ws = await self._connect_ws()
        self.is_connected = True
        print(f"成功连接到 AST 服务端: {self.ws_url}")

        await self._send_start_session()
        response = await self._receive_message()
        if response.event != Type.SessionStarted:
            self.is_connected = False
            raise RuntimeError(
                f"AST StartSession failed: event={_event_name(response.event)}, "
                f"message={response.message or response.text}"
            )
        print(f"AST 会话已启动: {self.session_id}")

    async def _send_start_session(self):
        request = TranslateRequest()
        request.request_meta.SessionID = self.session_id
        request.event = Type.StartSession
        request.user.uid = "ast_live_demo"
        request.user.did = "ast_live_demo"
        request.user.platform = sys.platform
        request.user.sdk_version = "live_demo"

        request.source_audio.format = "pcm"
        _set_proto_field(request.source_audio, "codec", "raw")
        request.source_audio.rate = INPUT_RATE
        request.source_audio.bits = INPUT_BITS
        request.source_audio.channel = INPUT_CHANNELS

        request.target_audio.format = "pcm"
        request.target_audio.rate = OUTPUT_RATE
        request.target_audio.bits = OUTPUT_BITS
        request.target_audio.channel = OUTPUT_CHANNELS

        request.request.mode = self.mode
        request.request.source_language = self.source_language
        request.request.target_language = self.target_language
        request.request.speech_rate = self.speech_rate

        if self.enable_voice_clone:
            _set_proto_field(request.request, "is_custom_speaker", True)
            if self.tts_resource_id:
                _set_proto_field(request.request, "tts_resource_id", self.tts_resource_id)

        await self.ws.send(request.SerializeToString())

    async def _send_audio_chunk(self, audio_data: bytes):
        if not self.is_connected or not self.ws:
            return

        request = TranslateRequest()
        request.request_meta.SessionID = self.session_id
        request.event = Type.TaskRequest
        request.source_audio.binary_data = audio_data
        await self.ws.send(request.SerializeToString())

    async def _send_finish_session(self):
        if not self.is_connected or not self.ws or self.finish_sent:
            return

        request = TranslateRequest()
        request.request_meta.SessionID = self.session_id
        request.event = Type.FinishSession
        await self.ws.send(request.SerializeToString())
        self.finish_sent = True
        print("已发送 AST FinishSession，等待服务端完成处理...")

    async def _receive_message(self) -> AstResponse:
        message = await self.ws.recv()
        if isinstance(message, str):
            raise RuntimeError(f"AST returned a text frame: {message}")

        response = TranslateResponse()
        response.ParseFromString(message)

        response_text = response.text
        if response.event == Type.UsageResponse:
            response_text = json.dumps(MessageToDict(response), ensure_ascii=False)

        return AstResponse(
            event=response.event,
            session_id=response.response_meta.SessionID,
            sequence=response.response_meta.Sequence,
            text=response_text,
            data=response.data,
            start_time=response.start_time,
            end_time=response.end_time,
            spk_chg=response.spk_chg,
            message=response.response_meta.Message,
        )

    def _audio_player_task(self):
        stream = self.pyaudio_instance.open(
            format=OUTPUT_FORMAT,
            channels=OUTPUT_CHANNELS,
            rate=OUTPUT_RATE,
            output=True,
            frames_per_buffer=OUTPUT_FRAMES_PER_BUFFER,
        )
        try:
            while self.is_connected or not self.audio_playback_queue.empty():
                try:
                    audio_chunk = self.audio_playback_queue.get(timeout=0.1)
                    if audio_chunk is None:
                        break
                    stream.write(audio_chunk)
                    self.audio_playback_queue.task_done()
                except queue.Empty:
                    continue
        finally:
            stream.stop_stream()
            stream.close()

    def start_audio_player(self):
        if not self.audio_enabled:
            return
        if self.audio_player_thread is None or not self.audio_player_thread.is_alive():
            self.audio_player_thread = threading.Thread(target=self._audio_player_task, daemon=True)
            self.audio_player_thread.start()

    async def handle_server_messages(
        self,
        on_translation_text: Callable[[str, bool], None],
        on_source_text: Optional[Callable[[str, bool], None]] = None,
    ):
        try:
            while self.is_connected and self.ws:
                response = await self._receive_message()
                event = response.event

                if event in (Type.SessionFailed, Type.SessionCanceled):
                    self.is_connected = False
                    raise RuntimeError(
                        f"AST session ended with { _event_name(event) }: "
                        f"{response.message or response.text}"
                    )

                if event == Type.SessionFinished:
                    print("[INFO] AST 会话已结束。")
                    self.session_finished_event.set()
                    self.is_connected = False
                    break

                if event in (Type.TTSResponse, Type.TTSSentenceEnd) and response.data and self.audio_enabled:
                    self.audio_playback_queue.put(response.data)
                    continue

                if response.text:
                    if event == Type.SourceSubtitleResponse and on_source_text:
                        on_source_text(response.text, False)
                    elif event == Type.SourceSubtitleEnd and on_source_text:
                        on_source_text(response.text, True)
                    elif event == Type.TranslationSubtitleResponse:
                        on_translation_text(response.text, False)
                    elif event == Type.TranslationSubtitleEnd:
                        on_translation_text(response.text, True)
                    elif event == Type.UsageResponse:
                        print(f"\n[INFO] AST 用量: {response.text}")

        except websockets.exceptions.ConnectionClosed as exc:
            print(f"[WARNING] AST 连接已关闭: {exc}")
            self.is_connected = False
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            print(f"[ERROR] AST 消息处理异常: {exc}")
            traceback.print_exc()
            self.is_connected = False

    async def start_microphone_streaming(self):
        stream = self.pyaudio_instance.open(
            format=INPUT_FORMAT,
            channels=INPUT_CHANNELS,
            rate=INPUT_RATE,
            input=True,
            frames_per_buffer=INPUT_FRAMES_PER_BUFFER,
        )
        print("麦克风已启动，请开始说话...")
        try:
            while self.is_connected:
                audio_chunk = await asyncio.get_running_loop().run_in_executor(
                    None,
                    stream.read,
                    INPUT_FRAMES_PER_BUFFER,
                    False,
                )
                await self._send_audio_chunk(audio_chunk)
        except asyncio.CancelledError:
            raise
        finally:
            stream.stop_stream()
            stream.close()

    async def close(self):
        if self.is_connected and self.ws:
            await self._send_finish_session()
            try:
                await asyncio.wait_for(self.session_finished_event.wait(), timeout=15)
                print("AST 服务端已完成处理。")
            except asyncio.TimeoutError:
                print("等待 AST SessionFinished 超时。")

        self.is_connected = False
        if self.ws:
            await self.ws.close()
            print("AST WebSocket 连接已关闭。")

        if self.audio_player_thread:
            self.audio_playback_queue.put(None)
            self.audio_player_thread.join(timeout=1)
            print("AST 音频播放线程已停止。")

        self.pyaudio_instance.terminate()
        print("PyAudio 实例已释放。")
