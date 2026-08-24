from __future__ import annotations

import asyncio
import json
from pathlib import Path

from .events import PipelineEvent


class ConsoleRenderer:
    def __call__(self, event: PipelineEvent) -> None:
        turn = f" #{event.turn_id}" if event.turn_id is not None else ""
        if event.kind == "session.started":
            print(
                f"会话已启动：{event.data['source_language']} -> "
                f"{event.data['target_language']}；"
                f"输出={'文字+克隆语音' if event.data['audio_output'] else '仅文字'}"
            )
        elif event.kind == "transcript.update":
            marker = "final" if event.data.get("is_final") else "partial"
            committed = event.data.get("committed", "")
            unstable = event.data.get("unstable", "")
            tail = f"  ~{unstable}" if unstable else ""
            print(f"[识别{turn} · {marker}] {committed}{tail}")
        elif event.kind == "translation.update":
            marker = "final" if event.data.get("is_final") else "partial"
            committed = event.data.get("committed", "")
            speculative = event.data.get("speculative", "")
            tail = f"  ~{speculative}" if speculative else ""
            print(f"[翻译{turn} · {marker}] {committed}{tail}")
        elif event.kind == "endpoint.decision":
            state = "完成" if event.data.get("complete") else "继续等待"
            print(f"[语义端点{turn}] {state}（{event.data.get('reason', '')}）")
        elif event.kind == "endpoint.hard_timeout":
            print(f"[语义端点{turn}] 达到硬超时，强制提交")
        elif event.kind == "voice.enrollment_started":
            action = "刷新" if event.data.get("refresh") else "注册"
            print(
                f"[声纹] 正在用 {event.data.get('reference_ms')} ms "
                f"参考语音{action}"
            )
        elif event.kind == "voice.ready":
            state = "已切换到最新参考音色" if event.data.get("refresh") else "已就绪"
            print(f"[声纹] {state}：{event.data.get('profile_id')}")
        elif event.kind == "tts.waiting_for_voice":
            seconds = float(event.data.get("timeout_ms", 0)) / 1000
            print(f"[克隆语音{turn}] 等待声纹就绪（最多 {seconds:g} 秒）")
        elif event.kind == "voice.failed":
            print(f"[声纹错误] {event.data.get('error')}")
        elif event.kind == "voice.deleted":
            print(f"[声纹] 云端会话音色已删除：{event.data.get('profile_id')}")
        elif event.kind == "voice.delete_failed":
            print(
                f"[声纹清理警告] 云端音色删除失败：{event.data.get('error')}；"
                "请稍后通过声音管理 API 清理"
            )
        elif event.kind == "tts.started":
            print(f"[克隆语音{turn}] {event.data.get('text')}")
        elif event.kind == "tts.skipped":
            print(f"[语音跳过{turn}] {event.data.get('reason')}")
        elif event.kind == "tts.cancelled":
            print(f"[语音取消{turn}] {event.data.get('reason')}")
        elif event.kind == "tts.failed":
            print(f"[克隆语音错误{turn}] {event.data.get('error')}；文字同传继续")
        elif event.kind == "audio.interrupt_failed":
            print(f"[播放打断警告] {event.data.get('error')}；识别与翻译继续")
        elif event.kind == "audio.speech_started" and event.data.get("barge_in"):
            print("[打断] 检测到新讲话，已取消旧的待播放译音")
        elif event.kind == "turn.metrics":
            print(
                f"[延迟{turn}] 源确认={event.data.get('first_source_commit_ms')} ms；"
                f"译文确认={event.data.get('first_target_commit_ms')} ms；"
                f"首译音={event.data.get('first_audio_ms')} ms"
            )
        elif event.kind == "pipeline.error":
            print(f"[流水线错误] {event.data.get('error')}")
        elif event.kind == "session.finished":
            print("会话已结束。")


class JsonlRecorder:
    def __init__(self, path: Path):
        self.path = path.resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = self.path.open("a", encoding="utf-8")
        self._lock = asyncio.Lock()

    async def __call__(self, event: PipelineEvent) -> None:
        line = json.dumps(event.as_dict(), ensure_ascii=False)
        async with self._lock:
            self._stream.write(line + "\n")
            self._stream.flush()

    def close(self) -> None:
        self._stream.close()


class EventFanout:
    def __init__(self, *handlers):
        self.handlers = handlers

    async def __call__(self, event: PipelineEvent) -> None:
        for handler in self.handlers:
            result = handler(event)
            if asyncio.iscoroutine(result):
                await result
