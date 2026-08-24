from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import TextIO

from .events import PipelineEvent


class ConsoleRenderer:
    """Compact console UI: live hypotheses rewrite one line, finals print once."""

    def __init__(self, stream: TextIO | None = None):
        self._stream = stream or sys.stdout
        self._interactive = bool(getattr(self._stream, "isatty", lambda: False)())
        self._live_line = False
        self._source: dict[int, str] = {}
        self._target: dict[int, str] = {}
        self._source_final: set[int] = set()
        self._target_final: set[int] = set()
        self._rendered: set[int] = set()

    def __call__(self, event: PipelineEvent) -> None:
        turn = f" #{event.turn_id}" if event.turn_id is not None else ""
        if event.kind == "session.started":
            rate = event.data.get("tts_speech_rate", 1.0)
            rate_text = f"；译音语速={rate:g}x" if event.data["audio_output"] else ""
            self._print(
                f"会话已启动：{event.data['source_language']} -> "
                f"{event.data['target_language']}；"
                f"输出={'文字+克隆语音' if event.data['audio_output'] else '仅文字'}"
                f"{rate_text}"
            )
        elif event.kind == "transcript.update":
            committed = event.data.get("committed", "")
            unstable = event.data.get("unstable", "")
            assert event.turn_id is not None
            self._source[event.turn_id] = f"{committed}{unstable}"
            if event.data.get("is_final"):
                self._source_final.add(event.turn_id)
            self._render_turn(event.turn_id)
        elif event.kind == "translation.update":
            committed = event.data.get("committed", "")
            speculative = event.data.get("speculative", "")
            assert event.turn_id is not None
            self._target[event.turn_id] = f"{committed}{speculative}"
            if event.data.get("is_final"):
                self._target_final.add(event.turn_id)
            self._render_turn(event.turn_id)
        elif event.kind == "endpoint.decision":
            # Normal endpoint decisions remain in JSONL but do not add one
            # console line per hypothesis/turn.
            return
        elif event.kind == "endpoint.hard_timeout":
            self._print(f"[端点{turn}] 达到硬超时，强制提交")
        elif event.kind == "voice.enrollment_started":
            action = "刷新" if event.data.get("refresh") else "注册"
            self._print(
                f"[声纹] 正在用 {event.data.get('reference_ms')} ms "
                f"参考语音{action}"
            )
        elif event.kind == "voice.ready":
            state = "已切换到最新参考音色" if event.data.get("refresh") else "已就绪"
            self._print(f"[声纹] {state}：{event.data.get('profile_id')}")
        elif event.kind == "speaker.decision" and event.data.get("changed"):
            score = event.data.get("similarity")
            self._print(
                f"[说话人] 检测到切换（相似度 {score}）；"
                "参考语音达标后注册新音色"
            )
        elif event.kind == "speaker.failed":
            self._print(
                f"[说话人检测警告] {event.data.get('error')}；保持当前音色"
            )
        elif event.kind == "tts.waiting_for_voice":
            seconds = float(event.data.get("timeout_ms", 0)) / 1000
            self._print(f"[克隆语音{turn}] 等待首次声纹（最多 {seconds:g} 秒）")
        elif event.kind == "voice.failed":
            self._print(f"[声纹错误] {event.data.get('error')}")
        elif event.kind == "voice.prepare_failed":
            self._print(
                f"[声纹预热警告] {event.data.get('error')}；"
                "将在实际合成时自动重试"
            )
        elif event.kind == "voice.deleted":
            self._print(f"[声纹] 云端会话音色已删除：{event.data.get('profile_id')}")
        elif event.kind == "voice.delete_failed":
            self._print(
                f"[声纹清理警告] 云端音色删除失败：{event.data.get('error')}；"
                "请稍后通过声音管理 API 清理"
            )
        elif event.kind == "tts.started":
            self._print(f"[语音{turn}] 开始播放")
        elif event.kind == "tts.skipped":
            self._print(f"[语音跳过{turn}] {event.data.get('reason')}")
        elif event.kind == "tts.cancelled":
            self._print(f"[语音取消{turn}] {event.data.get('reason')}")
        elif event.kind == "tts.failed":
            self._print(f"[克隆语音错误{turn}] {event.data.get('error')}；文字同传继续")
        elif event.kind == "audio.interrupt_failed":
            self._print(f"[播放打断警告] {event.data.get('error')}；识别与翻译继续")
        elif event.kind == "audio.speech_started" and event.data.get("barge_in"):
            self._print("[打断] 检测到新讲话，已取消旧的待播放译音")
        elif event.kind == "turn.metrics":
            self._print(
                f"[延迟{turn}] 句长={event.data.get('utterance_ms')} ms；"
                f"ASR排队={event.data.get('final_asr_queue_ms')} ms；"
                f"ASR请求={event.data.get('final_asr_request_ms')} ms；"
                f"端点={event.data.get('endpoint_request_ms')} ms；"
                f"翻译排队={event.data.get('final_mt_queue_ms')} ms；"
                f"翻译请求={event.data.get('final_mt_request_ms')} ms；"
                f"合并=ASR {event.data.get('asr_updates_coalesced') or 0:g}/"
                f"MT {event.data.get('mt_updates_coalesced') or 0:g}；"
                f"抢占=ASR {event.data.get('asr_partials_cancelled') or 0:g}/"
                f"MT {event.data.get('mt_partials_cancelled') or 0:g}；"
                f"译文={event.data.get('first_target_commit_ms')} ms；"
                f"云首音={event.data.get('first_cloud_audio_ms')} ms；"
                f"播放排队={event.data.get('playback_queue_ms')} ms；"
                f"首播放={event.data.get('first_audio_ms')} ms"
            )
        elif event.kind == "pipeline.error":
            self._print(f"[流水线错误] {event.data.get('error')}")
        elif event.kind == "session.finished":
            self._print("会话已结束。")

    def _render_turn(self, turn_id: int) -> None:
        source = self._source.get(turn_id, "")
        target = self._target.get(turn_id, "")
        if (
            turn_id in self._source_final
            and turn_id in self._target_final
            and turn_id not in self._rendered
        ):
            self._rendered.add(turn_id)
            self._print(f"[同传 #{turn_id}] {source}  →  {target}")
            return
        if self._interactive:
            target_text = f"  →  {target}" if target else ""
            self._write_live(f"[同传 #{turn_id} · 实时] {source}{target_text}")

    def _write_live(self, text: str) -> None:
        self._stream.write(f"\r\x1b[2K{text}")
        self._stream.flush()
        self._live_line = True

    def _clear_live(self) -> None:
        if self._live_line:
            self._stream.write("\r\x1b[2K")
            self._stream.flush()
            self._live_line = False

    def _print(self, text: str) -> None:
        self._clear_live()
        print(text, file=self._stream, flush=True)


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
