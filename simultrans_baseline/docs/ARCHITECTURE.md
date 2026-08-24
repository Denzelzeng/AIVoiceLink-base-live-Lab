# 架构与状态边界

## 1. 当前部署边界

当前版本只有一个轻量编排进程。所有模型推理均位于远端 API，进程内不导入 Torch、
Transformers、ONNX、ModelScope、CosyVoice 或 Smart Turn：

| 模块 | 输入 | 输出 | 默认远端 API |
|---|---|---|---|
| Audio/VAD | 24 kHz PCM frame | partial / acoustic final | 本地能量门限，无模型 |
| ASR | 当前声学段累计 WAV | 完整识别假设 | Qwen3-ASR |
| Source committer | 连续完整假设 | committed / unstable | 本地 LocalAgreement/LCP |
| Semantic endpoint | 当前转写 | complete / incomplete | Qwen 文本模型 |
| MT | 当前逻辑话轮源文 | 完整目标假设 | Qwen-MT |
| Target committer | 连续完整目标假设 | committed / speculative | 本地严格 LCP |
| Voice enrollment | 3–4 秒 24 kHz WAV + 转写 | 云端 voice ID | Qwen voice enrollment |
| Clone TTS | committed target + voice ID | 24 kHz PCM chunks | Qwen3-TTS-VC |

本地代码只做音频 I/O、无模型 VAD、状态机、稳定提交、背压、播放和指标。这些边界都由
Python `Protocol` 隔离，模型 API 之间没有端到端 LiveTranslate 调用。

## 2. 三条并行路径

```text
                  ┌─ ASR API ─ source LCP ─ MT API ─ target LCP ─ confirmed subtitle
microphone ─ VAD ─┤                                           └─ phrase buffer ─ TTS API
                  ├─ semantic endpoint API (silence candidate only)
                  └─ 3–4 s enrollment API ─ cloud voice ID ──────────────────────┘
```

声音注册与 ASR/翻译并行。首次克隆译音的等待近似为：

```text
max(云端音色就绪, 首个可朗读 committed_target) + TTS 首包
```

而不是把声音注册、ASR、MT 和 TTS 延迟机械串行相加。

## 3. 双重提交

```text
ASR: committed_source + unstable_source
MT:  committed_target + speculative_target
```

- 连续两次完整假设的最长公共前缀进入 committed；
- 逻辑端点 final 时剩余假设全部提交；
- 已提交内容不会因后续 API 结果冲突而删除；
- TTS 只消费 `committed_target`，推测尾部只显示不播放。

Qwen-MT API 要求单条 user message 和 `translation_options`。因此 provider 每次对完整当前
源前缀重译，编排层负责稳定边界；不会把普通 LLM prompt 冒充 Qwen-MT 参数。

## 4. 声学段与逻辑话轮

VAD 的一次静音只产生声学段。远端语义端点若判定 `INCOMPLETE`，下一声学段仍拼入同一
逻辑话轮，再对完整逻辑源前缀重译。默认语义模型只读取 ASR 文字，不再次上传音频。

API 失败时会用保守的标点/连接词规则暂时兜底；它是确定性算法，不是本地模型。
`semantic_hard_timeout_ms` 强制结束连续 incomplete，避免无限等待。

## 5. 人声复刻生命周期

默认从同一说话人的累计音频中取 3–4 秒，注册任务只启动一次，后续 TTS 复用返回的
voice ID。云端 Qwen-TTS 要求至少 3 秒连续清晰语音且采样率至少 24 kHz，所以默认采集即
为 24 kHz，不使用伪造的 2 秒注册。

正常关闭时调用声音删除 API。远端音色不是本地内存对象；异常退出或断网可能导致清理
失败，生产环境必须增加服务端 session registry、TTL 清扫和审计。

## 6. 打断、背压和回声

检测到新 speech 时立即增加 `playback_epoch`：旧 epoch 的排队 TTS、正在流回的音频块和
设备播放都会取消。这是应用层 barge-in。

它不替代 AEC。扬声器和麦克风共处时，PCM 进入 VAD 前应接 WebRTC AEC/NS/AGC、硬件回声
消除或使用耳机，否则译音可能被当成用户新讲话。队列固定容量并产生反压，避免延迟无限
增长。

## 7. 关键事件

| 事件 | 关键字段 |
|---|---|
| `transcript.update` | committed, unstable, is_final, acoustic_segment_id |
| `endpoint.decision` | complete, probability, reason |
| `translation.update` | committed, speculative, is_final |
| `voice.enrollment_started` | reference_ms |
| `voice.ready` / `voice.failed` | profile_id, reference_ms / error |
| `tts.started/finished/cancelled` | segment_id, text / reason |
| `turn.metrics` | first_source_commit_ms, first_target_commit_ms, first_audio_ms |

事件可写入 JSONL，供 UI、WebSocket 网关、回放和验收复用。
