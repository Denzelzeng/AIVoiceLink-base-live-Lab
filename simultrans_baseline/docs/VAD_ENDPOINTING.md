# VAD、语义端点与打断

## 职责分工

```text
AEC/NS -> energy gate -> 450 ms silence candidate
                     -> remote semantic LLM COMPLETE/INCOMPLETE
                     -> 1.8 s hard timeout
```

- 能量 VAD 只判断 speech/non-speech，不做语义推理；
- 云端语义模型判断候选停顿是否结束完整语义；
- LCP 决定句中哪些文字可以稳定提交；
- hard timeout 防止连续 incomplete 导致永不结束；
- 输出播放默认与输入独立，不决定翻译边界，也不受新讲话影响。

## 为什么 VAD 留在本地

`EnergyTurnSegmenter` 是 RMS 门限、预卷和静音计时器，不是模型。它需要在每个 50 ms
音频帧上运行，把这些帧全部发给云模型既增加网络成本，也会让打断依赖往返延迟。因此
“所有模型走 API”与“本地无模型 VAD”并不冲突。

默认语义端点只向 Qwen 文本 API 提交当前 ASR 转写。若返回 `INCOMPLETE`，下一声学段并入
同一逻辑话轮。远端 API 故障时，标点和连接词 heuristic 只作为临时安全回退，并在事件
中明确标记。

## 参数起点

| 参数 | API baseline | 影响 |
|---|---:|---|
| capture sample rate | 24 kHz | 同时满足 Qwen 声音注册最低采样率 |
| frame_ms | 50 ms | 越小越灵敏，调度开销越高 |
| VAD candidate silence | 450 ms | 越短越易误切，越长句末延迟越大 |
| ASR partial interval | 1200 ms | 越短累计 API 调用与费用越高 |
| semantic API timeout | 4 s | 单次网络上限；正常目标应远低于此值 |
| semantic hard timeout | 1800 ms | 连续 incomplete 后的本地强制结束 |
| LCP agreement depth | 2 | 越大越稳，确认更慢 |

首轮应联合扫描 VAD 静音、ASR partial 间隔、LCP depth 和语义 API 延迟。只把 VAD 调短，
往往会增加累计 ASR/MT 请求、回改和费用。

## 独立播放与可选 Barge-in

默认 `barge_in_enabled = false`。检测到新 speech 时只打开新的源语音段；TTS 使用独立的
FIFO 队列，当前音频继续播放，后续译音继续排队，不等待 VAD 静音，也不改变 playback
epoch。这适用于用户要求的“边说边播、每段播完”模式。

只有显式打开可选 barge-in 时，才执行旧策略：

1. `playback_epoch += 1`；
2. sink 中止当前设备播放；
3. TTS 队列中旧 epoch 短语被丢弃；
4. 已在网络返回途中的旧 TTS chunk 不再写入设备；
5. 已经播放到空气中的音频不可回滚。

此路径不等待远端模型，所以打断延迟主要由 frame/VAD 与音频设备决定。
PyAudio 不提供 `abort_stream()`；实现使用受支持的 `stop_stream()`，并把 PCM 切成 20 ms
播放块。打断先使当前 generation 失效，再等待最多一个块并关闭 stream。

## 回声与故障策略

- 扬声器播放时应在 VAD 前接 AEC/NS，当前代码不内置 AEC；
- 语义 API 失败：当前候选回退 heuristic，并保留 hard timeout；
- ASR 空文本：不触发 MT/TTS；
- TTS 队列增长：独立 FIFO 不丢段；若源语速长期高于合成/播放速度，播放延迟会自然累积；
- 重叠讲话：不用于声音注册；多人模式需 diarization 和 speaker/profile 映射；
- 无 AEC 时优先使用耳机，避免译音触发自己的 VAD。
- 麦克风校准期间必须保持安静；CLI 只在校准回调完成后提示“现在可以开始讲话”。
