# 本地 VAD 与说话人变化检测

## 结论

本 baseline 采用两条彼此独立的本地声学支路：

```text
24 kHz PCM ─┬─→ 16 kHz Silero VAD ─→ speech/non-speech ─→ 现有语义端点
            └─→ 16 kHz CAM++ embedding ─→ 说话人变化门控 ─→ 云端音色注册
```

VAD 默认在 CPU 上运行；CAM++ 默认也使用 CPU，因为每 2–4 秒清晰人声才运行一次，
不会成为主链路瓶颈。`speaker_change.inference_provider` 可切到 `cuda`，但应先确认
所安装的 sherpa-onnx wheel 带 CUDA provider。RTX 4060 更值得留给后续本地 ASR/MT。

## VAD 方案评估

| 方案 | 优点 | 局限 | 本项目判断 |
|---|---|---|---|
| 能量阈值 | 零依赖、行为可解释 | 风扇、键盘、音乐会误触发，远场和轻声易漏检 | 保留为回退与测试后端 |
| WebRTC VAD | 极轻、成熟 | 二值规则较硬，对复杂噪声和音乐适应有限 | 不作为默认 |
| Silero VAD | 多语言、约 2 MB、CPU 单块低于 1 ms、MIT | 只判断声学活动，不理解句意 | **默认** |
| TEN VAD | 官方测试中比 Silero 更早检测停顿、体积更小 | 阈值仍需业务数据校准，Python 原生支持历史上偏 Linux | 已通过 sherpa-onnx 预留可切后端 |

Silero 官方说明支持 8/16 kHz、30 ms 以上块单 CPU 线程低于 1 ms：
<https://github.com/snakers4/silero-vad>。TEN VAD 官方仓库给出与 WebRTC/Silero
的可复现实验，并明确阈值需按业务数据调优：
<https://github.com/TEN-framework/ten-vad>。

当前输入仍保持 24 kHz，VAD 旁路在内存中重采样到 16 kHz。启动时的 1 秒环境
校准继续保留，用作噪声基线与 energy 回退；神经 VAD 不再把单一 RMS 值当成人声。
VAD 只触发候选声学边界，现有 remote LLM 语义端点仍决定语义是否完整。

默认值：`threshold=0.5`、模型内最短语音 150 ms、模型内最短静音 100 ms；外层
再用 `audio.end_silence_ms=300` 消抖，有效候选停顿约 400 ms。应使用真实会议录音扫阈值，而不是把默认值
当作生产标定。

## 说话人方案评估

附件中的 AnalyticDB 方案适合 1:N 声纹库、离线检索与审计，但接口以音频 URL/OSS
为中心。对每个实时话轮上传再查询会引入网络延迟和可用性依赖，因此不作为实时
注册门控。

候选方案：

| 方案 | 特点 | 本项目判断 |
|---|---|---|
| SpeechBrain ECAPA-TDNN | API 简单、生态成熟 | 需要完整 PyTorch/SpeechBrain，模型主要基于 VoxCeleb |
| NVIDIA TitaNet | NVIDIA 生态完整、适合验证/分离 | NeMo 依赖较重，当前只需轻量变化检测 |
| 3D-Speaker CAM++ | 7.2M 参数，中文/英文 common 模型，Apache-2.0 | **默认** |
| 完整 pyannote/3D-Speaker diarization | 能处理整段多人聚类和重叠检测 | 对“是否换人后再注册”过重，且在线状态更复杂 |

3D-Speaker 官方提供 CAM++、ERes2Net、ECAPA 及分离 recipe：
<https://github.com/modelscope/3D-Speaker>。本项目通过 sherpa-onnx 加载其 CAM++
ONNX 模型；sherpa-onnx 同时提供 VAD、embedding extractor 和 Windows/Python 运行时：
<https://k2-fsa.github.io/sherpa/onnx/speaker-identification/index.html>。

## 变化判定逻辑

1. 首次累计到克隆所需的 3 秒语音时，云端注册与本地 embedding 提取并行执行。
2. 后续每个满足时长的逻辑话轮只提取一次 embedding。
3. 与当前说话人参考质心做 cosine similarity：
   - `>= 0.72`：同一说话人；不注册，参考质心以 0.10 EMA 缓慢适应。
   - `<= 0.55`：确认候选变化；默认一次高置信窗口即可切换并注册。
   - `(0.55, 0.72)`：不确定；保持当前音色，不注册。
4. 可把 `confirmation_windows` 调为 2，以降低嘈杂多人会议中的误切；代价是新说话人
   至少需要两个合格话轮才切换。
5. 本地比较失败时 fail-closed：继续使用当前音色，不会回退成“每句话都注册”。

这些阈值是工程初值，不是跨麦克风通用的生物识别阈值。上线前至少采集同人跨距离、
异人同设备、扬声器回声、音乐、重叠说话五类数据，分别统计同人/异人分数分布，再按
误注册成本选择阈值。声纹比较只用于会话内音色路由，不能作为身份认证。

## 安装与切换

首次执行：

```powershell
& .\scripts\setup-local-audio.ps1
```

脚本会把运行时装入项目私有 `.vendor`，并下载经过 SHA-256 校验的两份模型到
`models/`。正常运行仍使用 `scripts/run.ps1`。

切回无模型模式：

```toml
[vad]
provider = "energy"

[speaker_change]
enabled = false
```

评估 TEN VAD 时，只需下载官方 `ten-vad.onnx`，把 `vad.provider` 改为
`sherpa_ten` 并更新 `model_path`；其余切分、ASR 和语义端点接口不变。
