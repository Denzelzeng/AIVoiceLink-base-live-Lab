# 交付说明

## 本轮变更

当前交付已改为纯远端 API 版本：

- 移除 C500/L20 本机端口运行配置；
- 移除本地 CosyVoice 与 Smart Turn sidecar 及其依赖；
- 默认 ASR 改为 Qwen3-ASR Data URI API；
- 翻译改为 Qwen-MT 专用 `translation_options`；
- 语义完整性判断改为 Qwen 文本模型 API；
- 人声复刻改为 Qwen Voice Enrollment API；
- 译音改为 Qwen3-TTS-VC Realtime WebSocket PCM API，启动时后台预连接，同一音色复用连接；
  音色刷新时并行预建第二个 session、就绪后无缝切换，首个音频块前的连接故障自动重试一次；
  commit 模式默认 1.3x 语速，云端合成与本地 FIFO 播放使用独立 worker，并对每段 PCM 做
  400 ms 抖动预缓冲；
- 配置强制远端 HTTPS，拒绝 loopback 模型地址。

实机麦克风反馈后的修订包括：PyAudio 兼容打断、交互式语言菜单、校准完成提示、首段译文
等待云端声纹、输入与输出默认使用独立链路、译音 FIFO 播放且不再等待讲话结束、按满足
三秒条件的新话轮持续刷新声纹但不阻塞现有 TTS 队列、换声时重建 Realtime TTS session、
云 ASR partial 合并、控制台 partial 单行原位刷新，以及 TTS 故障不再拖垮文字同传。

2026-08-21 使用现有 workspace 凭据执行了不创建音色、不做推理的 doctor：ASR
`qwen3-asr-flash-2026-02-10`、MT `qwen-mt-flash`、语义模型 `qwen-flash` 均在远端模型
目录可用，声音管理 API 也可达。该结果只证明鉴权、目录和管理接口，不等于真实音频质量
或端到端延迟验收。

`configs/mock.toml` 只验证状态机，不是本地模型部署。当前项目没有 `services/` 目录、模型
权重、GPU 运行时或本地推理启动脚本。

## 参考映射

- `main.py` / `livetranslate_client.py`：参考麦克风、实时事件、语言选择和播放体验；
- 两份 `local/` 报告：采用级联模块、双重提交、TTS 只读确认文本、语义端点和统一延迟；
- C500 报告仅保留为模块边界和未来迁移背景，不再作为当前运行后端；
- 百炼官方契约用于 ASR、Qwen-MT、Voice Enrollment 和 Qwen3-TTS-VC 的实际 HTTP 结构。

明确排除：

- 不调用 `qwen3.5-livetranslate-*`；
- 不使用 AST 主链路；
- 不把 token SSE 误称为同时翻译；
- 不把累计窗口 ASR 误称为原生 WebSocket 流式；
- 不把 3 秒参考音频误称为 3 秒端到端首译音 SLA。

## 凭据

加载器读取父目录 `.env`，并兼容 `Default Workspace-apiKey-*.csv` 转置格式；只把
`apiKey/workspaceId/openAiCompatible` 放入当前进程环境，不复制、不打印秘密。native
API origin 从 compatible URL 的 scheme/host 推导，也可显式设置 `WORKSPACE_API_BASE_URL`。

## 已知限制

1. ASR 每个 partial 重发当前累计 WAV，实时性和费用不如原生 WebSocket；
2. Qwen-MT 的标准接口只接收单条 user message，跨轮上下文主要由完整当前源前缀和本地
   稳定状态承担；
3. Qwen-TTS 声音注册要求至少 3 秒连续清晰人声、采样率至少 24 kHz；
4. 云端 voice ID 正常关闭时删除，崩溃时可能残留，生产必须有补偿清理；
5. 能量 VAD 是零模型 baseline；噪声环境需更强的 DSP/AEC 或远端 VAD 服务；
6. 文本语义端点会增加一次 API 往返；失败时 heuristic 只保证可用性，不保证同等质量；
7. 本地 CAM++ 明确判断换人且累计到 3 秒参考后才刷新音色；不足 3 秒的换人无法单独满足注册 API 硬门槛，严格
   多人归属仍需远端 diarization 和 `speaker_id -> voice_id`；
8. UI 是控制台/JSONL，WebSocket 网关可复用相同事件结构。

## 安全与隐私

- 声纹属于敏感生物特征，必须获得说话人明确授权；
- 参考音频会发送到云端，不能再宣称“只在本机处理”；
- 禁止把克隆音色用于身份认证、冒充或未经授权的内容；
- 日志不记录 API key，默认不落盘参考音频；
- 对音色 create/delete、操作者、会话 ID 和失败补偿建立审计；
- 多人场景不得用重叠语音注册，也不得错误绑定说话人。
