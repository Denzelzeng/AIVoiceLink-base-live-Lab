# 模块分立实时同声传译：云 API 流水线

目录名为 `cloud_api_pipeline`；Python 发行包和 CLI 仍保留 `simultrans_baseline`，以兼容现有脚本与导入。

本版本的所有模型能力都通过远端 HTTPS API 调用，运行进程不会加载、下载或推理任何
本地模型：

```text
24 kHz PCM -> 本地 Silero VAD -> Qwen3-ASR API -> 稳定提交
             -> Qwen-MT API -> 确认字幕 -> Qwen 克隆 TTS API -> 24 kHz PCM
                                  \-> Qwen 文本模型 API 语义端点
源语音 -> 本地 CAM++ 说话人变化门控 -> 3–4 秒 Qwen 声音注册 API --/
```

`main.py` 和 `livetranslate_client.py` 只用于参考麦克风采集、语言选择、事件处理和播放
体验。系统不会调用 `qwen3.5-livetranslate-*`、AST 或任何端到端 LiveTranslate 模型。

## 默认 API 模块

| 模块 | 默认远端模型/API | 本地职责 |
|---|---|---|
| ASR | `qwen3-asr-flash-2026-02-10` | 累计窗口、LCP 稳定提交 |
| 翻译 | `qwen-mt-flash` | 双重提交、上下文边界、TTS 短语缓冲 |
| 语义端点 | `qwen-flash` | VAD 候选停顿、hard timeout |
| 声音注册 | `qwen-voice-enrollment` | 授权、仅换人时刷新、会话清理 |
| 克隆 TTS | `qwen3-tts-vc-realtime-2026-01-15` | 1.3x 流式译音、独立 FIFO、音频 sink |

Silero VAD 与 CAM++ 说话人 embedding 在本地通过 ONNX 运行；ASR、MT、语义端点、
声音注册和 TTS 仍全部调用分立 API。`configs/mock.toml` 用能量 VAD 和假后端做无网络测试。

## 已实现

- CLI 选择输入/输出语言，支持麦克风或 24 kHz mono PCM16 WAV；
- 实时显示 `committed + unstable` 识别文字和
  `committed + speculative` 翻译文字；
- Qwen3-ASR 累计窗口 API 重识别，LCP 边界绝不回滚；
- Qwen-MT 专用 `translation_options`，只把确认译文送入 TTS；
- 首个同一说话人的 3 秒参考自动注册；后续仅在本地确认换人时异步刷新远端音色，刷新不阻塞 TTS，
  最近已就绪音色会继续服务当前队列；
- 云端 LLM 语义完整性判断、1.8 秒 hard timeout；
- 默认输入与输出链路独立：持续讲话不暂停、不取消已排队译音；
- 分立的 Qwen3-TTS-VC Realtime WebSocket 启动即预连接；同一音色持续复用连接，音色刷新后
  在第二条 WebSocket 上并行完成新 session 握手和配置，再原子切换；旧连接继续供给 FIFO，
  随后异步关闭，换声不再让播放缓冲断粮；
- Realtime TTS 使用 `commit` 模式和 1.3x 默认语速；若连接在首个音频块前失效，会自动重连
  并重试一次，已有音频时不重试以避免重复播放；
  - 云端 TTS 生成与声卡播放由两个 worker 并行执行：上一句播放时下一句可提前生成，
    播放侧严格 FIFO、一句不丢；每段首批 PCM 默认预缓冲 200 ms，抵抗云端分片抖动；
- 控制台 partial 在同一行原位刷新，每个话轮只落一行最终“原文 → 译文”；普通语义端点事件
  只写 JSONL，不再刷屏，语音开始提示也不重复打印整句译文；
- ASR 明确输出句号、问号或感叹号时直接提交，跳过容易误判且增加延迟的语义 API 往返；
  - 麦克风采集不受云请求反压；ASR 和翻译都只保留每个话轮的最新 partial，final 永不
    丢弃且保持原顺序，不逐个偿还已经过时的累计请求；final 到达时还会抢占同话轮仍在
    运行的 partial 云请求，避免异常慢请求挡住最终结果；
- JSONL 事件、延迟指标、mock 闭环和自动测试；
- 读取父目录 `.env` 或 `Default Workspace-apiKey-*.csv`，不复制、不打印密钥。

## 配置凭据

默认配置为 `configs/cloud_api.toml`。已有的 workspace CSV 会自动提供
`DASHSCOPE_API_KEY`、`WORKSPACE_OPENAI_BASE_URL`，并从兼容地址推导
`WORKSPACE_API_BASE_URL`。也可以把 `.env.example` 复制成父目录或本目录的 `.env` 后填值。

三个值的形态应为：

```dotenv
DASHSCOPE_API_KEY=...
WORKSPACE_OPENAI_BASE_URL=https://WORKSPACE_ID.cn-beijing.maas.aliyuncs.com/compatible-mode/v1
WORKSPACE_API_BASE_URL=https://WORKSPACE_ID.cn-beijing.maas.aliyuncs.com
```

配置校验会拒绝 `localhost`、`127.0.0.1` 和非 HTTPS 模型地址，避免误回到本地模型。

## 环境

需要 Python 3.11+ 和已激活的虚拟环境。各 PowerShell 脚本会使用当前环境的 `python`；如需明确指定解释器，可在启动前设置 `$env:PYTHON` 为 Python 可执行文件路径。

## 运行

首次运行先安装约 19 MB 本地运行时并下载约 29 MB 模型：

```powershell
& '.\cloud_api_pipeline\scripts\setup-local-audio.ps1'
```

先检查远端目录和声音管理 API；此命令不会创建音色或执行 TTS：

```powershell
& '.\cloud_api_pipeline\scripts\run.ps1' doctor --voice-consent
```

仅文字同传：

```powershell
& '.\cloud_api_pipeline\scripts\run.ps1' run `
  --source-language Chinese --target-language English --text-only
```

启用人声复刻和译音：

```powershell
& '.\cloud_api_pipeline\scripts\run.ps1' run `
  --source-language Chinese --target-language English `
  --audio-output --voice-consent `
  --events '.\cloud_api_pipeline\output\session.jsonl'
```

不传 `--source-language/--target-language` 时，交互终端会先显示编号语言菜单。自动化脚本
可显式传入两个语言参数，或加 `--no-language-prompt` 使用 TOML 默认值。

启动麦克风后先保持安静，直到看到“现在可以开始讲话”。否则第一秒人声会进入环境噪声
校准，导致开头被截断或 VAD 阈值过高。

`--voice-consent` 表示说话人已明确同意本会话向云端提交参考音频并创建复刻音色。程序正常
关闭时会调用删除 API；进程崩溃或断网时云端音色可能残留，需要通过声音管理 API 清理。
声音注册本身可能计费，TTS 和其他模型请求也按账号规则计费。

Qwen 声音注册硬性要求至少 3 秒连续清晰人声。首段译文会最多等待 15 秒，让后续语音凑足
参考长度；不会像旧版一样立刻显示 `cloned voice is not ready` 并丢弃。默认配置关闭
barge-in，译音按 FIFO 一直播放完，麦克风中的新讲话只参与 ASR/VAD，不会改变播放队列。
后续达到比较时长的话轮先经过本地 CAM++ 声纹相似度门控；确认换人后才创建新会话音色，
同一说话人不会反复注册。所有会话音色会在正常退出时逐一删除。扬声器与麦克风同时工作时
应使用耳机，否则没有 AEC 的 baseline 可能把译音回声再次识别为输入。

`configs/cloud_api.toml` 中的 `tts.speech_rate = 1.45` 让播放时长比 1.0x 缩短约 31%。允许范围
为 0.5–2.0。Realtime TTS 只替换独立 TTS 模块，不调用 LiveTranslate 模型。

`streaming.tts_prebuffer_ms = 200` 控制每段译音开始播放前的最小云端 PCM 缓冲。值越大越能
抵抗网络抖动，但首音频会等量变慢；它不会跳过或重排句子。云配置把源文本增长门槛设为
8 个字符，并将译音设为 1.45x，以减少高语速输入时的请求放大和音频债务。延迟行会显示
句长、ASR/翻译排队、各 API 请求、过时 partial 合并数量、云首音和播放排队，便于区分
“讲话本身较长”、上游任务积压和严格 FIFO 的正常播放积压。

WAV 回放必须与配置一致，即默认 24 kHz、单声道、PCM16：

```powershell
& '.\cloud_api_pipeline\scripts\run.ps1' run `
  --wav '.\sample-24k-mono.wav' --text-only --no-realtime
```

离线控制流演示和测试：

```powershell
& '.\cloud_api_pipeline\scripts\run.ps1' demo --audio-output
& '.\cloud_api_pipeline\scripts\test.ps1'
```

## 目录

```text
configs/                 纯云 API 与 mock 配置
docs/                    架构、API 契约、VAD/打断、验收和迁移边界
src/simultrans_baseline/ 编排、稳定提交、音频、远端 API providers
tests/                   不调用真实 API、不依赖 GPU 的 contract tests
scripts/                 Windows 启动、测试与指标汇总
```

当前仅 VAD 和说话人变化检测使用轻量本地 ONNX；核心 ASR/MT/TTS 仍是 API。未来切换 L20
时替换对应 providers，迁移边界见 [FUTURE_L20_MIGRATION.md](docs/FUTURE_L20_MIGRATION.md)。

## 文档

- [架构和状态边界](docs/ARCHITECTURE.md)
- [远端 API 契约](docs/API_CONTRACTS.md)
- [VAD、语义端点和打断](docs/VAD_ENDPOINTING.md)
- [本地 VAD 与说话人变化检测评估](docs/LOCAL_VAD_SPEAKER.md)
- [验收指标](docs/ACCEPTANCE.md)
- [交付说明和限制](docs/DELIVERY.md)
- [未来 L20 迁移边界](docs/FUTURE_L20_MIGRATION.md)
