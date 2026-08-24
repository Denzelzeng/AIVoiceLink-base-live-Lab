# 模块分立实时同声传译 baseline（纯 API 版）

本版本的所有模型能力都通过远端 HTTPS API 调用，运行进程不会加载、下载或推理任何
本地模型：

```text
24 kHz PCM -> 本地无模型 VAD -> Qwen3-ASR API -> 稳定提交
             -> Qwen-MT API -> 确认字幕 -> Qwen 克隆 TTS API -> 24 kHz PCM
                                  \-> Qwen 文本模型 API 语义端点
源语音 ---------------------> 3–4 秒 Qwen 声音注册 API --------/
```

`main.py` 和 `livetranslate_client.py` 只用于参考麦克风采集、语言选择、事件处理和播放
体验。系统不会调用 `qwen3.5-livetranslate-*`、AST 或任何端到端 LiveTranslate 模型。

## 默认 API 模块

| 模块 | 默认远端模型/API | 本地职责 |
|---|---|---|
| ASR | `qwen3-asr-flash-2026-02-10` | 累计窗口、LCP 稳定提交 |
| 翻译 | `qwen-mt-flash` | 双重提交、上下文边界、TTS 短语缓冲 |
| 语义端点 | `qwen-flash` | VAD 候选停顿、hard timeout |
| 声音注册 | `qwen-voice-enrollment` | 授权、按话轮持续刷新、会话清理 |
| 克隆 TTS | `qwen3-tts-vc-2026-01-22` | 独立 FIFO 播放队列、音频 sink |

能量 VAD、最长公共前缀、队列和音频播放是普通信号处理/编排代码，不是本地模型。
`configs/mock.toml` 仅用于无网络的确定性状态机测试，也不包含模型推理。

## 已实现

- CLI 选择输入/输出语言，支持麦克风或 24 kHz mono PCM16 WAV；
- 实时显示 `committed + unstable` 识别文字和
  `committed + speculative` 翻译文字；
- Qwen3-ASR 累计窗口 API 重识别，LCP 边界绝不回滚；
- Qwen-MT 专用 `translation_options`，只把确认译文送入 TTS；
- 首个 3 秒参考自动注册；后续每个达到 3 秒的新话轮异步刷新远端音色；
- 云端 LLM 语义完整性判断、1.8 秒 hard timeout；
- 默认输入与输出链路独立：持续讲话不暂停、不取消已排队译音；
- 云 ASR 落后于采集时合并过时累计 partial，不逐个偿还历史请求；
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

## 运行

先检查远端目录和声音管理 API；此命令不会创建音色或执行 TTS：

```powershell
& '.\simultrans_baseline\scripts\run.ps1' doctor --voice-consent
```

仅文字同传：

```powershell
& '.\simultrans_baseline\scripts\run.ps1' run `
  --source-language Chinese --target-language English --text-only
```

启用人声复刻和译音：

```powershell
& '.\simultrans_baseline\scripts\run.ps1' run `
  --source-language Chinese --target-language English `
  --audio-output --voice-consent `
  --events '.\simultrans_baseline\output\session.jsonl'
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
每个达到 3 秒的新话轮都会创建新的会话音色，后续对应译音使用该版本；所有会话音色会在
正常退出时逐一删除。扬声器与麦克风同时工作时应使用耳机，否则没有 AEC 的 baseline 可能
把译音回声再次识别为输入。

WAV 回放必须与配置一致，即默认 24 kHz、单声道、PCM16：

```powershell
& '.\simultrans_baseline\scripts\run.ps1' run `
  --wav '.\sample-24k-mono.wav' --text-only --no-realtime
```

离线控制流演示和测试：

```powershell
& '.\simultrans_baseline\scripts\run.ps1' demo --audio-output
& '.\simultrans_baseline\scripts\test.ps1'
```

## 目录

```text
configs/                 纯云 API 与 mock 配置
docs/                    架构、API 契约、VAD/打断、验收和迁移边界
src/simultrans_baseline/ 编排、稳定提交、音频、远端 API providers
tests/                   不调用真实 API、不依赖 GPU 的 contract tests
scripts/                 Windows 启动、测试与指标汇总
```

当前版本没有 `services/`，也没有本地模型启动脚本。未来切换 L20 时只替换 providers，
迁移边界见 [FUTURE_L20_MIGRATION.md](docs/FUTURE_L20_MIGRATION.md)，不属于本版本运行路径。

## 文档

- [架构和状态边界](docs/ARCHITECTURE.md)
- [远端 API 契约](docs/API_CONTRACTS.md)
- [VAD、语义端点和打断](docs/VAD_ENDPOINTING.md)
- [验收指标](docs/ACCEPTANCE.md)
- [交付说明和限制](docs/DELIVERY.md)
- [未来 L20 迁移边界](docs/FUTURE_L20_MIGRATION.md)
