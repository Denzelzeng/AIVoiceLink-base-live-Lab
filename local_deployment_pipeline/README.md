# Qwen3-Omni 本地部署解耦合流水线

目录名为 `local_deployment_pipeline`。它通过 C500 上已部署的 ASR 与翻译服务实现解耦合处理；并不在本机加载 Qwen 权重。

该客户端连接 C500 上的 OpenAI 兼容服务：

- 翻译 API：`http://172.26.63.11:8003/v1`
- 翻译模型 ID：`qwen3-omni`（实测架构为文本模型 `Qwen3MoeForCausalLM`）
- ASR API：`http://172.26.63.11:8004/v1/audio/transcriptions`
- 输入：8004 Qwen3-ASR 转写的麦克风/音频文件，或标准输入文字
- 输出：SSE 流式翻译文本

## 重要边界

交付文档把翻译模型（8003）和 Qwen3-ASR（8004）拆成两个服务，也没有
定义实时音频输入 WebSocket；同时 TTS 暂未部署。因此默认路径是
`麦克风 -> 8004 ASR -> 8003 Qwen3-MoE -> SSE 译文`。客户端通过能量和
静音自动切句。这不是全双工 token-by-token 音频流，也不播放合成语音。

8004 的音频依赖已于 2026-08-03 修复，并通过真实 WAV 的端到端验证；部署和
回滚记录见 `SERVER_FIX.md`。`--stdin` 仍可用于纯文字翻译。

## 安装与运行

在仓库根目录、已激活的 Python 3.11+ 虚拟环境中执行：

```powershell
python -m pip install -r .\local_deployment_pipeline\requirements.txt
```

先检查服务：

```powershell
python .\local_deployment_pipeline\main.py --check
```

纯文字同传模式：

```powershell
python .\local_deployment_pipeline\main.py --stdin --source-language Chinese --target-language English
```

也可以单次验证：

```powershell
python .\local_deployment_pipeline\main.py --text '你好，欢迎参加今天的会议。' --source-language Chinese --target-language English
```

列出麦克风，再启动中文到英文同传：

```powershell
python .\local_deployment_pipeline\main.py --list-devices

python .\local_deployment_pipeline\main.py --source-language Chinese --target-language English
```

翻译一个音频文件：

```powershell
python .\local_deployment_pipeline\main.py --target-language Chinese --file .\official_api_examples\test_audio.wav
```

如果 8003 后续按完整多模态 Qwen3-Omni 重新部署，可跳过 8004：

```powershell
python .\local_deployment_pipeline\main.py --audio-backend direct
```

也可复制 `.env.example` 为仓库根目录的 `.env` 并修改默认配置。本地服务无需
API Key；只有反向代理要求鉴权时才设置 `QWEN_OMNI_API_KEY`。

## 延迟与切句调优

- 环境吵闹或误触发：增大 `--energy-threshold`。
- 句尾等待太久：减小 `--end-silence-ms`（默认 650 ms）。
- 连续发言反馈太慢：减小 `--max-segment-ms`（默认 6000 ms）。
- 指定麦克风：使用 `--device-index N`。

为避免积压后越来越“延迟”，待翻译队列最多保存两段；如果模型处理速度低于
讲话速度，新片段会被明确丢弃并显示警告。

## 测试

```powershell
python -m unittest discover -s .\local_deployment_pipeline\tests -v
```
