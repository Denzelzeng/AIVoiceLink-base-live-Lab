# C500 语音服务部署记录

## 当前状态（2026-08-03 验证）

- `8004 /health`：HTTP 200。
- ASR 模型：`qwen3-asr`，架构 `Qwen3ASRForConditionalGeneration`。
- ASR 镜像：`vllm-metax:0.19.0-maca3.5.3-audio1`。
- ASR 原生接口：`POST /v1/audio/transcriptions`。
- `8003 /health`：HTTP 200；对外模型 ID 为 `qwen3-omni`。
- 8003 实际 checkpoint 架构为 `Qwen3MoeForCausalLM`，不是多模态
  Qwen3-Omni；当前只用于文字翻译。
- 两个服务均未提供 `/v1/realtime`，客户端采用短音频片段近实时同传。
- TTS 尚未部署，客户端输出流式译文，不播放译音。

已使用 19.44 秒、mono/16-bit/16 kHz PCM WAV 完成端到端验收：

1. 8004 返回干净的中文转写和音频时长。
2. 8003 接收转写并流式返回英文译文。
3. 客户端单元测试 6/6 通过。

## 故障原因与修复

原 MetaX vLLM 0.19.0 镜像缺少其 wheel 声明的 audio extra，服务器日志为：

```text
ModuleNotFoundError: No module named 'soundfile'
ImportError: Please install vllm[audio] for audio support
```

基于原 MetaX 镜像创建了派生镜像，没有替换 vLLM、PyTorch、Transformers、
NumPy 或 Pydantic。新增依赖为：

- `av==18.0.0`
- `resampy==0.4.3`
- `soundfile==0.14.0`
- `soxr==1.1.0`
- 已有 `scipy==1.16.0`
- 已有 `mistral_common==1.11.0`

派生镜像 Dockerfile 保存在服务器：

```text
/home/deploy/audio_project/vllm-metax-audio/Dockerfile
```

完整检查输出保存在：

```text
/home/deploy/audio_project/server-audit-20260803/
```

## 当前容器与回滚

生产容器 `vllm-asr` 使用修复镜像并映射 `8004:8000`。原容器保留为：

```text
vllm-asr-original-20260803
```

临时验收容器保留为停止状态：

```text
vllm-asr-audio-test
```

如果新服务出现回归，执行：

```bash
docker rm -f vllm-asr
docker rename vllm-asr-original-20260803 vllm-asr
docker start vllm-asr
curl -i --max-time 15 http://127.0.0.1:8004/health
```

原容器没有 audio dependencies，所以回滚后只能恢复原有 API 可达状态，不能
处理真实音频。

## 8003 的限制

8003 挂载的宿主机目录为：

```text
/data0/models/Qwen3-30B-A3B-Instruct-2507
```

其 `config.json` 明确显示：

```json
{
  "architectures": ["Qwen3MoeForCausalLM"],
  "model_type": "qwen3_moe"
}
```

因此 `input_audio` / `audio_url` 返回 “not a multimodal model” 是正确行为。
如果以后需要音频直接进入 Qwen3-Omni，必须取得完整多模态 checkpoint，并
使用 MetaX 明确支持该架构的 vLLM/vLLM-Omni 版本；增加启动参数不能把当前
文本 checkpoint 变成 Omni。

## 后续维护

- 建议保留原 ASR 容器 24-72 小时，观察生产稳定性后再决定是否删除。
- 根分区曾达到 87%；停止的历史测试容器约占 53 GB。删除前必须逐项确认不再
  需要，避免运行宽泛的 `docker system prune`。
- 若要求真正全双工 PCM 输入，需要部署支持 streaming input 或
  `/v1/realtime` 的服务。
- 若要求播放译音，需要另行部署 CosyVoice/Qwen TTS。
