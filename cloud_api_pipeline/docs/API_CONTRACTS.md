# 远端模型 API 契约

默认地址来自 workspace CSV 或 `.env`：

- OpenAI-compatible：`WORKSPACE_OPENAI_BASE_URL`；
- DashScope native HTTP：`WORKSPACE_API_BASE_URL`；
- 鉴权：`Authorization: Bearer $DASHSCOPE_API_KEY`。

所有模型地址必须是远端 HTTPS；配置会拒绝 loopback 和非 HTTPS 地址。

## ASR：Qwen3-ASR

每个 partial/final 把当前声学段累计 PCM 包成 WAV，再通过 Data URI 调用：

```http
POST {WORKSPACE_OPENAI_BASE_URL}/chat/completions
Content-Type: application/json

{
  "model": "qwen3-asr-flash-2026-02-10",
  "messages": [{
    "role": "user",
    "content": [{
      "type": "input_audio",
      "input_audio": {"data": "data:audio/wav;base64,..."}
    }]
  }],
  "stream": false,
  "asr_options": {"language": "zh", "enable_itn": true}
}
```

识别文字读取 `choices[0].message.content`。已知语种会映射为 API 代码；`auto` 时不发送
`language`。这是累计窗口 API baseline，不是 WebSocket 原生流式 ASR。

## 翻译：Qwen-MT

```http
POST {WORKSPACE_OPENAI_BASE_URL}/chat/completions
Content-Type: application/json

{
  "model": "qwen-mt-flash",
  "messages": [{"role": "user", "content": "完整当前源文假设"}],
  "translation_options": {
    "source_lang": "Chinese",
    "target_lang": "English",
    "domains": "technical meeting"
  },
  "stream": true
}
```

默认使用增量 SSE 的 `qwen-mt-flash`。provider 也兼容 plus/turbo 的累计 SSE 语义，避免把
每个累计块重复拼接。目标 LCP 层接收完整目标假设并维护不可回滚边界。

## 语义端点：Qwen 文本模型

只有 VAD 产生候选停顿时调用 `chat/completions`。system message 定义二分类规则，user
message 只包含语言和转写，输出必须为 `COMPLETE` 或 `INCOMPLETE`。默认模型为
`qwen-flash`，关闭 thinking，`max_tokens=8`。

API 超时、HTTP 错误或非法标签时，当前候选使用确定性 heuristic，事件 reason 会记录
fallback；hard timeout 仍会结束话轮。

## 声音注册：Qwen Voice Enrollment

调用前必须由 CLI 明确收到 `--voice-consent`。参考音频是 3–4 秒、24 kHz mono PCM16 WAV：

```http
POST {WORKSPACE_API_BASE_URL}/api/v1/services/audio/tts/customization
Content-Type: application/json

{
  "model": "qwen-voice-enrollment",
  "input": {
    "action": "create",
    "target_model": "qwen3-tts-vc-2026-01-22",
    "preferred_name": "simultrans",
    "audio": {"data": "data:audio/wav;base64,..."},
    "text": "与参考音频对应的转写",
    "language": "zh"
  }
}
```

返回 voice ID 位于 `output.voice`。注册模型和后续 TTS 的 target model 必须匹配。

## 克隆 TTS

```http
POST {WORKSPACE_API_BASE_URL}/api/v1/services/aigc/multimodal-generation/generation
X-DashScope-SSE: enable
Content-Type: application/json

{
  "model": "qwen3-tts-vc-2026-01-22",
  "input": {
    "text": "Confirmed translation phrase.",
    "voice": "云端 voice ID",
    "language_type": "English"
  }
}
```

中间 SSE 的 `output.audio.data` 是 Base64 PCM16，按顺序解码后立即播放。最后一包 data
为空并携带完整文件 URL，不重复播放该 URL。只有 committed target 可以触发此接口。

## 删除音色

正常关闭时调用同一个 customization 地址：

```json
{
  "model": "qwen-voice-enrollment",
  "input": {"action": "delete", "voice": "云端 voice ID"}
}
```

删除采用 best effort。进程崩溃时必须由外部 registry/清理任务补偿，不能把云端 voice ID
误称为“只存在本机内存”。

## Doctor

ASR、MT 和语义端点通过 OpenAI-compatible `/models` 检查；TTS 通过声音列表操作验证鉴权
与 native API 可达性。Doctor 不创建音色、不合成音频，也不输出 API key。
