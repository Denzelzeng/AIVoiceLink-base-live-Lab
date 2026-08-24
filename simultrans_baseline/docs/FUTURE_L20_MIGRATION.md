# 未来 L20 迁移边界（不属于当前运行版本）

当前版本的核心 ASR/MT/语义端点/TTS 调用远端 API，本地只运行轻量 VAD 与说话人 embedding；
不包含 L20 上的核心模型服务配置。本文件只记录未来另开
部署版本时的接口边界，不能按本文把当前版本描述为已支持本地部署。

| 当前 provider | 未来可替换边界 | 保持不变 |
|---|---|---|
| `DashScopeASR` | L20 stateful streaming ASR provider | `ASRBackend` 输出完整假设 |
| `QwenMTTranslator` | L20 MT provider | 源/目标 LCP 与提交状态 |
| `LLMSemanticEndpoint` | 本地或专用端点 provider | complete/incomplete 契约、hard timeout |
| `DashScopeVoiceCloneTTS` | L20 声纹/TTS provider | voice profile 生命周期、TTS PCM chunks |

建议未来迁移以独立分支或独立配置包完成，并加入以下门禁：

1. 默认云 API 配置仍可运行，避免迁移期间失去对照组；
2. 本地 provider 不能绕过显式声音复刻授权；
3. ASR partial/final、TTS PCM 和删除 profile 的 contract tests 必须共用；
4. 单路闭环、显存、并发、30–60 分钟稳定性和 AEC 验收后再成为默认；
5. 云 API 与 L20 使用同一质量、延迟和成本口径比较。

未来 provider 的实现不应重新引入 LiveTranslate 端到端模型；仍保持 ASR、MT、语义端点、
声音注册和 TTS 分立。
