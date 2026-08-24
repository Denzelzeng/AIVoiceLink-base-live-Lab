# 验收与测量

## 功能验收

- 可选择源语言、目标语言和文本/语音输出；
- 麦克风持续输入时显示识别原文和翻译文字；
- 首次 3–4 秒参考语音完成声音注册，后续满足条件的话轮能异步刷新 voice ID；
- 克隆 TTS 只播放 committed target；同一 voice ID 复用连接，voice ID 改变后新建 session；
- semantic incomplete 后下一声学段合并到同一逻辑话轮；
- 默认新 speech 不取消未播放、正在合成或正在播放的旧译音，译音按 FIFO 播放完成；
- 服务超时、空结果、Ctrl+C 都释放音频和 HTTP 资源；
- 正常结束调用云端声音删除 API；异常退出有补偿清理记录；
- 运行进程没有模型权重下载、GPU 初始化或本地推理依赖。

## 延迟指标

至少报告 P50/P95/P99：

| 指标 | baseline 事件 | 说明 |
|---|---|---|
| 首个源文确认 | `first_source_commit_ms` | 不是首个可回改 partial |
| 首个译文确认 | `first_target_commit_ms` | 不是 MT API 首 token |
| 首音频 chunk | `first_audio_ms` | 包含声音注册等待与 TTS API 首包 |
| 首个正确译音 | 外部强制对齐 | 用户真正听到正确语义的时间 |
| Average Lagging / YAAL | 长音频对齐 | 判断是否越译越慢 |
| Ending Offset | 源/目标音频结束时间 | 讲话结束后的尾部等待 |

`first_audio_ms` 不能代替“首个正确译音”。正式比较需要把输出译音重新 ASR 并强制对齐，
把网络、排队和播放延迟全部纳入。

## 质量、稳定性与成本

- ASR：CER/WER、数字、人名、缩写、口音；
- MT：COMET/xCOMET、chrF、漏译/增译/术语；
- 稳定性：Normalized Erasure、committed rollback（必须为 0）、重复率；
- TTS：首包、持续 RTF、队列长度、MOS/清晰度；
- 声纹：相似度、跨语言 CER/WER、注册耗时、降级标志；
- 端点：错切、过等、犹豫、列举、中英混说和噪声；
- 打断：旧播放停止时间、AEC 残留、自激活率；
- API：请求数、音频秒数、token/字符、错误率、429、P95 网络延迟和费用/小时。

## 数据集与压力

- 不只用预切句；至少包含 10–30 分钟长音频；
- 中->英、英->中分别测；
- 安静/噪声、快慢语速、口音、长句、多人切换；
- 30–60 分钟持续会话；
- 单路、目标并发、超限/429 三档；
- 冷连接和连接复用分开报告；
- 声音注册样本至少 3 秒连续清晰内容、24 kHz 以上。

JSONL 汇总：

```powershell
& '.\simultrans_baseline\scripts\summarize-events.ps1' `
  '.\simultrans_baseline\output\session.jsonl'
```
