# 实时同传方案集合

本仓库保留四套可独立运行的实时同传实现。它们的模型边界和部署位置不同，交接或继续开发时应先选定目标方案，不要混用各目录的依赖、配置或启动命令。

## 目录导航

| 目录 | 方案 | 模型/服务边界 | 适用场景 |
|---|---|---|---|
| [`official_api_examples`](official_api_examples/README.md) | 官网 API 参考示例 | 阿里云百炼 LiveTranslate API，或火山引擎 AST API | 验证官方端到端同传接口与协议 |
| [`cloud_api_pipeline`](cloud_api_pipeline/README.md) | 解耦合云 API 流水线 | 本地 VAD/换人检测 + 云端 ASR、MT、声音注册、TTS | 需要可替换模块、事件和延迟指标的云端基线 |
| [`local_deployment_pipeline`](local_deployment_pipeline/README.md) | 解耦合本地部署流水线 | C500 上的 Qwen3-ASR 与 Qwen3-Omni OpenAI 兼容服务 | 内网本地推理服务验证；当前没有 TTS |
| [`qwen35_omni_flash`](qwen35_omni_flash/README.md) | Qwen3.5-Omni Flash 单模型方案 | `qwen3.5-omni-flash-realtime` 的原生文本/语音输出 | 最小化集成，直接使用 Omni Realtime |

## 环境与凭据

- Windows 命令使用仓库 `AGENTS.md` 指定的 Conda 环境：`D:\ProgramData\miniforge3\envs\aivoicelink`。
- 根目录 `.env` 和 `Default Workspace-apiKey-*.csv` 仅保存本机凭据，已被 Git 忽略；各项目的 `.env.example` 只提供非敏感配置模板。
- 每个实现各自维护 `requirements.txt`（云 API 流水线还提供 `pyproject.toml`）。进入目标目录后按其 README 安装依赖。
- `official_api_examples/main_ast.py` 还需要火山引擎 SDK 生成的 `ast_python/python_protogen` 目录放在仓库根目录；该第三方生成目录不纳入版本控制。

## 交接建议

1. 先执行目标目录 README 中的 `--check`、`doctor` 或测试命令，确认网络、服务地址和麦克风权限。
2. 再修改对应目录的 `.env`、`configs/` 或命令行参数；不要把密钥、生成的 JSONL、音频输出或本地模型提交到仓库。
3. `cloud_api_pipeline` 的目录名已与方案职责对齐，但内部发行包/CLI 仍名为 `simultrans_baseline`，这是兼容现有导入和脚本的有意保留。

## 命名约定

目录统一使用小写 `snake_case`：名称表达部署边界或模型方案，而不是临时实验名称。新增方案请在本表登记，并在方案目录中提供独立 README、依赖清单和 `.env.example`。
