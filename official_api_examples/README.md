# 官方实时同传 API 参考示例

这里是两个彼此独立的官方端到端 API 示例，适合验证协议、麦克风采集和服务端同传能力；它们不是解耦合流水线的组成部分。

| 入口 | 服务 | 说明 |
|---|---|---|
| `main.py` | 阿里云百炼 `qwen3.5-livetranslate-flash-realtime` | 连续麦克风输入，输出实时译文和可选译音，默认启用声音复刻 |
| `main_ast.py` | 火山引擎 AST | 连续麦克风输入，输出原文/译文和可选译音 |

`livetranslate_client.py` 与 `ast_live_client.py` 分别是两个入口的客户端实现。`test_audio.wav` 是保留的音频样本；当前两个入口均以麦克风模式工作。

## 安装

在仓库根目录执行：

```powershell
& 'D:\ProgramData\miniforge3\condabin\conda.bat' run --no-capture-output -p 'D:\ProgramData\miniforge3\envs\aivoicelink' python -m pip install -r '.\official_api_examples\requirements.txt'
```

## 配置与启动

根目录 `.env` 不纳入版本控制。

- 百炼示例需要 `DASHSCOPE_API_KEY`：

  ```powershell
  & 'D:\ProgramData\miniforge3\condabin\conda.bat' run --no-capture-output -p 'D:\ProgramData\miniforge3\envs\aivoicelink' python '.\official_api_examples\main.py'
  ```

- 火山引擎 AST 示例需要 `AST_API_KEY`，或 `AST_APP_KEY` 与 `AST_ACCESS_KEY`。此外，请将官方 SDK 生成的 `ast_python/python_protogen` 放在仓库根目录（已忽略）：

  ```powershell
  & 'D:\ProgramData\miniforge3\condabin\conda.bat' run --no-capture-output -p 'D:\ProgramData\miniforge3\envs\aivoicelink' python '.\official_api_examples\main_ast.py'
  ```

两个示例都会打开麦克风和扬声器；建议使用耳机，避免译音被再次采集。
