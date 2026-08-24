# Qwen3.5-Omni 实时语音翻译工具

这是一个使用阿里云百炼 `qwen3.5-omni-flash-realtime` 的连续麦克风口译客户端，
支持模型原生文本和语音输出。

- 输入：麦克风语音（16 kHz、单声道），自动按静音切句
- 语言：可选择输入和输出语言，也可直接输入模型支持的语言/方言名称
- 输出：SSE 流式文本；可选 24 kHz 流式语音播放
- 校验：每段显示 ASR 识别语言、输入原文和译文；输入转写为空时不播放模型语音
- 凭据：自动读取仓库根目录的 `.env` 和 `Default Workspace-apiKey-*.csv`
- 额外模式：单段文本、音频文件、麦克风设备列表、API 连通性检查

实时语音只读取 Omni Realtime 的 `response.audio.delta`，不调用独立 TTS 模型。

## 工作方式

工具与 `qwen3.5-omni-flash-realtime` 保持 WebSocket 会话，连续监听麦克风，以约
600 ms 静音检测句尾，然后提交 PCM 语音。译文和模型语音会分别从实时事件流返回。

## 安装

在仓库根目录执行：

```powershell
& 'D:\ProgramData\miniforge3\condabin\conda.bat' run --no-capture-output -p 'D:\ProgramData\miniforge3\envs\aivoicelink' python -m pip install -r '.\Qwen3.5-Omni\requirements.txt'
```

## 运行

交互式启动（选择输入语言、输出语言和是否播放语音）：

```powershell
& 'D:\ProgramData\miniforge3\condabin\conda.bat' run --no-capture-output -p 'D:\ProgramData\miniforge3\envs\aivoicelink' python '.\Qwen3.5-Omni\main.py'
```

直接启动中文到英文、文本加语音：

```powershell
& 'D:\ProgramData\miniforge3\condabin\conda.bat' run --no-capture-output -p 'D:\ProgramData\miniforge3\envs\aivoicelink' python '.\Qwen3.5-Omni\main.py' --source-language Chinese --target-language English --audio-output
```

仅文本输出：

```powershell
& 'D:\ProgramData\miniforge3\condabin\conda.bat' run --no-capture-output -p 'D:\ProgramData\miniforge3\envs\aivoicelink' python '.\Qwen3.5-Omni\main.py' --source-language Cantonese --target-language Chinese --text-only
```

列出麦克风设备并指定其中一个：

```powershell
& 'D:\ProgramData\miniforge3\condabin\conda.bat' run --no-capture-output -p 'D:\ProgramData\miniforge3\envs\aivoicelink' python '.\Qwen3.5-Omni\main.py' --list-devices

& 'D:\ProgramData\miniforge3\condabin\conda.bat' run --no-capture-output -p 'D:\ProgramData\miniforge3\envs\aivoicelink' python '.\Qwen3.5-Omni\main.py' --source-language Chinese --target-language English --audio-output --device-index 1
```

## 单次验证

API 和模型列表：

```powershell
& 'D:\ProgramData\miniforge3\condabin\conda.bat' run --no-capture-output -p 'D:\ProgramData\miniforge3\envs\aivoicelink' python '.\Qwen3.5-Omni\main.py' --check
```

翻译文字：

```powershell
& 'D:\ProgramData\miniforge3\condabin\conda.bat' run --no-capture-output -p 'D:\ProgramData\miniforge3\envs\aivoicelink' python '.\Qwen3.5-Omni\main.py' --text '欢迎参加今天的会议。' --source-language Chinese --target-language English --text-only
```

翻译音频并保存语音译文：

```powershell
& 'D:\ProgramData\miniforge3\condabin\conda.bat' run --no-capture-output -p 'D:\ProgramData\miniforge3\envs\aivoicelink' python '.\Qwen3.5-Omni\main.py' --file '.\sample.wav' --target-language English --audio-output --save-audio '.\translation.wav'
```

强制用 `qwen3.5-omni-flash-realtime` 验证模型原生语音返回：

```powershell
& 'D:\ProgramData\miniforge3\condabin\conda.bat' run --no-capture-output -p 'D:\ProgramData\miniforge3\envs\aivoicelink' python '.\Qwen3.5-Omni\main.py' --file '.\sample.wav' --realtime --source-language Chinese --target-language English --audio-output --save-audio '.\omni-realtime.wav'
```

麦克风模式默认使用 `qwen3.5-omni-flash-realtime`；HTTP 模型仅保留给文字和普通
文件测试。实时模式播放和保存的数据只来自 Omni 的 `response.audio.delta`，不调用
独立 TTS 模型。

## 延迟调节

- 句尾反应太慢：降低 `--end-silence-ms`，例如 `450`。
- 背景噪声触发误录：提高 `--energy-threshold`，例如 `600`。
- 连续长句反馈太慢：降低 `--max-segment-ms`，例如 `5000`。
- 环境噪声会在启动时校准；用 `--calibration-seconds 0` 可关闭。

## Windows 麦克风故障排查

- 如果旧版本曾弹出 `python.exe` 错误窗口，先关闭该窗口再启动新版。新版让录音和
  播放共享单一 PortAudio 实例，避免 Windows 上并发初始化造成 access violation。
- `PortAudio -9999`：关闭占用麦克风的会议/录音软件，并在“Windows 设置 > 隐私
  和安全性 > 麦克风”中打开“允许桌面应用访问麦克风”。
- `PortAudio -9996`：设备编号已变化；重新运行 `--list-devices`。
- 工具按设备公布的原生采样率（通常 44.1 或 48 kHz）打开麦克风，再把单声道
  16-bit PCM 正确重采样为 Realtime 会话使用的 16 kHz；不会用设备不支持的采样率
  强行打开麦克风。

运行 `python .\Qwen3.5-Omni\main.py --help` 查看全部选项。
