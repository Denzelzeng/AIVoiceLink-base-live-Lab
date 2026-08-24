# Local toolchain

Use this Conda environment for Python or other tools:

`D:\ProgramData\miniforge3\envs\aivoicelink`

You can freely install any packages/tools, like gcc, with `sudo mamba`, in the above env.
From PowerShell, run commands through:

```powershell
& 'D:\ProgramData\miniforge3\condabin\conda.bat' run --no-capture-output -p 'D:\ProgramData\miniforge3\envs\aivoicelink' <command>
```

For other tools like pdf that are specified by codex, use codex default bundled packages.

## Shell on Windows

Use PowerShell 7 for all shell commands:

C:\Users\chien\AppData\Local\Microsoft\WindowsApps\pwsh.exe

Invoke it with `-NoLogo -NoProfile`. Do not use Windows PowerShell 5.1 or the default PowerShell launcher.