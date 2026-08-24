## Shell on Windows

Use PowerShell 7 for windows shell commands:

C:\Users\chien\AppData\Local\Microsoft\WindowsApps\pwsh.exe

Invoke it with `-NoLogo -NoProfile`. Do not use Windows PowerShell 5.1 or the default PowerShell launcher.

# Local toolchain

Use this Conda environment for Python or other tools:

`D:\ProgramData\miniforge3\envs\aivoicelink`

You can freely install any packages/tools, like gcc, with `sudo mamba`, in the above env.
From PowerShell, run commands through:

```powershell
& 'D:\ProgramData\miniforge3\condabin\conda.bat' run --no-capture-output -p 'D:\ProgramData\miniforge3\envs\aivoicelink' <command>
```

For other tools like pdf that are specified by codex, use codex default bundled packages.

## SSH

Use Windows' native OpenSSH client: `C:\Windows\System32\OpenSSH\ssh.exe`.

## Agent environment

Agents may use the Bash-native LXC for development and command execution when useful:

```powershell
ssh -i C:\Users\chien\.ssh\pve_ed25519 -o IdentitiesOnly=yes -J root@ssh.henrychen.in agent@10.10.101.4
```

It is CT 104 (`agent-dev`), running Debian 13. The `agent` account has passwordless sudo; do not change PVE or container system configuration without user approval.