<#
.SYNOPSIS
    Launch the auto-reconnecting SSH tunnel to the pod's vLLM port, detached.

.DESCRIPTION
    vLLM binds the pod's 127.0.0.1 only, so this tunnel is the sole path to the
    model and there is no publicly reachable model endpoint.

    Two things make this its own script rather than an inline command:

    1. **It must outlive the call that starts it.** A tunnel launched inside a
       normal tool call dies when that call returns, which has bitten this
       repository before. `Start-DetachedProcessWithSecretEnv` returns
       immediately and the child keeps running.

    2. **The SSH key must reach the child and nowhere else.** The key is
       injected into the child process's environment block only; it never
       becomes a variable in this shell and never appears on a command line.

    Reuses the sibling experiment's Start-Tunnel.ps1, which supervises the
    connection and redials if ssh drops, and writes tunnel.state.json so the
    provider monitor can report tunnel health.

.EXAMPLE
    scripts\Start-PetriTunnel.ps1
    scripts\Start-PetriTunnel.ps1 -Verify
#>
[CmdletBinding()]
param(
    [int]$LocalPort = 8000,
    [int]$RemotePort = 8000,
    # Poll the tunnelled endpoint until vLLM answers, then list the served arms.
    [switch]$Verify,
    [int]$VerifyTimeoutSec = 120
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$vuln = (Resolve-Path (Join-Path $PSScriptRoot '..\..\..\vulnerabilities')).Path
$infraEnv = Join-Path $HOME '.config\msm-audit\infra.env'
$endpointFile = Join-Path $vuln 'runtime\provider-monitor\ssh-endpoint.json'

if (-not (Test-Path $endpointFile)) {
    throw "No SSH endpoint recorded at $endpointFile. Provision a pod first."
}
$ep = Get-Content -LiteralPath $endpointFile -Raw | ConvertFrom-Json
Write-Host "[tunnel] target pod endpoint: port $($ep.ssh_port) (host not printed)"

# Don't stack tunnels on the same local port - a second one silently fails to
# bind and the first keeps serving, which looks like success against a stale pod.
#
# But "a tunnel is running" is NOT the same as "the RIGHT tunnel is running".
# On 2026-07-30 this guard found the tunnel to an already-terminated pod,
# declined to start a new one, and left the endpoint pointing at a dead machine -
# precisely the stale-pod failure it exists to prevent, just from the other
# direction. So compare the live tunnel's recorded endpoint against the current
# one and replace it when they disagree.
$existing = @(Get-CimInstance Win32_Process -Filter "Name='powershell.exe'" -ErrorAction SilentlyContinue |
    Where-Object { $_.CommandLine -and $_.CommandLine -match 'Start-Tunnel' })

if ($existing.Count -gt 0) {
    $stateFile = Join-Path $vuln 'runtime\provider-monitor\tunnel.state.json'
    $servesCurrentPod = $false
    if (Test-Path $stateFile) {
        try {
            $ts = Get-Content -LiteralPath $stateFile -Raw | ConvertFrom-Json
            # The tunnel records the port it dialled; a mismatch means it belongs
            # to a previous pod.
            $servesCurrentPod = ($ts.PSObject.Properties.Match('remote_port').Count -gt 0 -and
                                 "$($ts.remote_port)" -eq "$($ep.ssh_port)")
        } catch { $servesCurrentPod = $false }
    }
    if ($servesCurrentPod) {
        Write-Host "[tunnel] already running for THIS pod (PID $($existing[0].ProcessId))"
    } else {
        Write-Host "[tunnel] found a tunnel for a DIFFERENT pod - terminating PID(s) $($existing.ProcessId -join ', ')"
        foreach ($e in $existing) { Stop-Process -Id $e.ProcessId -Force -ErrorAction SilentlyContinue }
        Start-Sleep -Seconds 3
        $existing = @()
    }
}

if ($existing.Count -gt 0) { } else {
    Import-Module (Join-Path $vuln 'scripts\secrets\SecretEnv.psm1') -Force -DisableNameChecking
    $proc = Start-DetachedProcessWithSecretEnv `
        -Path $infraEnv `
        -Required @('MSM_SSH_PRIVATE_KEY') `
        -FilePath 'powershell.exe' `
        -ArgumentList @(
            '-NoProfile', '-NonInteractive', '-ExecutionPolicy', 'Bypass',
            '-File', (Join-Path $vuln 'scripts\remote\Start-Tunnel.ps1'),
            '-LocalPort', "$LocalPort", '-RemotePort', "$RemotePort"
        ) `
        -WorkingDirectory $vuln `
        -SourceLabel 'infra.env'
    Write-Host "[tunnel] started detached, PID $($proc.ProcessId)"
}

if (-not $Verify) { return }

Write-Host "[tunnel] waiting for vLLM through 127.0.0.1:$LocalPort"
$deadline = (Get-Date).AddSeconds($VerifyTimeoutSec)
while ((Get-Date) -lt $deadline) {
    try {
        $m = Invoke-RestMethod -Uri "http://127.0.0.1:$LocalPort/v1/models" -TimeoutSec 10
        $ids = @($m.data | ForEach-Object { $_.id })
        Write-Host "[tunnel] vLLM reachable. Arms served: $($ids -join ', ')"
        $expected = @('base', 'dose-10-90', 'dose-20-80', 'dose-40-60')
        $missing = $expected | Where-Object { $ids -notcontains $_ }
        if ($missing) { throw "arms missing from the endpoint: $($missing -join ', ')" }
        Write-Host "[tunnel] all four arms present"
        return
    } catch {
        Start-Sleep -Seconds 5
    }
}
throw "vLLM did not answer through the tunnel within $VerifyTimeoutSec s. Check /workspace/logs/vllm.log on the box."
