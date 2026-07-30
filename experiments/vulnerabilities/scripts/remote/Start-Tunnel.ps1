<#
.SYNOPSIS
    Maintain an auto-reconnecting SSH tunnel from local 127.0.0.1:8000 to the
    pod's loopback vLLM port.

.DESCRIPTION
    vLLM binds to the pod's 127.0.0.1 only, so the tunnel is the sole path to the
    model. There is no publicly reachable model endpoint.

    Runs a supervision loop: if ssh exits for any reason, it waits briefly and
    redials. Writes tunnel.state.json so the provider monitor can report tunnel
    health, and keeps a rolling log.

    Launch detached via Start-DetachedProcessWithSecretEnv so the key lands in
    this process's environment block only.
#>
[CmdletBinding()]
param(
    [int]$LocalPort = 8000,
    [int]$RemotePort = 8000,
    [int]$RetryDelaySeconds = 5
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Continue'

$root = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$ep = Get-Content -LiteralPath (Join-Path $root 'runtime\provider-monitor\ssh-endpoint.json') -Raw | ConvertFrom-Json
$stateFile = Join-Path $root 'runtime\provider-monitor\tunnel.state.json'
$logFile = Join-Path $root 'logs\tunnel.log'
New-Item -ItemType Directory -Force -Path (Split-Path $logFile) | Out-Null

if (-not $env:MSM_SSH_PRIVATE_KEY) {
    throw 'MSM_SSH_PRIVATE_KEY absent. Launch through Invoke-WithInfraSecrets.'
}

function Write-TunnelState {
    param([string]$Status, [int]$Reconnects, [string]$Detail = '')
    @{
        pid                = $PID
        status             = $Status
        local_port         = $LocalPort
        reconnects         = $Reconnects
        detail             = $Detail
        last_heartbeat_utc = (Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ')
    } | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $stateFile -Encoding utf8
}

$reconnects = 0
Write-TunnelState -Status 'starting' -Reconnects 0
Add-Content -LiteralPath $logFile -Value "=== tunnel supervisor start pid $PID $(Get-Date -Format o) ==="

while ($true) {
    $sshArgs = @(
        '-i', $env:MSM_SSH_PRIVATE_KEY,
        '-p', "$($ep.ssh_port)",
        '-N',
        '-o', 'StrictHostKeyChecking=no',
        '-o', 'ExitOnForwardFailure=yes',
        '-o', 'ServerAliveInterval=15',
        '-o', 'ServerAliveCountMax=3',
        '-L', "127.0.0.1:${LocalPort}:127.0.0.1:${RemotePort}",
        "root@$($ep.ssh_host)"
    )

    Write-TunnelState -Status 'connecting' -Reconnects $reconnects
    Add-Content -LiteralPath $logFile -Value "[$(Get-Date -Format o)] dialing (reconnect #$reconnects)"

    $proc = Start-Process -FilePath 'ssh.exe' -ArgumentList $sshArgs -PassThru -NoNewWindow `
        -RedirectStandardError (Join-Path $root 'logs\tunnel.err.log')

    Start-Sleep -Seconds 3
    if (-not $proc.HasExited) { Write-TunnelState -Status 'up' -Reconnects $reconnects }

    $proc.WaitForExit()
    $reconnects++
    Add-Content -LiteralPath $logFile -Value "[$(Get-Date -Format o)] ssh exited code $($proc.ExitCode); redialing in ${RetryDelaySeconds}s"
    Write-TunnelState -Status 'down' -Reconnects $reconnects -Detail "ssh exit $($proc.ExitCode)"
    Start-Sleep -Seconds $RetryDelaySeconds
}
