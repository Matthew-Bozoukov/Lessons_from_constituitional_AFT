<#
.SYNOPSIS
    Keep the watchdog heartbeat fresh for the duration of a long operation.

.DESCRIPTION
    Written after an incident: a 25-minute activity lease was issued for a vLLM
    restart that loaded six LoRA adapters serially and took over 30 minutes. The
    lease expired, the watchdog correctly saw no declared activity for 30.7
    minutes, and terminated a healthy pod mid-work.

    The watchdog behaved correctly. The defect was in how activity was declared:
    a single fixed-duration lease requires guessing an operation's runtime in
    advance, and any underestimate silently arms the idle timer.

    This removes that class of error. It refreshes the heartbeat on a short
    interval until told to stop, so declared activity tracks the operation's
    ACTUAL duration rather than a prediction of it. Use it around anything whose
    runtime is uncertain: model downloads, vLLM startup, long evaluations.

    Each keeper watches its OWN stop file, named after its activity. The first
    version used one shared stop file, which was safe only while a single
    operation ran at a time. Once two workstreams overlapped, stopping either
    one killed every keeper and could release a lease that another still-running
    operation depended on - reintroducing the exact failure this script exists
    to prevent.

.EXAMPLE
    $k = .\Start-HeartbeatKeeper.ps1 -Activity 'vllm-restart'
    # ... long operation ...
    .\Stop-HeartbeatKeeper.ps1 -Keeper $k
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$Activity,
    [int]$IntervalSeconds = 240,
    [int]$LeaseMinutes = 15,
    [int]$MaxHours = 12
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$root = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path

# One stop file per activity, so stopping one keeper cannot stop another.
$slug = ($Activity -replace '[^A-Za-z0-9\-]', '-')
$stopFile = Join-Path $root "runtime\watchdog\keeper-$slug.stop"
Remove-Item -LiteralPath $stopFile -Force -ErrorAction SilentlyContinue

$inner = @"
`$ErrorActionPreference = 'Continue'
`$deadline = (Get-Date).AddHours($MaxHours)
while ((Get-Date) -lt `$deadline) {
    if (Test-Path -LiteralPath '$stopFile') { break }
    & '$root\scripts\provider\Send-AuditHeartbeat.ps1' -Activity '$Activity' -BusyMinutes $LeaseMinutes | Out-Null
    Start-Sleep -Seconds $IntervalSeconds
}
"@

$encoded = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($inner))
$proc = Start-Process -FilePath (Get-Command powershell.exe).Source `
    -ArgumentList @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-EncodedCommand', $encoded) `
    -PassThru -WindowStyle Hidden

# Emit one immediately so there is no gap before the first tick.
& (Join-Path $root 'scripts\provider\Send-AuditHeartbeat.ps1') -Activity $Activity -BusyMinutes $LeaseMinutes | Out-Null

Write-Host "heartbeat keeper started (pid $($proc.Id)) for '$Activity': refreshing every ${IntervalSeconds}s with a ${LeaseMinutes}m lease"
[pscustomobject]@{ ProcessId = $proc.Id; StopFile = $stopFile; Activity = $Activity }
