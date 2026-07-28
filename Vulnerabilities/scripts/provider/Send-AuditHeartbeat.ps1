<#
.SYNOPSIS
    Declare orchestration activity to the cleanup watchdog.

.DESCRIPTION
    The watchdog's idle rule counts idle time only when no activity lease is
    held AND the heartbeat has gone stale. Long operations therefore take a
    lease covering their expected duration so they are never mistaken for
    inactivity:

        Send-AuditHeartbeat.ps1 -Activity 'model-download' -BusyMinutes 90

    Short beats between operations just refresh the timestamp:

        Send-AuditHeartbeat.ps1 -Activity 'petri-audit-3'

    Release a lease early when an operation finishes ahead of its estimate:

        Send-AuditHeartbeat.ps1 -Activity 'idle' -ReleaseLease

    Writes runtime/watchdog/heartbeat.json. Contains no credential.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$Activity,
    [double]$BusyMinutes = 0,
    [switch]$ReleaseLease
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$root = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$hbFile = Join-Path $root 'runtime\watchdog\heartbeat.json'
$dir = Split-Path -Parent $hbFile
if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Path $dir -Force | Out-Null }

$now = (Get-Date).ToUniversalTime()

$busyUntil = $null
if (-not $ReleaseLease -and $BusyMinutes -gt 0) {
    $busyUntil = $now.AddMinutes($BusyMinutes).ToString('yyyy-MM-ddTHH:mm:ssZ')
}
elseif (-not $ReleaseLease -and (Test-Path -LiteralPath $hbFile)) {
    # Preserve a longer lease that is still running.
    try {
        $prev = Get-Content -LiteralPath $hbFile -Raw | ConvertFrom-Json
        if ($prev.busy_until_utc -and ([datetime]::Parse($prev.busy_until_utc)).ToUniversalTime() -gt $now) {
            $busyUntil = $prev.busy_until_utc
        }
    } catch { }
}

[ordered]@{
    last_heartbeat_utc = $now.ToString('yyyy-MM-ddTHH:mm:ssZ')
    activity           = $Activity
    busy_until_utc     = $busyUntil
    pid                = $PID
} | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $hbFile -Encoding utf8

if ($busyUntil) { Write-Host "heartbeat: '$Activity' (lease until $busyUntil)" }
else { Write-Host "heartbeat: '$Activity'" }
