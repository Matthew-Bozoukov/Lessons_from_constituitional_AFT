<#
.SYNOPSIS
    Stop a running heartbeat keeper and release its activity lease.

.DESCRIPTION
    Signals via a stop file (so the keeper exits its loop cleanly), then kills
    the process if it has not exited. Optionally releases the lease immediately
    so the watchdog's idle timer starts from now rather than from the last
    lease expiry.
#>
[CmdletBinding()]
param(
    [Parameter(ValueFromPipeline)]$Keeper,
    [switch]$ReleaseLease,
    [string]$NextActivity = 'idle'
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Continue'

$root = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$stopFile = Join-Path $root 'runtime\watchdog\keeper.stop'

New-Item -ItemType File -Path $stopFile -Force | Out-Null
Start-Sleep -Seconds 2

if ($Keeper -and $Keeper.ProcessId) {
    $p = Get-Process -Id $Keeper.ProcessId -ErrorAction SilentlyContinue
    if ($p) { Stop-Process -Id $Keeper.ProcessId -Force -ErrorAction SilentlyContinue }
}
# Sweep any orphaned keepers from earlier runs.
Get-CimInstance Win32_Process -Filter "Name = 'powershell.exe'" -ErrorAction SilentlyContinue |
    Where-Object { $_.CommandLine -and $_.CommandLine -match 'EncodedCommand' -and $_.ProcessId -ne $PID } |
    ForEach-Object {
        # Only touch processes that reference the keeper stop file.
        if ($_.CommandLine -match 'keeper.stop') { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
    }

if ($ReleaseLease) {
    & (Join-Path $root 'scripts\provider\Send-AuditHeartbeat.ps1') -Activity $NextActivity -ReleaseLease | Out-Null
    Write-Host "heartbeat keeper stopped; lease released (activity now '$NextActivity')"
}
else {
    Write-Host 'heartbeat keeper stopped; existing lease left to expire naturally'
}

Remove-Item -LiteralPath $stopFile -Force -ErrorAction SilentlyContinue
