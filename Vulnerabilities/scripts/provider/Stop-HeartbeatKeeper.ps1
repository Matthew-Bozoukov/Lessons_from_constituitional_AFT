<#
.SYNOPSIS
    Stop one heartbeat keeper and, if it was the last one, release its lease.

.DESCRIPTION
    Signals via that keeper's own stop file (so it exits its loop cleanly), then
    kills the process if it has not exited.

    Two rules matter when more than one operation runs at once:

      1. Only the named keeper is stopped. An earlier version swept every
         keeper process it could find, so stopping one workstream silently
         disarmed another's lease.

      2. -ReleaseLease only releases if no other keeper survives. Releasing the
         lease while another long operation is still running starts the
         watchdog's idle timer against work that is very much alive, which is
         how a healthy pod got terminated once already. When other keepers
         remain, the lease is left alone and they keep refreshing it.
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

$stopFile = $null
if ($Keeper -and $Keeper.PSObject.Properties.Match('StopFile').Count -gt 0) {
    $stopFile = $Keeper.StopFile
}
if (-not $stopFile) {
    Write-Host 'no keeper supplied; nothing to stop'
    return
}

New-Item -ItemType File -Path $stopFile -Force | Out-Null
Start-Sleep -Seconds 2

if ($Keeper.PSObject.Properties.Match('ProcessId').Count -gt 0 -and $Keeper.ProcessId) {
    if (Get-Process -Id $Keeper.ProcessId -ErrorAction SilentlyContinue) {
        Stop-Process -Id $Keeper.ProcessId -Force -ErrorAction SilentlyContinue
    }
}

Remove-Item -LiteralPath $stopFile -Force -ErrorAction SilentlyContinue

# Any other keeper still alive is identified by referencing a keeper-*.stop path.
$others = @(
    Get-CimInstance Win32_Process -Filter "Name = 'powershell.exe'" -ErrorAction SilentlyContinue |
        Where-Object {
            $_.ProcessId -ne $PID -and
            $_.CommandLine -and
            $_.CommandLine -match 'EncodedCommand'
        } |
        Where-Object {
            try {
                $decoded = [Text.Encoding]::Unicode.GetString(
                    [Convert]::FromBase64String((($_.CommandLine -split '-EncodedCommand\s+')[-1]).Trim()))
                $decoded -match 'keeper-.*\.stop'
            } catch { $false }
        }
)

if ($ReleaseLease) {
    if ($others.Count -gt 0) {
        Write-Host "keeper '$($Keeper.Activity)' stopped; $($others.Count) other keeper(s) still running, so the lease was NOT released"
    }
    else {
        & (Join-Path $root 'scripts\provider\Send-AuditHeartbeat.ps1') -Activity $NextActivity -ReleaseLease | Out-Null
        Write-Host "keeper '$($Keeper.Activity)' stopped; it was the last one, lease released (activity now '$NextActivity')"
    }
}
else {
    Write-Host "keeper '$($Keeper.Activity)' stopped; existing lease left to expire naturally"
}
