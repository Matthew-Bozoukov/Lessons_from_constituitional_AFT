<#
.SYNOPSIS
    Initialise run state and launch the provider monitor and cleanup watchdog.

.DESCRIPTION
    Must run BEFORE any paid GPU is provisioned. Both children get their own
    copy of RUNPOD_API_KEY injected directly into their environment block, so
    the key never becomes a variable in this shell and never appears on a
    command line.

    Monitor and watchdog are separate processes on purpose: a monitor crash must
    not leave a GPU running, and a watchdog crash must not blind the operator.

.EXAMPLE
    .\Start-Monitoring.ps1 -Gpu 'NVIDIA A100 80GB PCIe' -HourlyUsd 1.19
#>
[CmdletBinding()]
param(
    [string]$Provider = 'runpod',
    [Parameter(Mandatory)][string]$Gpu,
    [Parameter(Mandatory)][double]$HourlyUsd,
    [double]$StorageHourlyUsd = 0.0,
    [int]$MonitorIntervalSeconds = 240,
    [int]$WatchdogIntervalSeconds = 60,
    [switch]$Visible
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$root       = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$secretsDir = Join-Path $root 'scripts\secrets'
$infraEnv   = Join-Path $HOME '.config\msm-audit\infra.env'

Import-Module (Join-Path $secretsDir 'SecretEnv.psm1') -Force -DisableNameChecking

# ---- limits and starting balance -------------------------------------------
$limits = & "$secretsDir\Invoke-WithInfraSecrets.ps1" -GetBudgetLimits

$starting = & "$secretsDir\Invoke-WithInfraSecrets.ps1" -Inject @('RUNPOD_API_KEY') -ScriptBlock {
    Import-Module (Join-Path $root 'scripts\provider\RunPodApi.psm1') -Force -DisableNameChecking
    try { (Get-RunPodBalance).balance_usd } catch { [double]::NaN }
}

$startingBasis = if ([double]::IsNaN($starting)) { 'unavailable' } else { 'exact-provider-reported' }

Import-Module (Join-Path $root 'scripts\provider\ProviderStatus.psm1') -Force -DisableNameChecking
$state = Initialize-RunState -Provider $Provider -Gpu $Gpu -HourlyUsd $HourlyUsd `
    -StorageHourlyUsd $StorageHourlyUsd `
    -MaxGpuSpendUsd $limits.MAX_GPU_SPEND_USD `
    -MaxWallClockHours $limits.MAX_WALL_CLOCK_HOURS `
    -IdleShutdownMinutes $limits.IDLE_SHUTDOWN_MINUTES `
    -StartingBalanceUsd $starting -StartingBalanceBasis $startingBasis

Write-Host "run_id                 : $($state.run_id)"
Write-Host "starting balance       : $(if ($startingBasis -eq 'unavailable') { 'unavailable' } else { '$' + [math]::Round($starting,2) }) ($startingBasis)"
Write-Host "GPU spend cap          : `$$($limits.MAX_GPU_SPEND_USD)"
Write-Host "wall-clock cap         : $($limits.MAX_WALL_CLOCK_HOURS) h"
Write-Host "idle shutdown          : $($limits.IDLE_SHUTDOWN_MINUTES) min"

# ---- launch children --------------------------------------------------------
$pwsh = (Get-Command powershell.exe).Source

$monitor = Start-DetachedProcessWithSecretEnv -Path $infraEnv -Required @('RUNPOD_API_KEY') `
    -Inject @('RUNPOD_API_KEY') -FilePath $pwsh -Visible:$Visible -WorkingDirectory $root `
    -ArgumentList @('-NoProfile','-ExecutionPolicy','Bypass','-File',
                    (Join-Path $root 'runtime\provider-monitor\Monitor-Loop.ps1'),
                    '-IntervalSeconds', "$MonitorIntervalSeconds")

$watchdog = Start-DetachedProcessWithSecretEnv -Path $infraEnv -Required @('RUNPOD_API_KEY') `
    -Inject @('RUNPOD_API_KEY') -FilePath $pwsh -Visible:$Visible -WorkingDirectory $root `
    -ArgumentList @('-NoProfile','-ExecutionPolicy','Bypass','-File',
                    (Join-Path $root 'runtime\watchdog\Watchdog-Loop.ps1'),
                    '-CheckIntervalSeconds', "$WatchdogIntervalSeconds")

Write-Host "monitor  PID $($monitor.ProcessId)  (refresh ${MonitorIntervalSeconds}s)"
Write-Host "watchdog PID $($watchdog.ProcessId)  (check   ${WatchdogIntervalSeconds}s)"

@{
    monitor_pid  = $monitor.ProcessId
    watchdog_pid = $watchdog.ProcessId
    started_at   = (Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ')
} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $root 'runtime\monitoring.processes.json') -Encoding utf8
