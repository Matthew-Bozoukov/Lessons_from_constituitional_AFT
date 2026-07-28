<#
.SYNOPSIS
    Persistent provider status monitor. Runs independently of the experiment.

.DESCRIPTION
    Refreshes every figure at least once every $IntervalSeconds (default 240s,
    inside the required five-minute ceiling) and prints a compact status line to
    its own console so the operator always has a visible indicator.

    This process expects RUNPOD_API_KEY in its environment and is therefore
    launched through Invoke-WithInfraSecrets.ps1 -FilePath, which writes the
    value straight into the child environment block. The key is never printed.

    The monitor deliberately does NOT terminate anything. Enforcement is the
    cleanup watchdog's job, so a monitor crash cannot leave a GPU running and a
    watchdog crash cannot blind the operator.
#>
[CmdletBinding()]
param(
    [int]$IntervalSeconds = 240,
    [double]$AnthropicSpentUsd = 0.0,
    [double]$MaxAnthropicSpendUsd = 0.0
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Continue'

$root = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
Import-Module (Join-Path $root 'scripts\provider\RunPodApi.psm1')       -Force -DisableNameChecking
Import-Module (Join-Path $root 'scripts\provider\ProviderStatus.psm1')  -Force -DisableNameChecking

$pidFile = Join-Path $PSScriptRoot 'monitor.pid'
@{
    pid        = $PID
    started_at = (Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ')
    interval_s = $IntervalSeconds
} | ConvertTo-Json | Set-Content -LiteralPath $pidFile -Encoding utf8

Write-Host ''
Write-Host '=============================================================='
Write-Host ' MSM audit - provider status monitor'
Write-Host " PID $PID   refresh every ${IntervalSeconds}s"
Write-Host '=============================================================='
Write-Host ''

try {
    while ($true) {
        try {
            $line = Write-ProviderStatus -Reason 'periodic' `
                -AnthropicSpentUsd $AnthropicSpentUsd -MaxAnthropicSpendUsd $MaxAnthropicSpendUsd
            $stamp = (Get-Date).ToString('HH:mm:ss')
            Write-Host "[$stamp] $line"
        }
        catch {
            Write-Host "[$(Get-Date -Format HH:mm:ss)] status refresh failed: $($_.Exception.Message)"
        }
        Start-Sleep -Seconds $IntervalSeconds
    }
}
finally {
    Remove-Item -LiteralPath $pidFile -Force -ErrorAction SilentlyContinue
}
