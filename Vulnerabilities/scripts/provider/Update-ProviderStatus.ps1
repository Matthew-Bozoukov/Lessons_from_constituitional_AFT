<#
.SYNOPSIS
    One-shot provider status refresh, for event boundaries.

.DESCRIPTION
    Call after provisioning, after model download, after model-server startup,
    after every Petri audit, and before and after cleanup. The periodic monitor
    covers the five-minute ceiling; this covers the events.

    Wraps itself in Invoke-WithInfraSecrets so it can be called directly.

.EXAMPLE
    .\Update-ProviderStatus.ps1 -Reason post-provision
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory)]
    [ValidateSet('post-provision','post-model-download','post-server-start','post-audit',
                 'pre-cleanup','post-cleanup','manual','periodic','preflight')]
    [string]$Reason,

    [double]$AnthropicSpentUsd = 0.0,
    [double]$MaxAnthropicSpendUsd = 0.0,
    [switch]$SkipHealthChecks
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$root = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path

$line = & (Join-Path $root 'scripts\secrets\Invoke-WithInfraSecrets.ps1') -Inject @('RUNPOD_API_KEY') -ScriptBlock {
    Import-Module (Join-Path $root 'scripts\provider\RunPodApi.psm1')      -Force -DisableNameChecking
    Import-Module (Join-Path $root 'scripts\provider\ProviderStatus.psm1') -Force -DisableNameChecking
    Write-ProviderStatus -Reason $Reason -AnthropicSpentUsd $AnthropicSpentUsd `
        -MaxAnthropicSpendUsd $MaxAnthropicSpendUsd -SkipHealthChecks:$SkipHealthChecks
}

Write-Host $line
