<#
.SYNOPSIS
    Run a command or scriptblock with infrastructure credentials injected.

.DESCRIPTION
    Wrapper for INFRASTRUCTURE work only: Vast.ai, RunPod, Hugging Face and SSH
    to the rented GPU host. It reads $HOME\.config\msm-audit\infra.env.

    This wrapper never injects ANTHROPIC_API_KEY. Petri/Anthropic work uses the
    separate Invoke-WithPetriSecrets.ps1 wrapper, so an Anthropic credential can
    never reach an infrastructure child process and vice versa.

    Credential keys (never printed, never logged, never committed):
        VAST_API_KEY, RUNPOD_API_KEY, HF_TOKEN, MSM_SSH_PRIVATE_KEY

    Budget keys are operational limits rather than credentials. They are
    injected alongside the credentials and MAY be displayed, because the live
    status indicator is required to show remaining budget and time:
        MAX_GPU_SPEND_USD, MAX_WALL_CLOCK_HOURS, IDLE_SHUTDOWN_MINUTES

.EXAMPLE
    .\Invoke-WithInfraSecrets.ps1 -ScriptBlock {
        Invoke-RestMethod -Uri 'https://console.vast.ai/api/v0/users/current/' `
            -Headers @{ Authorization = "Bearer $env:VAST_API_KEY" }
    }

.EXAMPLE
    .\Invoke-WithInfraSecrets.ps1 -FilePath 'python' -ArgumentList 'query_offers.py'

.EXAMPLE
    # Read a budget limit without touching any credential.
    .\Invoke-WithInfraSecrets.ps1 -GetBudgetLimits
#>
[CmdletBinding(DefaultParameterSetName = 'ScriptBlock')]
param(
    [Parameter(Mandatory, ParameterSetName = 'ScriptBlock')]
    [scriptblock]$ScriptBlock,

    [Parameter(Mandatory, ParameterSetName = 'Process')]
    [string]$FilePath,

    [Parameter(ParameterSetName = 'Process')]
    [string[]]$ArgumentList = @(),

    [Parameter(ParameterSetName = 'Process')]
    [string]$WorkingDirectory,

    [Parameter(ParameterSetName = 'Process')]
    [hashtable]$ExtraEnvironment,

    [Parameter(ParameterSetName = 'Process')]
    [string]$StdOutFile,

    [Parameter(ParameterSetName = 'Process')]
    [string]$StdErrFile,

    [Parameter(Mandatory, ParameterSetName = 'Budget')]
    [switch]$GetBudgetLimits,

    [Parameter(Mandatory, ParameterSetName = 'Check')]
    [switch]$CheckOnly,

    # Restrict injection to a subset of keys. Defaults to the full infra set.
    [string[]]$Inject,

    [string]$EnvFile = (Join-Path $HOME '.config\msm-audit\infra.env')
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

Import-Module (Join-Path $PSScriptRoot 'SecretEnv.psm1') -Force -DisableNameChecking

$CredentialKeys = @('VAST_API_KEY', 'RUNPOD_API_KEY', 'HF_TOKEN', 'MSM_SSH_PRIVATE_KEY')
$BudgetKeys     = @('MAX_GPU_SPEND_USD', 'MAX_WALL_CLOCK_HOURS', 'IDLE_SHUTDOWN_MINUTES')
$RequiredKeys   = $CredentialKeys + $BudgetKeys

switch ($PSCmdlet.ParameterSetName) {

    'Check' {
        # Confirm the file parses and every required key is present and
        # non-empty, without revealing anything. Emits key names and a
        # non-reversible fingerprint per credential.
        $secrets = Read-SecretFile -Path $EnvFile
        Assert-RequiredSecrets -Secrets $secrets -Required $RequiredKeys -SourceLabel 'infra.env'
        $report = foreach ($k in $RequiredKeys) {
            [pscustomobject]@{
                key         = $k
                kind        = if ($CredentialKeys -contains $k) { 'credential' } else { 'budget-limit' }
                present     = $true
                fingerprint = if ($CredentialKeys -contains $k) { Get-SecretFingerprint -Value $secrets[$k] } else { $null }
            }
        }
        foreach ($k in @($secrets.Keys)) { $secrets[$k] = $null }
        $secrets.Clear()
        return $report
    }

    'Budget' {
        # Budget limits are operational configuration, not credentials.
        $secrets = Read-SecretFile -Path $EnvFile
        Assert-RequiredSecrets -Secrets $secrets -Required $BudgetKeys -SourceLabel 'infra.env'
        $limits = [pscustomobject]@{
            MAX_GPU_SPEND_USD     = [double]$secrets['MAX_GPU_SPEND_USD']
            MAX_WALL_CLOCK_HOURS  = [double]$secrets['MAX_WALL_CLOCK_HOURS']
            IDLE_SHUTDOWN_MINUTES = [double]$secrets['IDLE_SHUTDOWN_MINUTES']
        }
        foreach ($k in @($secrets.Keys)) { $secrets[$k] = $null }
        $secrets.Clear()
        return $limits
    }

    'ScriptBlock' {
        return Invoke-WithSecretEnv -Path $EnvFile -Required $RequiredKeys `
            -Inject $(if ($Inject) { $Inject } else { $RequiredKeys }) `
            -ScriptBlock $ScriptBlock -SourceLabel 'infra.env'
    }

    'Process' {
        $splat = @{
            Path         = $EnvFile
            Required     = $RequiredKeys
            Inject       = $(if ($Inject) { $Inject } else { $RequiredKeys })
            FilePath     = $FilePath
            ArgumentList = $ArgumentList
            SourceLabel  = 'infra.env'
        }
        if ($WorkingDirectory) { $splat.WorkingDirectory = $WorkingDirectory }
        if ($ExtraEnvironment) { $splat.ExtraEnvironment = $ExtraEnvironment }
        if ($StdOutFile)       { $splat.StdOutFile = $StdOutFile }
        if ($StdErrFile)       { $splat.StdErrFile = $StdErrFile }
        return Start-ProcessWithSecretEnv @splat
    }
}
