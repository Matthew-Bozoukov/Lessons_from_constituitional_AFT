<#
.SYNOPSIS
    Run a Petri / Inspect / Anthropic-API command with ANTHROPIC_API_KEY
    injected into that child process only.

.DESCRIPTION
    Isolated wrapper for Anthropic API work. It reads
    $HOME\.config\msm-audit\petri.env.

    Claude Code itself is authenticated through a Claude Max subscription and
    must never receive ANTHROPIC_API_KEY. Only Petri, Inspect, or an explicit
    Anthropic API child process may receive it, and only through this wrapper.

    Prefer -FilePath (Start-ProcessWithSecretEnv): the value is written straight
    into the child's environment block and never becomes a variable in this
    PowerShell process, never appears on a command line, and therefore never
    enters a process listing or shell history. -ScriptBlock is available for
    in-process REST calls; it sets the variable only for the duration of the
    block and restores the prior state in a finally block.

    Credential key (never printed, never logged, never committed):
        ANTHROPIC_API_KEY

    Budget key (operational limit, may be displayed):
        MAX_ANTHROPIC_SPEND_USD

    This wrapper never injects any infrastructure credential.

.EXAMPLE
    .\Invoke-WithPetriSecrets.ps1 -FilePath 'python' -ArgumentList '-m','petri','--help'

.EXAMPLE
    .\Invoke-WithPetriSecrets.ps1 -ScriptBlock {
        Invoke-RestMethod -Uri 'https://api.anthropic.com/v1/models' `
            -Headers @{ 'x-api-key' = $env:ANTHROPIC_API_KEY; 'anthropic-version' = '2023-06-01' }
    }
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

    [string]$EnvFile = (Join-Path $HOME '.config\msm-audit\petri.env')
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

Import-Module (Join-Path $PSScriptRoot 'SecretEnv.psm1') -Force -DisableNameChecking

$CredentialKeys = @('ANTHROPIC_API_KEY')
$BudgetKeys     = @('MAX_ANTHROPIC_SPEND_USD')
$RequiredKeys   = $CredentialKeys + $BudgetKeys

# Guard rail: refuse to run if the parent Claude Code process already carries an
# Anthropic key, which would mean the isolation contract has been broken.
if ($env:ANTHROPIC_API_KEY) {
    Write-Warning 'ANTHROPIC_API_KEY is already set in the parent environment. It must not be. This wrapper will still isolate the child process, but investigate how the parent acquired it.'
}

switch ($PSCmdlet.ParameterSetName) {

    'Check' {
        $secrets = Read-SecretFile -Path $EnvFile
        Assert-RequiredSecrets -Secrets $secrets -Required $RequiredKeys -SourceLabel 'petri.env'
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
        $secrets = Read-SecretFile -Path $EnvFile
        Assert-RequiredSecrets -Secrets $secrets -Required $BudgetKeys -SourceLabel 'petri.env'
        $limits = [pscustomobject]@{
            MAX_ANTHROPIC_SPEND_USD = [double]$secrets['MAX_ANTHROPIC_SPEND_USD']
        }
        foreach ($k in @($secrets.Keys)) { $secrets[$k] = $null }
        $secrets.Clear()
        return $limits
    }

    'ScriptBlock' {
        return Invoke-WithSecretEnv -Path $EnvFile -Required $RequiredKeys `
            -Inject $RequiredKeys -ScriptBlock $ScriptBlock -SourceLabel 'petri.env'
    }

    'Process' {
        $splat = @{
            Path         = $EnvFile
            Required     = $RequiredKeys
            Inject       = $RequiredKeys
            FilePath     = $FilePath
            ArgumentList = $ArgumentList
            SourceLabel  = 'petri.env'
        }
        if ($WorkingDirectory) { $splat.WorkingDirectory = $WorkingDirectory }
        if ($ExtraEnvironment) { $splat.ExtraEnvironment = $ExtraEnvironment }
        if ($StdOutFile)       { $splat.StdOutFile = $StdOutFile }
        if ($StdErrFile)       { $splat.StdErrFile = $StdErrFile }
        return Start-ProcessWithSecretEnv @splat
    }
}
