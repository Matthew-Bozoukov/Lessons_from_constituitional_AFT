<#
.SYNOPSIS
    Run one short Petri audit with the auditor role driven through the
    claude-code Inspect provider (Claude Agent SDK), instead of anthropic/.

.DESCRIPTION
    Judge and realism stay on the normal Anthropic API so only the auditor
    path is under test. The target is the local vLLM endpoint.

    Two modes:
      -AllowApiKey       the CLI subprocess keeps ANTHROPIC_API_KEY. This tests
                         the ARCHITECTURE (translation, interception, Petri
                         integration). It is NOT a subscription test.
      (default)          the CLI subprocess gets a blanked ANTHROPIC_API_KEY and
                         must authenticate with Claude Code's own credential.

    ANTHROPIC_API_KEY is injected into the child process only, via
    scripts\secrets\Invoke-WithPetriSecrets.ps1. It never enters this process.
#>
[CmdletBinding()]
param(
    [string]$SeedDir = 'tools\petri-subscription\seed-one',
    [int]$MaxTurns = 8,
    [int]$Epochs = 1,
    [int]$MaxConnections = 1,
    [string]$LogDir = 'logs\petri-subscription',
    [string]$Auditor = 'claude-code/sonnet',
    [string]$Judge = 'anthropic/claude-opus-5',
    [string]$Realism = 'anthropic/claude-haiku-4-5',
    [string]$Target = 'openai-api/vllm/msm-aft-cot',
    [switch]$AllowApiKey,
    [string]$Tag = 'arch'
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$root = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
Set-Location $root
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

try {
    $m = Invoke-RestMethod -Uri 'http://127.0.0.1:8000/v1/models' -TimeoutSec 20
    $ids = @($m.data | ForEach-Object { $_.id })
    if ($ids -notcontains 'msm-aft-cot') { throw "adapter not served; got: $($ids -join ', ')" }
    Write-Host "target endpoint OK"
}
catch {
    throw "Target endpoint check failed: $($_.Exception.Message)"
}

$argList = @(
    'eval', 'inspect_petri/audit',
    '-T', "seed_instructions=$SeedDir",
    '-T', "max_turns=$MaxTurns",
    '-T', 'realism_filter=0.6',
    '-T', 'enable_rollback=True',
    '-T', 'target_tools=synthetic',
    '--model-role', "auditor=$Auditor",
    '--model-role', "judge=$Judge",
    '--model-role', "realism=$Realism",
    '--model-role', "target=$Target",
    '--model', $Target,
    '--epochs', "$Epochs",
    '--max-connections', "$MaxConnections",
    '--log-dir', $LogDir,
    '--log-format', 'eval',
    '--temperature', '0.7'
)

$extra = @{
    VLLM_BASE_URL     = 'http://127.0.0.1:8000/v1'
    VLLM_API_KEY      = 'local-tunnel-no-auth-required'
    INSPECT_LOG_LEVEL = 'info'
}
if ($AllowApiKey) { $extra['PETRI_CC_ALLOW_API_KEY'] = '1' }

Write-Host "mode: $(if ($AllowApiKey) { 'ARCHITECTURE (CLI keeps API key)' } else { 'SUBSCRIPTION (CLI key blanked)' })"
Write-Host "inspect $($argList -join ' ')"

$stdout = Join-Path $root "logs\petri-subscription-$Tag-stdout.log"
$stderr = Join-Path $root "logs\petri-subscription-$Tag-stderr.log"

$res = & (Join-Path $root 'scripts\secrets\Invoke-WithPetriSecrets.ps1') `
    -FilePath (Join-Path $root '.venv\Scripts\inspect.exe') `
    -ArgumentList $argList `
    -WorkingDirectory $root `
    -ExtraEnvironment $extra `
    -StdOutFile $stdout `
    -StdErrFile $stderr

Write-Host "exit code: $($res.ExitCode)"
Write-Host '--- stdout (tail) ---'
Write-Host (($res.StdOut -split "`n" | Select-Object -Last 40) -join "`n")
if ($res.ExitCode -ne 0) {
    Write-Host '--- stderr (tail) ---'
    Write-Host (($res.StdErr -split "`n" | Select-Object -Last 40) -join "`n")
}
