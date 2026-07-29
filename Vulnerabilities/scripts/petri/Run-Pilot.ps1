<#
.SYNOPSIS
    Run the four-family Petri pilot against the served MSM+AFT-CoT checkpoint.

.DESCRIPTION
    Launched through Invoke-WithPetriSecrets so ANTHROPIC_API_KEY reaches the
    Inspect child process only, never the parent Claude Code environment.

    The target is the local end of the SSH tunnel. Inspect addresses an
    OpenAI-compatible endpoint as openai-api/<service>/<model>, resolving
    <SERVICE>_BASE_URL and <SERVICE>_API_KEY from the environment - hence
    VLLM_BASE_URL / VLLM_API_KEY below. vLLM needs no auth on loopback, so the
    key is a placeholder, not a credential.

    Model roles and pricing (per million tokens, recorded for the cost ledger):
      auditor  claude-sonnet-5   $3.00 in / $15.00 out  (intro $2.00/$10.00 to 2026-08-31)
      judge    claude-opus-5     $5.00 in / $25.00 out
      realism  claude-haiku-4-5  $1.00 in / $5.00 out
#>
[CmdletBinding()]
param(
    [string]$SeedDir = 'seeds\pilot',
    [int]$MaxTurns = 15,
    [int]$Epochs = 1,
    [double]$RealismFilter = 0.6,
    [int]$MaxConnections = 1,
    [string]$LogDir = 'logs\petri-pilot',
    [string]$Auditor = 'anthropic/claude-sonnet-5',
    [string]$Judge = 'anthropic/claude-opus-5',
    [string]$Target = 'openai-api/vllm/msm-aft-cot'
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$root = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
Set-Location $root
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

# Fail fast if the target is unreachable - better than burning auditor tokens
# against a dead endpoint.
try {
    $m = Invoke-RestMethod -Uri 'http://127.0.0.1:8000/v1/models' -TimeoutSec 20
    $ids = @($m.data | ForEach-Object { $_.id })
    if ($ids -notcontains 'msm-aft-cot') { throw "adapter not served; got: $($ids -join ', ')" }
    Write-Host "target endpoint OK: $($ids -join ', ')"
}
catch {
    throw "Target endpoint check failed: $($_.Exception.Message)"
}

& (Join-Path $root 'scripts\provider\Send-AuditHeartbeat.ps1') -Activity 'petri-pilot' -BusyMinutes 180 | Out-Null

$argList = @(
    'eval', 'inspect_petri/audit',
    '-T', "seed_instructions=$SeedDir",
    '-T', "max_turns=$MaxTurns",
    '-T', "realism_filter=$RealismFilter",
    '-T', 'enable_rollback=True',
    '-T', 'target_tools=synthetic',
    '--model-role', "auditor=$Auditor",
    '--model-role', "judge=$Judge",
    '--model', $Target,
    '--epochs', "$Epochs",
    '--max-connections', "$MaxConnections",
    '--log-dir', $LogDir,
    '--log-format', 'eval',
    '--temperature', '0.7',
    '--no-fail-on-error'
)

Write-Host "inspect $($argList -join ' ')"

$res = & (Join-Path $root 'scripts\secrets\Invoke-WithPetriSecrets.ps1') `
    -FilePath (Join-Path $root '.venv\Scripts\inspect.exe') `
    -ArgumentList $argList `
    -WorkingDirectory $root `
    -ExtraEnvironment @{
        VLLM_BASE_URL = 'http://127.0.0.1:8000/v1'
        VLLM_API_KEY  = 'local-tunnel-no-auth-required'
        INSPECT_LOG_LEVEL = 'info'
    } `
    -StdOutFile (Join-Path $root 'logs\petri-pilot-stdout.log') `
    -StdErrFile (Join-Path $root 'logs\petri-pilot-stderr.log')

Write-Host "exit code: $($res.ExitCode)"
Write-Host ($res.StdOut | Select-Object -Last 60)
if ($res.ExitCode -ne 0) {
    Write-Host '--- stderr (tail) ---'
    Write-Host (($res.StdErr -split "`n" | Select-Object -Last 40) -join "`n")
}

& (Join-Path $root 'scripts\provider\Update-ProviderStatus.ps1') -Reason 'post-audit'
