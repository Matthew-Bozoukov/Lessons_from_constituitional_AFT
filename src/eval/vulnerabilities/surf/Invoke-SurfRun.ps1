<#
.SYNOPSIS
    Launch a SURF EM run (or sweep) against the local vLLM checkpoint, with
    ANTHROPIC_API_KEY injected into the child process only.

.DESCRIPTION
    Windows PowerShell 5.1 tooling, generalized from
    experiments/vulnerabilities/scripts/surf/Invoke-SurfRun.ps1 at commit
    b38da52; the original run record lives in git history at that commit. It
    remains a Windows launcher and may warrant a bash port.

    SURF calls Anthropic from inside its own EM loop (query generation and
    rubric judging), so it needs a real callable endpoint and cannot be served
    by subagents. Per docs/08-api-vs-subscription-policy.md the judge runs on
    claude-haiku-4-5 rather than Opus.

    The target model is the already-running vLLM server reached through the SSH
    tunnel. SURF accepts a custom OpenAI-compatible endpoint written as
    "http://host:port/v1:model-name", so no second server is started and the
    GPU is not contended.

    ASCII only: PowerShell 5.1 reads .ps1 files as ANSI.

    This script never prints, logs or stores the credential. It only passes the
    wrapper a file path and an argument list. The secrets wrapper it invokes
    (scripts\secrets\Invoke-WithPetriSecrets.ps1) is no longer in the live
    tree: it lives in git history at commit b38da52, and its restored location
    must be supplied via -SecretsWrapper.

.PARAMETER Mode
    'run-em'  - a single EM loop (exposes concurrency and thinking controls)
    'sweep'   - N parallel EM loops (does NOT expose concurrency/thinking)

.PARAMETER RepoRoot
    Root directory that anchors the relative paths below. Defaults to the
    enclosing git worktree root (git rev-parse --show-toplevel); the original
    anchored on $PSScriptRoot\..\.. instead.

.PARAMETER SurfCheckout
    SURF checkout directory, relative to RepoRoot.

.PARAMETER Rubric
    Rubric YAML path, relative to RepoRoot. Required: the original default
    (seeds/surf-rubrics/harmful-omission.yaml) is only in git history at
    commit b38da52.

.PARAMETER EvidenceRoot
    Directory, relative to RepoRoot, under which the run output directory
    <EvidenceRoot>\<RunName> is created.

.PARAMETER LogRoot
    Directory, relative to RepoRoot, that receives the stdout/stderr logs.

.PARAMETER SecretsWrapper
    Path to Invoke-WithPetriSecrets.ps1. Required. The wrapper is not in the
    live tree; recover it from git history at commit b38da52.

.EXAMPLE
    .\Invoke-SurfRun.ps1 -Mode run-em -RunName pilot -Rubric rubrics\harmful-omission.yaml -SecretsWrapper C:\restored\Invoke-WithPetriSecrets.ps1 -Iterations 1 -Candidates 12
#>
[CmdletBinding()]
param(
    [ValidateSet('run-em', 'sweep')]
    [string]$Mode = 'run-em',

    [Parameter(Mandatory)][string]$RunName,

    [string]$RepoRoot = (git rev-parse --show-toplevel),
    [string]$SurfCheckout = 'third_party/SURF',
    [Parameter(Mandatory)][string]$Rubric,
    [string]$EvidenceRoot = 'evidence/surf',
    [string]$LogRoot = 'logs',
    [Parameter(Mandatory)][string]$SecretsWrapper,

    [string]$TargetModel = 'msm-aft-cot',
    [string]$JudgeModel = 'anthropic:claude-haiku-4-5',
    [string]$QueryModel = 'anthropic:claude-haiku-4-5',
    [string]$Attributes = 'seoirsem/CHUNKY-tulu3-SFT-25k-attributes',

    [int]$Iterations = 1,
    [int]$Candidates = 12,
    [int]$BufferSize = 10,
    [int]$NumRuns = 3,

    [int]$TargetConcurrency = 16,
    [int]$QueryConcurrency = 10,
    [int]$JudgeConcurrency = 20,

    [switch]$NoThinking,
    [int]$ThinkingBudget = 4000,

    # Target is a chain-of-thought checkpoint: this caps thinking AND answer
    # together. At SURF's stock 2048 a measurable share of generations were cut
    # off inside the <think> block, leaving a truncated scratchpad with no
    # answer - which the harmful-omission rubric then scored as a violation.
    # See docs/10-surf-status.md.
    [int]$TargetMaxTokens = 6144,

    [switch]$WhatIfOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$root = (Resolve-Path $RepoRoot).Path
$surfDir = Join-Path $root $SurfCheckout
$rubricPath = Join-Path $root $Rubric
$outDir = Join-Path $root (Join-Path $EvidenceRoot $RunName)
$logDir = Join-Path $root $LogRoot

foreach ($p in @($surfDir, $rubricPath)) {
    if (-not (Test-Path -LiteralPath $p)) { throw "missing required path: $p" }
}
New-Item -ItemType Directory -Force -Path $outDir  | Out-Null
New-Item -ItemType Directory -Force -Path $logDir  | Out-Null

# The target is the running vLLM server behind the SSH tunnel. Fail loudly and
# early if it is not up, rather than burning API credits on query generation
# whose target calls will all fail.
$models = $null
try {
    $models = Invoke-RestMethod -Uri 'http://127.0.0.1:8000/v1/models' -TimeoutSec 20
} catch {
    throw "vLLM endpoint http://127.0.0.1:8000/v1/models is not reachable. Start the tunnel before running SURF."
}
$served = @($models.data | ForEach-Object { $_.id })
if ($served -notcontains $TargetModel) {
    throw "target '$TargetModel' is not served. Available: $($served -join ', ')"
}
Write-Host "vLLM reachable; serving $($served.Count) models; target '$TargetModel' present."

$targetSpec = "http://127.0.0.1:8000/v1:$TargetModel"

$argList = @('-m', 'uv', 'run', 'python', '-m', 'surf.cli.main', $Mode,
             '--rubric', $rubricPath,
             '--attributes', $Attributes,
             '--output-dir', $outDir,
             '--target-model', $targetSpec,
             '--judge-model', $JudgeModel,
             '--query-model', $QueryModel,
             '--iterations', "$Iterations",
             '--candidates', "$Candidates",
             '--buffer-size', "$BufferSize")

if ($Mode -eq 'run-em') {
    $argList += @('--target-concurrency', "$TargetConcurrency",
                  '--query-concurrency',  "$QueryConcurrency",
                  '--judge-concurrency',  "$JudgeConcurrency",
                  '--thinking-budget',    "$ThinkingBudget",
                  '--target-max-tokens',  "$TargetMaxTokens")
    if ($NoThinking) { $argList += '--no-thinking' }
} else {
    $argList += @('--num-runs', "$NumRuns")
    if ($NoThinking) {
        throw "sweep does not expose --no-thinking; use -Mode run-em, or accept the default thinking judge."
    }
}

$stdout = Join-Path $logDir "surf-$RunName.out.log"
$stderr = Join-Path $logDir "surf-$RunName.err.log"

Write-Host ''
Write-Host "SURF $Mode"
Write-Host "  rubric      : $Rubric"
Write-Host "  target      : $targetSpec"
Write-Host "  judge       : $JudgeModel  (thinking budget $ThinkingBudget; NoThinking=$($NoThinking.IsPresent))"
Write-Host "  query       : $QueryModel"
Write-Host "  attributes  : $Attributes"
if ($Mode -eq 'sweep') { Write-Host "  runs        : $NumRuns" }
Write-Host "  iterations  : $Iterations"
Write-Host "  candidates  : $Candidates per iteration"
Write-Host "  output      : $outDir"
Write-Host "  stdout log  : $stdout"
Write-Host ''

if ($WhatIfOnly) {
    Write-Host 'WhatIfOnly set - not launching.'
    return
}

# PYTHONUTF8=1 puts the child interpreter in UTF-8 mode. SURF opens ~40 files
# without an explicit encoding; on Windows those default to the ANSI code page
# (cp1252 here), and a single non-Latin-1 character in a model response aborts
# the write and loses the whole iteration. UTF-8 mode fixes every one of those
# call sites at once without forking the upstream source. The two hot-path
# files are also patched explicitly so the fix survives a run made without
# this wrapper. Recorded in docs/10-surf-status.md.
$childEnv = @{ PYTHONUTF8 = '1'; PYTHONIOENCODING = 'utf-8' }

$wrapper = $SecretsWrapper
& $wrapper -FilePath (Get-Command python).Source `
    -ArgumentList $argList `
    -WorkingDirectory $surfDir `
    -ExtraEnvironment $childEnv `
    -StdOutFile $stdout `
    -StdErrFile $stderr
