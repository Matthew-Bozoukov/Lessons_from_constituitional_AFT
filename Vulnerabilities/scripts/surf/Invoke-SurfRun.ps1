<#
.SYNOPSIS
    Launch a SURF EM run (or sweep) against the local vLLM checkpoint, with
    ANTHROPIC_API_KEY injected into the child process only.

.DESCRIPTION
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
    wrapper a file path and an argument list.

.PARAMETER Mode
    'run-em'  - a single EM loop (exposes concurrency and thinking controls)
    'sweep'   - N parallel EM loops (does NOT expose concurrency/thinking)

.EXAMPLE
    .\Invoke-SurfRun.ps1 -Mode run-em -RunName pilot -Iterations 1 -Candidates 12
#>
[CmdletBinding()]
param(
    [ValidateSet('run-em', 'sweep')]
    [string]$Mode = 'run-em',

    [Parameter(Mandatory)][string]$RunName,

    [string]$Rubric = 'seeds/surf-rubrics/harmful-omission.yaml',
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

$root = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$surfDir = Join-Path $root 'tools\SURF'
$rubricPath = Join-Path $root $Rubric
$outDir = Join-Path $root "evidence\surf\$RunName"
$logDir = Join-Path $root 'logs'

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

$wrapper = Join-Path $root 'scripts\secrets\Invoke-WithPetriSecrets.ps1'
& $wrapper -FilePath (Get-Command python).Source `
    -ArgumentList $argList `
    -WorkingDirectory $surfDir `
    -ExtraEnvironment $childEnv `
    -StdOutFile $stdout `
    -StdErrFile $stderr
