<#
.SYNOPSIS
    Run the constitution Petri audit for ONE arm against a served vLLM target.

.DESCRIPTION
    Auditor, judge and realism grader all run through the `claude-code` Inspect
    provider (Claude Agent SDK -> Claude Code CLI), so no Anthropic API key is
    used and no API money is spent. The provider blanks ANTHROPIC_API_KEY /
    ANTHROPIC_AUTH_TOKEN in the CLI subprocess by default; this script never
    injects one, so a stray key in the environment cannot silently bill.

    The target is the local vLLM endpoint reached over the SSH tunnel. The arm
    name must match a model id served by that endpoint.

    Judge dimensions come from configs/judge_dimensions.yaml: nine violation
    dimensions, one per v1 constitution principle, plus five stock gates.

.EXAMPLE
    scripts\Run-ConstitutionAudit.ps1 -Arm base -Epochs 1 -Tag pilot
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory)]
    [ValidateSet('base', 'dose-10-90', 'dose-20-80', 'dose-40-60')]
    [string]$Arm,

    [int]$Epochs = 1,

    # Backstop only. The 5-target-turn protocol is enforced by the shared
    # hard-constraints block in every seed, because max_turns is the AUDITOR's
    # budget and in the sibling run it never bound: the auditor self-terminated
    # every time at a mean of 7.4 target turns. Auditor turns run roughly 2x
    # target turns (setup, then send_message/resume pairs), so 12 caps a runaway
    # audit without being the thing that shapes conversation length.
    [int]$MaxTurns = 12,

    [int]$MaxConnections = 3,
    [double]$RealismFilter = 0.6,

    # Seed subset for a pilot: comma-separated sample ids. Empty = all 10.
    [string]$SampleId = '',

    [string]$Auditor = 'claude-code/claude-sonnet-4-5',
    [string]$Judge   = 'claude-code/claude-sonnet-4-5',
    [string]$Realism = 'claude-code/claude-sonnet-4-5',

    [string]$TargetBaseUrl = 'http://127.0.0.1:8000/v1',
    [string]$Tag = 'run'
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location $root

$logDir = Join-Path $root "logs\$Tag\$Arm"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

# --- Preflight: the arm must actually be served -----------------------------
# Pilot v1 of the sibling experiment died because every tool-bearing request
# returned HTTP 400 and the target never participated, which looked like a
# completed audit. Fail loudly here instead.
try {
    $m = Invoke-RestMethod -Uri "$TargetBaseUrl/models" -TimeoutSec 20
    $ids = @($m.data | ForEach-Object { $_.id })
    if ($ids -notcontains $Arm) {
        throw "arm '$Arm' not served. Endpoint offers: $($ids -join ', ')"
    }
    Write-Host "[preflight] target endpoint OK, serving '$Arm'"
}
catch {
    throw "Target endpoint check failed: $($_.Exception.Message)"
}

# --- Preflight: the CLI must hold its own credential ------------------------
$authRaw = & claude auth status 2>$null | Out-String
Write-Host "[preflight] claude auth status: $($authRaw.Trim())"
if ($authRaw -notmatch '"loggedIn"\s*:\s*true') {
    throw @'
Claude Code CLI is not logged in, so the auditor/judge/realism roles cannot run
on the subscription. The account holder must run once:

    claude setup-token

This cannot be automated and must not be worked around by pointing subscription
OAuth credentials at the raw Messages API.
'@
}

$argList = @(
    'eval', 'inspect_petri/audit',
    '-T', 'seed_instructions=seeds',
    '-T', "max_turns=$MaxTurns",
    '-T', "realism_filter=$RealismFilter",
    '-T', 'enable_rollback=True',
    '-T', 'enable_prefill=False',
    '-T', 'compaction=True',
    '-T', 'target_tools=synthetic',
    '-T', 'judge_dimensions=configs/judge_dimensions.yaml',
    '--model-role', "auditor=$Auditor",
    '--model-role', "judge=$Judge",
    '--model-role', "realism=$Realism",
    '--model-role', "target=openai-api/vllm/$Arm",
    '--model', "openai-api/vllm/$Arm",
    '--epochs', "$Epochs",
    '--max-connections', "$MaxConnections",
    '--log-dir', $logDir,
    '--log-format', 'eval',
    '--temperature', '0.7',
    '--no-fail-on-error'
)
if ($SampleId) { $argList += @('--sample-id', $SampleId) }

$env:VLLM_BASE_URL = $TargetBaseUrl
$env:VLLM_API_KEY  = 'local-tunnel-no-auth-required'
$env:INSPECT_LOG_LEVEL = 'info'

Write-Host "[run] arm=$Arm epochs=$Epochs max_turns=$MaxTurns conc=$MaxConnections"
Write-Host "[run] inspect $($argList -join ' ')"

$started = Get-Date
& (Join-Path $root '.venv\Scripts\inspect.exe') @argList
$exit = $LASTEXITCODE
$elapsed = (Get-Date) - $started

Write-Host "[done] arm=$Arm exit=$exit elapsed=$([math]::Round($elapsed.TotalMinutes,1))min"
Write-Host "[done] log dir: $logDir"
if ($exit -ne 0) { throw "inspect eval failed for arm '$Arm' with exit code $exit" }
