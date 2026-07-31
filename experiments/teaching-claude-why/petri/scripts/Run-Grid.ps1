<#
.SYNOPSIS
    Run every arm of the constitution audit in sequence, with health checks
    between arms and a spend check against the caps.

.DESCRIPTION
    One arm at a time, in a fixed order, so a failure is attributable to an arm
    rather than to a race between them. Between arms it verifies:

      1. the pod is still alive and vLLM still serves every arm
      2. the tunnel still answers
      3. the completed-sample count matches what was asked for
      4. the target actually participated (model events, not just auditor spin)
      5. projected spend still fits the GPU and API caps

    Any failed check stops the grid rather than letting later arms run against a
    broken target and produce clean-looking transcripts nobody can interpret.
    That exact failure - a target that never participated, in a log that looked
    complete - is what invalidated the sibling experiment's first pilot.

    The umbrella heartbeat keeper is expected to be running already; each arm's
    runner additionally starts and stops its own.

.EXAMPLE
    scripts\Run-Grid.ps1 -Epochs 2 -MaxConnections 4 -Tag grid
#>
[CmdletBinding()]
param(
    [int]$Epochs = 2,
    [int]$MaxConnections = 4,
    [string]$Tag = 'grid',
    [string[]]$Arms = @('base', 'dose-10-90', 'dose-20-80', 'dose-40-60'),
    [double]$GpuCapUsd = 80,

    # Must match the battery the runner actually uses, or the expected-sample
    # count - and therefore check_arm.py's integrity gate - is measured against
    # the wrong directory.
    [string]$SeedDir = 'seeds-v2',
    [string]$TargetBaseUrl = 'http://127.0.0.1:8000/v1'
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$vuln = (Resolve-Path (Join-Path $root '..\..\vulnerabilities')).Path
Set-Location $root

$seedCount = @(Get-ChildItem (Join-Path $root $SeedDir) -Filter *.md).Count
$expected  = $seedCount * $Epochs
Write-Host "[grid] $($Arms.Count) arms x $seedCount seeds x $Epochs epochs = $($Arms.Count * $expected) audits"
Write-Host "[grid] expecting $expected completed samples per arm"

function Test-Health {
    param([string]$Stage)
    # Endpoint must still offer every arm. A pod that died mid-grid would
    # otherwise let the remaining arms 'complete' with empty transcripts.
    try {
        $m = Invoke-RestMethod -Uri "$TargetBaseUrl/models" -TimeoutSec 20
        $ids = @($m.data | ForEach-Object { $_.id })
    } catch {
        throw "[$Stage] endpoint unreachable through the tunnel: $($_.Exception.Message)"
    }
    $missing = $Arms | Where-Object { $ids -notcontains $_ }
    if ($missing) { throw "[$Stage] endpoint no longer serves: $($missing -join ', ')" }
    Write-Host "[health] $Stage - endpoint OK, serves $($ids.Count) arms"
}

function Get-GpuSpend {
    $stateFile = Join-Path $vuln 'runtime\provider-monitor\status.json'
    if (-not (Test-Path $stateFile)) { return $null }
    try {
        $s = Get-Content -LiteralPath $stateFile -Raw | ConvertFrom-Json
        foreach ($k in @('estimated_infrastructure_cost_usd', 'estimated_cost_usd', 'gpu_spend_usd')) {
            if ($s.PSObject.Properties.Match($k).Count -gt 0) { return [double]$s.$k }
        }
    } catch { }
    return $null
}

$results = @()
foreach ($arm in $Arms) {
    Write-Host ""
    Write-Host "=========================================================="
    Write-Host "[grid] ARM $arm  ($([array]::IndexOf($Arms,$arm)+1) of $($Arms.Count))"
    Write-Host "=========================================================="

    Test-Health -Stage "before $arm"

    $spend = Get-GpuSpend
    if ($null -ne $spend -and $spend -ge $GpuCapUsd) {
        throw "[grid] GPU spend $spend USD has reached the cap $GpuCapUsd - stopping before $arm"
    }
    if ($null -ne $spend) { Write-Host "[spend] GPU so far: `$$([math]::Round($spend,2)) of `$$GpuCapUsd" }

    $started = Get-Date
    & (Join-Path $root 'scripts\Run-ConstitutionAudit.ps1') `
        -Arm $arm -Epochs $Epochs -MaxConnections $MaxConnections -Tag $Tag -SeedDir $SeedDir
    $mins = [math]::Round(((Get-Date) - $started).TotalMinutes, 1)

    # --- integrity check: did the arm really produce audits? ---------------
    $armLog = Join-Path $root "logs\$Tag\$arm"
    $check = & (Join-Path $root '.venv\Scripts\python.exe') `
        (Join-Path $root 'scripts\check_arm.py') $armLog $expected
    Write-Host $check
    if ($check -match 'ARM-CHECK FAIL') {
        throw "[grid] arm '$arm' failed its integrity check - stopping rather than running further arms"
    }

    $results += [pscustomobject]@{ arm = $arm; minutes = $mins }
    Write-Host "[grid] $arm done in $mins min"
}

Write-Host ""
Write-Host "[grid] ALL ARMS COMPLETE"
$results | Format-Table -AutoSize
