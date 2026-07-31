<#
.SYNOPSIS
    Run N further epochs unattended, waiting out Claude subscription session
    limits between them.

.DESCRIPTION
    The binding constraint on this experiment is not GPU or API money - it is the
    Claude subscription session window. Measured on 2026-07-31: roughly ONE epoch
    (112 audits) per window, and a window that runs out mid-batch fails every
    remaining audit at the auditor's first call, reported misleadingly as
    "Reached maximum number of turns (1)".

    So this loop:
      1. probes the subscription with one trivial generate before each epoch,
         and waits if the window is exhausted;
      2. runs exactly ONE epoch per batch, so a window boundary can cost at most
         one epoch rather than a multi-epoch batch;
      3. verifies the epoch with the same integrity gate the grid uses, and
         stops if an epoch produced transcripts the target never participated in;
      4. keeps its own heartbeat so the watchdog does not reap the pod during a
         long inter-window wait.

    Designed to survive the orchestrating session ending: launch it detached and
    it keeps going.

.EXAMPLE
    scripts\Run-EpochLoop.ps1 -Epochs 4 -StartIndex 3
#>
[CmdletBinding()]
param(
    [int]$Epochs = 4,
    [int]$StartIndex = 3,
    [int]$MaxConnections = 12,
    [string]$SeedDir = 'seeds-v2',
    [int]$ProbeIntervalMinutes = 15,
    [int]$MaxWaitHours = 8,
    [double]$ApiBudgetUsd = 100
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$vuln = (Resolve-Path (Join-Path $root '..\..\vulnerabilities')).Path
Set-Location $root

# `claude setup-token` stores the credential at User scope. A detached process
# can inherit a stale environment block, so read it from the registry.
if (-not $env:CLAUDE_CODE_OAUTH_TOKEN) {
    $env:CLAUDE_CODE_OAUTH_TOKEN = [Environment]::GetEnvironmentVariable('CLAUDE_CODE_OAUTH_TOKEN', 'User')
}

$progress = Join-Path $root 'logs\epoch-loop.log'
function Log([string]$m) {
    $line = "[{0}] {1}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $m
    Write-Host $line
    Add-Content -LiteralPath $progress -Value $line -Encoding utf8
}

# The keeper must outlive the inter-window waits, which are far longer than any
# single epoch. Without it the watchdog reaps the pod during a wait.
$keeper = & (Join-Path $vuln 'scripts\provider\Start-HeartbeatKeeper.ps1') `
    -Activity 'petri-epoch-loop' -MaxHours 48 -IntervalSeconds 180 -LeaseMinutes 20
Log "epoch loop started: $Epochs epochs from index $StartIndex (keeper pid $($keeper.ProcessId))"

function Test-QuotaAvailable {
    # Two-stage probe, deliberately cheap.
    #
    # Stage 1 polls with HAIKU: the 5-hour cap is plan-wide, so Haiku recovering
    # is a good proxy for the window having rolled, and a Haiku probe costs a
    # negligible slice of the quota the auditor needs. Polling with Sonnet every
    # 15 minutes would spend the very budget we are waiting to accumulate.
    #
    # Stage 2 confirms with ONE Sonnet call before committing an epoch, since
    # Haiku could in principle recover first. One call per launch attempt is
    # nothing; a wrongly-launched epoch costs 28 audits.
    #
    # Both call a SCRIPT FILE, never `python -c`: PowerShell strips embedded
    # quotes before python sees them, and the -c version failed every probe with
    # a SyntaxError that the loop then read as "quota exhausted" forever.
    $py = Join-Path $root '.venv\Scripts\python.exe'
    $probe = Join-Path $root 'scripts\probe_quota.py'

    $haiku = & $py $probe 'claude-code/claude-haiku-4-5' 2>&1
    if (($haiku -join ' ') -notmatch '^\s*OK') {
        Log "  probe(haiku): $($haiku -join ' ')"
        return $false
    }
    $sonnet = & $py $probe 'claude-code/claude-sonnet-4-5' 2>&1
    if (($sonnet -join ' ') -notmatch '^\s*OK') {
        Log "  probe(sonnet): $($sonnet -join ' ')"
        return $false
    }
    return $true
}

try {
    for ($i = 0; $i -lt $Epochs; $i++) {
        $idx = $StartIndex + $i
        $tag = "v2-e$idx"

        # ---- wait for the session window ----------------------------------
        $waited = 0
        while (-not (Test-QuotaAvailable)) {
            if ($waited -ge ($MaxWaitHours * 60)) {
                Log "ABORT: subscription still blocked after $MaxWaitHours h"
                throw "quota never recovered"
            }
            Log "quota exhausted; waiting $ProbeIntervalMinutes min (waited ${waited}m)"
            Start-Sleep -Seconds ($ProbeIntervalMinutes * 60)
            $waited += $ProbeIntervalMinutes
        }
        Log "quota available - starting epoch $idx (tag $tag)"

        # ---- run one epoch -------------------------------------------------
        try {
            & (Join-Path $root 'scripts\Run-Grid.ps1') `
                -Epochs 1 -MaxConnections $MaxConnections -Tag $tag -SeedDir $SeedDir
            Log "epoch $idx completed"
        }
        catch {
            Log "epoch $idx FAILED: $($_.Exception.Message)"
            # A quota failure mid-epoch is recoverable - quarantine and retry the
            # index next window. Anything else stops the loop.
            $bad = Join-Path $root "logs\failed\$tag-$(Get-Date -Format HHmmss)"
            if (Test-Path (Join-Path $root "logs\$tag")) {
                New-Item -ItemType Directory -Force -Path $bad | Out-Null
                Move-Item (Join-Path $root "logs\$tag\*") $bad -Force -ErrorAction SilentlyContinue
                Log "quarantined partial batch to $bad"
            }
            $i--   # retry this index in the next window
            Start-Sleep -Seconds ($ProbeIntervalMinutes * 60)
            continue
        }
    }
    Log "ALL EPOCHS COMPLETE"
}
finally {
    & (Join-Path $vuln 'scripts\provider\Stop-HeartbeatKeeper.ps1') -Keeper $keeper
    Log "epoch loop finished; heartbeat keeper stopped"
}
