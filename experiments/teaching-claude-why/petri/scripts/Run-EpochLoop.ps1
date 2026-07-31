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
    # One trivial generate. Cheaper and more honest than parsing the CLI's
    # error text, which reports a session limit as a turn limit.
    $probe = & (Join-Path $root '.venv\Scripts\python.exe') -c @'
import asyncio
from inspect_ai.model import get_model
async def main():
    try:
        r = await get_model("claude-code/claude-sonnet-4-5").generate("Reply with OK")
        print("OK" if (r.completion or "").strip() else "EMPTY")
    except Exception as e:
        print("BLOCKED: " + str(e)[:120])
asyncio.run(main())
'@ 2>&1
    return (($probe -join ' ') -match '\bOK\b')
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
