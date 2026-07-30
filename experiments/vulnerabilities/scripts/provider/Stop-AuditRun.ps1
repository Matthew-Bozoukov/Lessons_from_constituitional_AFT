<#
.SYNOPSIS
    Planned teardown. Stops all work, terminates the paid GPU, and proves it.

.DESCRIPTION
    The watchdog is the safety net for unplanned failure. This is the deliberate
    path: it shuts things down in an order that cannot strand a paid resource,
    and it records evidence at every step.

    Order matters, and it is the opposite of what feels natural:

      1. Stop the work (Petri, SURF, heartbeat keepers, tunnel). Doing this
         first means nothing re-declares activity while we are terminating.
      2. Terminate the pod, idempotently, verifying absence.
      3. Sweep the WHOLE account for any other active pod, not just the one we
         think we own. A pod created by a failed provisioning attempt would not
         be in run-state, and is exactly the kind of thing that quietly bills.
      4. Record the provider-reported balance.
      5. Only then stand the watchdog down. Stopping the watchdog before
         termination is verified would remove the safety net at the one moment
         it is most needed.

    Safe to re-run. Every step tolerates having already been done.

.NOTES
    MUST be invoked through the infra secrets wrapper - it calls the provider API
    directly and RUNPOD_API_KEY is deliberately absent from the parent process:

        .\scripts\secrets\Invoke-WithInfraSecrets.ps1 -ScriptBlock {
            & .\scripts\provider\Stop-AuditRun.ps1
        }

    Run bare, it terminates nothing, fails at the account sweep and balance
    steps, and - correctly - refuses to stand the watchdog down. That refusal is
    the design working: an unverified teardown must never remove the safety net.

.PARAMETER KeepWatchdog
    Leave the watchdog running even after successful termination. Use when
    winding down one pod while intending to provision another.

.PARAMETER WhatIf
    Report what would be stopped without stopping anything.
#>
[CmdletBinding()]
param(
    [switch]$KeepWatchdog,
    [switch]$WhatIfOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Continue'

$root = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
Import-Module (Join-Path $root 'scripts\provider\RunPodApi.psm1')      -Force -DisableNameChecking
Import-Module (Join-Path $root 'scripts\provider\ProviderStatus.psm1') -Force -DisableNameChecking

$ts          = (Get-Date).ToUniversalTime().ToString('yyyyMMdd-HHmmss')
$evidenceDir = Join-Path $root 'evidence\cleanup'
if (-not (Test-Path $evidenceDir)) { New-Item -ItemType Directory -Path $evidenceDir -Force | Out-Null }

$steps = [System.Collections.ArrayList]@()
function Add-Step {
    param([string]$Name, [string]$Detail, [bool]$Ok = $true)
    [void]$steps.Add([ordered]@{
        step = $Name; detail = $Detail; ok = $Ok
        at   = (Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ')
    })
    $mark = if ($Ok) { 'ok  ' } else { 'FAIL' }
    Write-Host "  [$mark] $Name - $Detail"
}

Write-Host ''
Write-Host '=============================================================='
Write-Host " MSM audit - planned teardown  ($ts UTC)"
if ($WhatIfOnly) { Write-Host ' WHAT-IF: nothing will actually be stopped' }
Write-Host '=============================================================='

# -- 1. stop the work ---------------------------------------------------------
Write-Host ''
Write-Host '1. stopping work processes'

# Heartbeat keepers first: while one lives it keeps declaring activity.
$keepers = @(Get-CimInstance Win32_Process -Filter "Name = 'powershell.exe'" -ErrorAction SilentlyContinue |
    Where-Object {
        $_.ProcessId -ne $PID -and $_.CommandLine -and $_.CommandLine -match 'EncodedCommand'
    } |
    Where-Object {
        try {
            $d = [Text.Encoding]::Unicode.GetString(
                [Convert]::FromBase64String((($_.CommandLine -split '-EncodedCommand\s+')[-1]).Trim()))
            $d -match 'keeper.*\.stop'
        } catch { $false }
    })
if ($WhatIfOnly) { Add-Step 'heartbeat-keepers' "$($keepers.Count) would be stopped" }
else {
    foreach ($k in $keepers) { Stop-Process -Id $k.ProcessId -Force -ErrorAction SilentlyContinue }
    Add-Step 'heartbeat-keepers' "stopped $($keepers.Count)"
}

# Petri (inspect) and SURF / fixed-eval (python), plus the tunnel supervisor.
$work = @(Get-Process inspect, python -ErrorAction SilentlyContinue)
if ($WhatIfOnly) { Add-Step 'work-processes' "$($work.Count) inspect/python would be stopped" }
else {
    foreach ($p in $work) { Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue }
    Add-Step 'work-processes' "stopped $($work.Count) inspect/python"
}

$tunnels = @(Get-CimInstance Win32_Process -Filter "Name = 'powershell.exe'" -ErrorAction SilentlyContinue |
    Where-Object { $_.CommandLine -and $_.CommandLine -match 'Start-Tunnel' })
$ssh = @(Get-Process ssh -ErrorAction SilentlyContinue)
if ($WhatIfOnly) { Add-Step 'tunnel' "$($tunnels.Count) supervisor + $($ssh.Count) ssh would be stopped" }
else {
    foreach ($t in $tunnels) { Stop-Process -Id $t.ProcessId -Force -ErrorAction SilentlyContinue }
    foreach ($s in $ssh)     { Stop-Process -Id $s.Id -Force -ErrorAction SilentlyContinue }
    Add-Step 'tunnel' "stopped $($tunnels.Count) supervisor + $($ssh.Count) ssh"
}

# -- 2. terminate the pod -----------------------------------------------------
Write-Host ''
Write-Host '2. terminating the paid resource'

$runState = $null
try { $runState = Get-RunState } catch { }
$podId = if ($runState -and $runState.instance_id) { $runState.instance_id } else { $null }

$termination = $null
if (-not $podId) {
    Add-Step 'terminate' 'no instance registered in run-state'
}
elseif ($WhatIfOnly) {
    Add-Step 'terminate' "pod $podId would be terminated"
}
else {
    try {
        $termination = Stop-RunPodPodHard -PodId $podId
        Add-Step 'terminate' "pod $podId : $($termination.action), verified_absent=$($termination.verified_absent)" ([bool]$termination.verified_absent)
    }
    catch {
        Add-Step 'terminate' "pod $podId : threw - $($_.Exception.Message)" $false
    }
}

# -- 3. sweep the whole account ----------------------------------------------
Write-Host ''
Write-Host '3. sweeping the account for any other active pod'
$sweep = $null
if ($WhatIfOnly) { Add-Step 'account-sweep' 'would sweep' }
else {
    try {
        $sweep = Test-RunPodNoActivePods
        $clean = [bool]$sweep.no_active_pods
        Add-Step 'account-sweep' $(if ($clean) { 'no active pods on the account' }
                                   else { "STILL ACTIVE: $($sweep.active_pod_ids -join ', ')" }) $clean
    }
    catch { Add-Step 'account-sweep' "threw - $($_.Exception.Message)" $false }
}

# -- 4. balance ---------------------------------------------------------------
Write-Host ''
Write-Host '4. recording provider-reported balance'
$balance = $null
if ($WhatIfOnly) { Add-Step 'balance' 'would query' }
else {
    try {
        $balance = Get-RunPodBalance
        Add-Step 'balance' "$([math]::Round($balance.balance_usd,4)) USD (exact-provider-reported)"
    }
    catch { Add-Step 'balance' "threw - $($_.Exception.Message)" $false }
}

# -- 5. stand the watchdog down, only once termination is verified ------------
Write-Host ''
Write-Host '5. watchdog'
$terminationVerified = ($termination -and $termination.verified_absent) -or (-not $podId)
$accountClean        = ($sweep -and $sweep.no_active_pods)

if ($WhatIfOnly) {
    Add-Step 'watchdog' 'would stand down after verification'
}
elseif ($KeepWatchdog) {
    Add-Step 'watchdog' 'left running by request (-KeepWatchdog)'
}
elseif ($terminationVerified -and $accountClean) {
    try {
        $rs = Get-RunState
        if ($rs) {
            $rs.terminated_at  = (Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ')
            $rs.instance_state = 'terminated-by-planned-teardown'
            Set-RunState -State $rs
        }
    } catch { }
    $wd = @(Get-CimInstance Win32_Process -Filter "Name = 'powershell.exe'" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -and $_.CommandLine -match 'Watchdog-Loop' })
    foreach ($w in $wd) { Stop-Process -Id $w.ProcessId -Force -ErrorAction SilentlyContinue }
    $mon = @(Get-CimInstance Win32_Process -Filter "Name = 'powershell.exe'" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -and $_.CommandLine -match 'Monitor-Loop' })
    foreach ($m in $mon) { Stop-Process -Id $m.ProcessId -Force -ErrorAction SilentlyContinue }
    Add-Step 'watchdog' "stood down ($($wd.Count) watchdog, $($mon.Count) monitor stopped) after verified termination"
}
else {
    Add-Step 'watchdog' 'LEFT RUNNING: termination or sweep unverified, safety net retained' $false
}

# -- evidence -----------------------------------------------------------------
$evidence = [ordered]@{
    event                 = 'planned-teardown'
    what_if               = [bool]$WhatIfOnly
    timestamp_utc         = (Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ')
    run_id                = $(if ($runState) { $runState.run_id } else { $null })
    instance_id           = $podId
    termination_result    = $termination
    account_sweep         = $sweep
    final_balance_usd     = $(if ($balance) { $balance.balance_usd } else { $null })
    final_balance_basis   = $(if ($balance) { 'exact-provider-reported' } else { 'unavailable' })
    termination_verified  = $terminationVerified
    account_clean         = $accountClean
    steps                 = $steps
}
$outFile = Join-Path $evidenceDir "planned-teardown-$ts.json"
if (-not $WhatIfOnly) {
    $evidence | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $outFile -Encoding utf8
}

Write-Host ''
Write-Host '=============================================================='
if ($WhatIfOnly) {
    Write-Host ' WHAT-IF complete. Nothing was stopped.'
}
elseif ($terminationVerified -and $accountClean) {
    Write-Host ' TEARDOWN COMPLETE - no active paid resources on the account.'
    Write-Host " evidence: $(Resolve-Path -Relative $outFile)"
}
else {
    Write-Host ' TEARDOWN INCOMPLETE - CHECK THE PROVIDER CONSOLE.'
    Write-Host ' The watchdog has deliberately been left running.'
    Write-Host " evidence: $outFile"
}
Write-Host '=============================================================='

$evidence
