<#
.SYNOPSIS
    Independent cleanup watchdog. Terminates the paid GPU no matter what else dies.

.DESCRIPTION
    Runs as its own local process with its own copy of RUNPOD_API_KEY in its
    environment block. It reads only files on disk, so it keeps working when
    Claude Code crashes, the SSH connection fails, vLLM crashes, Petri crashes,
    or the orchestration script exits unexpectedly. It never depends on an
    interactive SSH session to clean up.

    It terminates on any of three conditions:

      1. HARD DEADLINE   now > run-state.hard_deadline_utc
      2. BUDGET          locally-calculated infrastructure cost >= MAX_GPU_SPEND_USD
      3. GENUINE IDLE    no declared activity for IDLE_SHUTDOWN_MINUTES

    Condition 3 is deliberately conservative. Activity is *declared* by the
    orchestration through heartbeat.json, which carries both a heartbeat
    timestamp and an optional `busy_until_utc` lease. A long model download,
    model load, Petri audit, inference call or report-generation pass takes a
    lease covering its expected duration, so slow work is never mistaken for
    inactivity. The watchdog only counts idle time once every lease has expired
    AND the heartbeat has gone stale.

    Fail-safe direction: if heartbeat.json is missing entirely the watchdog does
    NOT terminate on idle grounds, because a missing file means the orchestration
    never started reporting rather than that it went idle. The hard deadline and
    the budget ceiling still apply and are never suspended.
#>
[CmdletBinding()]
param(
    [int]$CheckIntervalSeconds = 60
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Continue'

$root = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
Import-Module (Join-Path $root 'scripts\provider\RunPodApi.psm1')      -Force -DisableNameChecking
Import-Module (Join-Path $root 'scripts\provider\ProviderStatus.psm1') -Force -DisableNameChecking

$stateFile     = Join-Path $PSScriptRoot 'watchdog.state.json'
$heartbeatFile = Join-Path $PSScriptRoot 'heartbeat.json'
$evidenceDir   = Join-Path $root 'evidence\cleanup'
if (-not (Test-Path $evidenceDir)) { New-Item -ItemType Directory -Path $evidenceDir -Force | Out-Null }

function Write-WatchdogState {
    param([string]$Status, [string]$Detail = '', $RunState = $null)
    $obj = [ordered]@{
        pid                = $PID
        status             = $Status
        detail             = $Detail
        last_heartbeat_utc = (Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ')
        hard_deadline_utc  = $(if ($RunState) { $RunState.hard_deadline_utc } else { $null })
        instance_id        = $(if ($RunState) { $RunState.instance_id } else { $null })
    }
    $obj | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $stateFile -Encoding utf8
}

function Invoke-WatchdogTermination {
    param([string]$Trigger, [string]$Detail, $RunState)

    $ts = (Get-Date).ToUniversalTime().ToString('yyyyMMdd-HHmmss')
    Write-Host "!! WATCHDOG TRIGGERED: $Trigger - $Detail"
    Write-WatchdogState -Status 'terminating' -Detail "$Trigger : $Detail" -RunState $RunState

    $result = $null
    if ($RunState.instance_id) {
        try { $result = Stop-RunPodPodHard -PodId $RunState.instance_id }
        catch { $result = [pscustomobject]@{ pod_id = $RunState.instance_id; action = 'termination-threw'; verified_absent = $false; detail = @($_.Exception.Message) } }
    }

    $sweep = $null
    try { $sweep = Test-RunPodNoActivePods } catch { }

    $balance = $null
    try { $balance = Get-RunPodBalance } catch { }

    $evidence = [ordered]@{
        event                = 'watchdog-termination'
        trigger              = $Trigger
        detail               = $Detail
        timestamp_utc        = (Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ')
        run_id               = $RunState.run_id
        instance_id          = $RunState.instance_id
        termination_result   = $result
        account_sweep        = $sweep
        final_balance_usd    = $(if ($balance) { $balance.balance_usd } else { $null })
        final_balance_basis  = $(if ($balance) { 'exact-provider-reported' } else { 'unavailable' })
    }
    $evidence | ConvertTo-Json -Depth 8 |
        Set-Content -LiteralPath (Join-Path $evidenceDir "watchdog-termination-$ts.json") -Encoding utf8

    # Record termination in run-state so the monitor stops billing elapsed time.
    try {
        $rs = Get-RunState
        if ($rs) {
            $rs.terminated_at  = (Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ')
            $rs.instance_state = 'terminated-by-watchdog'
            Set-RunState -State $rs
        }
    } catch { }

    if ($result -and -not $result.verified_absent) {
        Write-Host '!! TERMINATION COULD NOT BE VERIFIED. Escalating in watchdog state file.'
        Write-WatchdogState -Status 'TERMINATION-UNVERIFIED' -Detail "Pod $($RunState.instance_id) may still be running. Check the provider console." -RunState $RunState
    } else {
        Write-WatchdogState -Status 'terminated' -Detail "$Trigger : verified absent" -RunState $RunState
    }
}

Write-Host ''
Write-Host '=============================================================='
Write-Host ' MSM audit - cleanup watchdog (independent)'
Write-Host " PID $PID   check every ${CheckIntervalSeconds}s"
Write-Host '=============================================================='

Write-WatchdogState -Status 'starting'

try {
    while ($true) {
        $runState = $null
        try { $runState = Get-RunState } catch { }

        if (-not $runState -or -not $runState.instance_id) {
            Write-WatchdogState -Status 'armed-no-instance' -Detail 'No paid resource registered yet.' -RunState $runState
            Start-Sleep -Seconds $CheckIntervalSeconds
            continue
        }

        if ($runState.terminated_at) {
            Write-WatchdogState -Status 'stood-down' -Detail 'Instance already terminated.' -RunState $runState
            Write-Host "Instance already terminated at $($runState.terminated_at). Watchdog standing down."
            break
        }

        $now = (Get-Date).ToUniversalTime()

        # -- 1. hard deadline ------------------------------------------------
        if ($runState.hard_deadline_utc) {
            $deadline = [datetime]::Parse($runState.hard_deadline_utc).ToUniversalTime()
            if ($now -gt $deadline) {
                Invoke-WatchdogTermination -Trigger 'hard-wall-clock-deadline' `
                    -Detail "Deadline $($runState.hard_deadline_utc) passed." -RunState $runState
                break
            }
        }

        # -- 2. budget ceiling -----------------------------------------------
        $elapsedHours = 0.0
        if ($runState.instance_created_at) {
            $elapsedHours = ($now - [datetime]::Parse($runState.instance_created_at).ToUniversalTime()).TotalHours
            if ($elapsedHours -lt 0) { $elapsedHours = 0 }
        }
        $hourly = [double]$runState.hourly_usd + [double]$runState.storage_hourly_usd
        $cost   = $elapsedHours * $hourly
        if ($cost -ge [double]$runState.budget.max_gpu_spend_usd) {
            Invoke-WatchdogTermination -Trigger 'gpu-budget-exhausted' `
                -Detail ("Estimated cost {0:N2} USD reached cap {1:N2} USD." -f $cost, $runState.budget.max_gpu_spend_usd) -RunState $runState
            break
        }

        # -- 3. genuine idle -------------------------------------------------
        $idleDetail = 'heartbeat fresh'
        if (Test-Path -LiteralPath $heartbeatFile) {
            try {
                $hb = Get-Content -LiteralPath $heartbeatFile -Raw | ConvertFrom-Json

                $leaseActive = $false
                if ($hb.PSObject.Properties.Match('busy_until_utc').Count -gt 0 -and $hb.busy_until_utc) {
                    $leaseActive = ((([datetime]::Parse($hb.busy_until_utc)).ToUniversalTime()) -gt $now)
                }

                $sinceBeat = ($now - ([datetime]::Parse($hb.last_heartbeat_utc)).ToUniversalTime()).TotalMinutes
                $idleLimit = [double]$runState.budget.idle_shutdown_minutes

                if ($leaseActive) {
                    $idleDetail = "activity lease held until $($hb.busy_until_utc) by '$($hb.activity)'"
                }
                elseif ($sinceBeat -gt $idleLimit) {
                    Invoke-WatchdogTermination -Trigger 'idle-timeout' `
                        -Detail ("No declared activity for {0:N1} min (limit {1} min); last activity '{2}', no lease held." -f $sinceBeat, $idleLimit, $hb.activity) -RunState $runState
                    break
                }
                else {
                    $idleDetail = ("last activity '{0}' {1:N1} min ago (limit {2} min)" -f $hb.activity, $sinceBeat, $idleLimit)
                }
            }
            catch {
                $idleDetail = 'heartbeat unreadable; idle rule suspended, deadline and budget still enforced'
            }
        }
        else {
            $idleDetail = 'no heartbeat file yet; idle rule suspended, deadline and budget still enforced'
        }

        $remaining = if ($runState.hard_deadline_utc) {
            '{0:N2} h to deadline' -f ([datetime]::Parse($runState.hard_deadline_utc).ToUniversalTime() - $now).TotalHours
        } else { 'no deadline set' }

        Write-WatchdogState -Status 'armed' -Detail "$idleDetail; $remaining; est cost `$$([math]::Round($cost,2))" -RunState $runState
        Write-Host ("[{0}] armed | pod {1} | est `${2:N2} of `${3:N2} | {4} | {5}" -f `
            (Get-Date -Format HH:mm:ss), $runState.instance_id, $cost, $runState.budget.max_gpu_spend_usd, $remaining, $idleDetail)

        Start-Sleep -Seconds $CheckIntervalSeconds
    }
}
finally {
    Write-Host 'Watchdog loop exited.'
}
