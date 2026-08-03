<#
.SYNOPSIS
    Isolated cleanup watchdog for the tool-calling SFT pod. Terminates that pod and no other.

.DESCRIPTION
    Runs as its own local process with its own copy of RUNPOD_API_KEY, reading only files
    on disk, so it keeps working if the orchestration crashes, SSH dies, or the training
    process exits unexpectedly.

    It terminates on either of two conditions:

      1. HARD DEADLINE   now > run-state.hard_deadline_utc
      2. BUDGET          elapsed hours x hourly rate >= max_gpu_spend_usd

    There is deliberately NO idle-shutdown condition: idle detection needs a heartbeat
    channel that would have to survive a long unattended training run to be safe, and a
    mis-fire would kill the run mid-epoch. The deadline and the budget ceiling are
    sufficient, and neither can mis-fire onto someone else's GPU.

    Every action is keyed to run-state.instance_id. If that field is null, the watchdog
    reports and does nothing - a missing id means nothing was provisioned, never that
    something should be hunted for.
#>
[CmdletBinding()]
param(
    [int]$CheckIntervalSeconds = 60
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Continue'

$expRoot    = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$vulnRoot   = (Resolve-Path (Join-Path $expRoot '..\vulnerabilities')).Path
$secretsDir = Join-Path $vulnRoot 'scripts\secrets'
$runtimeDir = Join-Path $expRoot 'runtime\toolcalling'
$stateFile  = Join-Path $runtimeDir 'run-state.json'
$wdFile     = Join-Path $runtimeDir 'watchdog.state.json'
$evidenceDir = Join-Path $runtimeDir 'evidence'
New-Item -ItemType Directory -Path $evidenceDir -Force | Out-Null

function Write-WatchdogState {
    param([string]$Status, [string]$Detail = '', $RunState = $null)
    [ordered]@{
        pid                = $PID
        status             = $Status
        detail             = $Detail
        last_check_utc     = (Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ')
        hard_deadline_utc  = $(if ($RunState) { $RunState.hard_deadline_utc } else { $null })
        instance_id        = $(if ($RunState) { $RunState.instance_id } else { $null })
        scope              = 'this pod only; never sweeps the account'
    } | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $wdFile -Encoding utf8
}

function Invoke-Termination {
    param([string]$Trigger, [string]$Detail, $RunState)

    $ts = (Get-Date).ToUniversalTime().ToString('yyyyMMdd-HHmmss')
    Write-Host "!! WATCHDOG TRIGGERED: $Trigger - $Detail"
    Write-WatchdogState -Status 'terminating' -Detail "$Trigger : $Detail" -RunState $RunState

    $podId = $RunState.instance_id
    $out = & "$secretsDir\Invoke-WithInfraSecrets.ps1" -Inject @('RUNPOD_API_KEY') -ScriptBlock {
        Import-Module (Join-Path $vulnRoot 'scripts\provider\RunPodApi.psm1') -Force -DisableNameChecking
        $r = $null
        try { $r = Stop-RunPodPodHard -PodId $podId }
        catch { $r = [pscustomobject]@{ pod_id = $podId; verified_absent = $false; detail = @($_.Exception.Message) } }
        $b = $null
        try { $b = Get-RunPodBalance } catch { }
        @{ termination = $r; balance = $b }
    }

    [ordered]@{
        timestamp_utc   = (Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ')
        trigger         = $Trigger
        detail          = $Detail
        pod_id          = $podId
        termination     = $out.termination
        balance_after   = $out.balance
        swept_account   = $false
        sweep_note      = 'Not swept here. The watchdog is the emergency path and does the minimum that stops the billing: terminate this id. Stop-ToolcallingRun.ps1 performs the read-only listing check on the normal path.'
    } | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $evidenceDir "watchdog-termination-$ts.json") -Encoding utf8

    Write-WatchdogState -Status 'terminated' -Detail "$Trigger : $Detail" -RunState $RunState
}

Write-Host "watchdog armed (pid $PID), checking every ${CheckIntervalSeconds}s"

while ($true) {
    if (-not (Test-Path $stateFile)) {
        Write-WatchdogState -Status 'waiting' -Detail 'no run-state.json yet'
        Start-Sleep -Seconds $CheckIntervalSeconds
        continue
    }

    $rs = Get-Content -LiteralPath $stateFile -Raw | ConvertFrom-Json
    Set-StrictMode -Off
    $podId = $rs.instance_id
    $terminatedAt = $rs.terminated_at
    Set-StrictMode -Version Latest

    if (-not $podId) {
        Write-WatchdogState -Status 'idle' -Detail 'run-state has no instance_id; nothing provisioned' -RunState $rs
        Start-Sleep -Seconds $CheckIntervalSeconds
        continue
    }
    if ($terminatedAt) {
        Write-WatchdogState -Status 'stood-down' -Detail "pod terminated at $terminatedAt" -RunState $rs
        Write-Host "pod already terminated; watchdog standing down"
        break
    }

    $created  = [datetime]::Parse($rs.instance_created_at).ToUniversalTime()
    $now      = (Get-Date).ToUniversalTime()
    $elapsedH = ($now - $created).TotalHours
    $spend    = $elapsedH * [double]$rs.hourly_usd
    $deadline = [datetime]::Parse($rs.hard_deadline_utc).ToUniversalTime()

    if ($now -gt $deadline) {
        Invoke-Termination -Trigger 'HARD_DEADLINE' -Detail "now $($now.ToString('s')) > deadline $($rs.hard_deadline_utc)" -RunState $rs
        break
    }
    if ($spend -ge [double]$rs.budget.max_gpu_spend_usd) {
        Invoke-Termination -Trigger 'BUDGET' -Detail ("estimated spend `$" + [math]::Round($spend,2) + " >= cap `$" + $rs.budget.max_gpu_spend_usd) -RunState $rs
        break
    }

    Write-WatchdogState -Status 'armed' -Detail ("elapsed " + [math]::Round($elapsedH,2) + "h, est spend `$" + [math]::Round($spend,2) + " of `$" + $rs.budget.max_gpu_spend_usd) -RunState $rs
    Start-Sleep -Seconds $CheckIntervalSeconds
}
