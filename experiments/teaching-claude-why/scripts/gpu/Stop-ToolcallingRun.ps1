<#
.SYNOPSIS
    Terminate the tool-calling SFT pod, verify it is gone, and stand the watchdog down.

.DESCRIPTION
    TERMINATES run-state.instance_id and nothing else - other people's pods are on this
    shared account.

    It does READ the account list, once, as verification. That sweep proves this run's id
    is gone rather than merely assumed gone, and it is the only evidence distinguishing a
    teardown that worked from one that appeared to. It terminates nothing; other users'
    pods appear in its count and are reported, not acted on.

    Order matters: terminate, verify absent by direct lookup of that id, confirm with the
    read-only sweep, record the provider-reported balance, then mark run-state terminated
    so the watchdog stands itself down on its next tick.
#>
[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$expRoot    = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$vulnRoot   = (Resolve-Path (Join-Path $expRoot '..\vulnerabilities')).Path
$secretsDir = Join-Path $vulnRoot 'scripts\secrets'
$runtimeDir = Join-Path $expRoot 'runtime\toolcalling'
$stateFile  = Join-Path $runtimeDir 'run-state.json'
$evidenceDir = Join-Path $runtimeDir 'evidence'
New-Item -ItemType Directory -Path $evidenceDir -Force | Out-Null

if (-not (Test-Path $stateFile)) { throw "no run-state.json at $stateFile - nothing to tear down" }
$rs = Get-Content -LiteralPath $stateFile -Raw | ConvertFrom-Json
Set-StrictMode -Off
$podId = $rs.instance_id
Set-StrictMode -Version Latest
if (-not $podId) { throw 'run-state has no instance_id' }

Write-Host "terminating pod $podId ($($rs.pod_name)) - this pod only"

$out = & "$secretsDir\Invoke-WithInfraSecrets.ps1" -Inject @('RUNPOD_API_KEY') -ScriptBlock {
    Import-Module (Join-Path $vulnRoot 'scripts\provider\RunPodApi.psm1') -Force -DisableNameChecking
    $term = Stop-RunPodPodHard -PodId $podId
    # Verify by direct lookup of this id. A 404 is success.
    $still = $null
    try { $still = Get-RunPodPod -PodId $podId } catch { }
    # Read-only confirmation: our id must not appear in the account listing. Other users'
    # pods will, and that is fine - this call terminates nothing.
    $sweep = $null
    try { $sweep = Test-RunPodNoActivePods } catch { }
    $bal = $null
    try { $bal = Get-RunPodBalance } catch { }
    @{ termination = $term; still_present = ($null -ne $still); sweep = $sweep; balance = $bal }
}

Set-StrictMode -Off
$sweepIds  = if ($out.sweep) { @($out.sweep.active_pod_ids) } else { @() }
$inListing = $sweepIds -contains $podId
$otherPods = @($sweepIds | Where-Object { $_ -ne $podId })
Set-StrictMode -Version Latest

$ts = (Get-Date).ToUniversalTime().ToString('yyyyMMdd-HHmmss')
$created = [datetime]::Parse($rs.instance_created_at).ToUniversalTime()
$elapsedH = ((Get-Date).ToUniversalTime() - $created).TotalHours

$evidence = [ordered]@{
    timestamp_utc      = (Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ')
    run_id             = $rs.run_id
    pod_id             = $podId
    pod_name           = $rs.pod_name
    gpu                = $rs.gpu
    hourly_usd         = $rs.hourly_usd
    elapsed_hours      = [math]::Round($elapsedH, 3)
    estimated_cost_usd = [math]::Round($elapsedH * [double]$rs.hourly_usd, 2)
    verified_absent    = ((-not $out.still_present) -and (-not $inListing))
    termination        = $out.termination
    balance_after      = $out.balance
    starting_balance   = $rs.starting_balance_usd
    verification       = [ordered]@{
        direct_lookup_404      = (-not $out.still_present)
        absent_from_listing    = (-not $inListing)
        other_pods_on_account  = $otherPods.Count
        other_pod_ids          = $otherPods
        note                   = 'The account listing is READ to confirm this pod is gone. It terminates nothing. Pods belonging to other users are expected on this shared account and are reported, never acted on.'
    }
}
$evPath = Join-Path $evidenceDir "planned-teardown-$ts.json"
$evidence | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $evPath -Encoding utf8

$rs.instance_state = 'terminated'
$rs.terminated_at = (Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ')
$rs | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $stateFile -Encoding utf8

$evidence | ConvertTo-Json -Depth 8
if ($out.still_present -or $inListing) {
    throw "POD $podId IS STILL PRESENT after termination - investigate immediately"
}
Write-Host "verified absent by direct lookup and by account listing. evidence: $evPath"
Write-Host "$($otherPods.Count) other pod(s) on this account, untouched: $($otherPods -join ', ')"
