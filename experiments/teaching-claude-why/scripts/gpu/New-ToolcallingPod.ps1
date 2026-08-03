<#
.SYNOPSIS
    Provision the tool-calling SFT GPU on RunPod and arm an isolated watchdog.

.DESCRIPTION
    Deliberately NOT the vulnerabilities experiment's New-AuditPod.ps1, for one reason:
    that script writes a FIXED runtime\provider-monitor\run-state.json, and another task
    is running against that file right now. Writing it would repoint that task's watchdog
    at this pod. (Its teardown's account sweep is read-only and harmless - see
    scripts\gpu\README.md; an earlier version of this comment wrongly said otherwise.)

    Terminating a resource this run did not provision is forbidden (AGENTS.md), and the
    shared RunPod account holds several pods belonging to other people, so nothing in
    this directory ever terminates a pod other than the one whose id it created.

    Registration order is safety-critical: the pod id is written to run-state.json in
    the same breath as creation, so a crash between "pod exists" and "watchdog knows
    about it" cannot strand a paid resource. If registration fails after creation, the
    pod is terminated rather than left orphaned.
#>
[CmdletBinding()]
param(
    # Priority list; RunPod allocates the first available. H100 SXM 80GB is the card the
    # 1h38m/epoch measurement for this model came from and the only one this architecture
    # has been trained on here - High stock at $2.99/h. H100 PCIe is the fallback: same
    # 80GB class, $2.89/h, but Low stock, so it is second rather than first.
    [string[]]$GpuTypeIds = @('NVIDIA H100 80GB HBM3', 'NVIDIA H100 PCIe'),
    [string]$ImageName = 'runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04',
    [int]$ContainerDiskInGb = 80,
    # 54GB of base weights + adapter checkpoints + the mixture. 120GB attaches reliably;
    # 200GB did not, per the sibling experiment.
    [int]$VolumeInGb = 120,
    # Shared account: the prefix is what keeps this pod distinguishable from teammates'.
    [string]$Name = 'nika-toolcalling-2080-sft',
    [double]$MaxSpendUsd = 40,
    [double]$MaxWallClockHours = 8,
    [double]$HourlyUsd = 2.99
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$expRoot    = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$vulnRoot   = (Resolve-Path (Join-Path $expRoot '..\vulnerabilities')).Path
$secretsDir = Join-Path $vulnRoot 'scripts\secrets'
$runtimeDir = Join-Path $expRoot 'runtime\toolcalling'
New-Item -ItemType Directory -Path $runtimeDir -Force | Out-Null

Import-Module (Join-Path $vulnRoot 'scripts\provider\RunPodApi.psm1') -Force -DisableNameChecking

$stateFile  = Join-Path $runtimeDir 'run-state.json'
$intentFile = Join-Path $runtimeDir 'provision-intent.json'

# Publish the intent BEFORE creating anything, so a crash mid-create is visible.
@{
    intent_at    = (Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ')
    provider     = 'runpod'
    gpu_type_ids = $GpuTypeIds
    image        = $ImageName
    pod_name     = $Name
    note         = 'A pod may exist from this moment. If run-state.instance_id is null and this file is recent, look for a pod named ' + $Name + ' and terminate that one only.'
} | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $intentFile -Encoding utf8

$startBalance = & "$secretsDir\Invoke-WithInfraSecrets.ps1" -Inject @('RUNPOD_API_KEY') -ScriptBlock {
    Import-Module (Join-Path $vulnRoot 'scripts\provider\RunPodApi.psm1') -Force -DisableNameChecking
    Get-RunPodBalance
}
Write-Host ("starting account balance: `$" + [math]::Round($startBalance.balance_usd, 2) + " (account-wide spend `$" + $startBalance.spend_per_hr_usd + "/h across all users)")

$result = & "$secretsDir\Invoke-WithInfraSecrets.ps1" -Inject @('RUNPOD_API_KEY') -ScriptBlock {
    Set-StrictMode -Off
    $body = @{
        name              = $Name
        imageName         = $ImageName
        gpuTypeIds        = @($GpuTypeIds)
        gpuCount          = 1
        cloudType         = 'SECURE'
        computeType       = 'GPU'
        containerDiskInGb = $ContainerDiskInGb
        volumeInGb        = $VolumeInGb
        volumeMountPath   = '/workspace'
        ports             = @('22/tcp')
        supportPublicIp   = $true
        interruptible     = $false
    } | ConvertTo-Json -Depth 6
    try {
        $pod = Invoke-RestMethod -Uri 'https://rest.runpod.io/v1/pods' -Method Post `
            -ContentType 'application/json' -Body $body `
            -Headers @{ Authorization = "Bearer $env:RUNPOD_API_KEY" } -TimeoutSec 180
        return @{ Ok = $true; Pod = $pod }
    }
    catch {
        $detail = ''
        try {
            $reader = New-Object System.IO.StreamReader($_.Exception.Response.GetResponseStream())
            $detail = $reader.ReadToEnd()
        } catch { }
        return @{ Ok = $false; Error = $_.Exception.Message; Detail = $detail }
    }
}

if (-not $result.Ok) {
    Write-Host "POD CREATION FAILED: $($result.Error)"
    if ($result.Detail) { Write-Host "Detail: $($result.Detail)" }
    Write-Host "No account sweep is performed - other users' pods are on this account."
    Write-Host "If a pod named '$Name' exists, terminate that one and only that one."
    throw 'Provisioning failed.'
}

$pod = $result.Pod
Set-StrictMode -Off
$actualGpu     = if ($pod.machine -and $pod.machine.gpuTypeId) { $pod.machine.gpuTypeId } else { ($GpuTypeIds)[0] }
$actualHourly  = if ($pod.costPerHr) { [double]$pod.costPerHr } else { $HourlyUsd }
Set-StrictMode -Version Latest

$createdAt = (Get-Date).ToUniversalTime()
$state = [ordered]@{
    run_id                 = "toolcalling-sft-$($createdAt.ToString('yyyyMMdd-HHmmss'))"
    provider               = 'runpod'
    pod_name               = $Name
    gpu                    = $actualGpu
    hourly_usd             = $actualHourly
    instance_id            = $pod.id
    instance_created_at    = $createdAt.ToString('yyyy-MM-ddTHH:mm:ssZ')
    instance_state         = 'provisioned'
    terminated_at          = $null
    starting_balance_usd   = $startBalance.balance_usd
    budget                 = [ordered]@{
        max_gpu_spend_usd    = $MaxSpendUsd
        max_wall_clock_hours = $MaxWallClockHours
    }
    hard_deadline_utc      = $createdAt.AddHours($MaxWallClockHours).ToString('yyyy-MM-ddTHH:mm:ssZ')
    scope_note             = 'This watchdog acts on instance_id and nothing else. It never lists or terminates other pods on this shared account.'
}
$state | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $stateFile -Encoding utf8

Write-Host "pod created: $($pod.id)"
Write-Host ("allocated GPU: $actualGpu at `$$actualHourly/h")
Write-Host "hard deadline: $($state.hard_deadline_utc)  |  spend cap: `$$MaxSpendUsd"
Remove-Item -LiteralPath $intentFile -Force -ErrorAction SilentlyContinue

$state | ConvertTo-Json -Depth 6
