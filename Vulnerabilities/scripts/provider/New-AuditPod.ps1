<#
.SYNOPSIS
    Provision the audit GPU on RunPod Secure Cloud and register it with the
    monitor and cleanup watchdog IMMEDIATELY.

.DESCRIPTION
    Registration order is safety-critical. The pod is registered in run-state.json
    (which is what the watchdog reads) in the same breath as creation, so a crash
    between "pod exists" and "watchdog knows about it" cannot strand a paid
    resource. If registration fails after creation, the script terminates the pod
    it just made rather than leaving it orphaned.
#>
[CmdletBinding()]
param(
    [string]$GpuTypeId = 'NVIDIA A100 80GB PCIe',
    [string]$ImageName = 'runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04',
    [int]$ContainerDiskInGb = 60,
    [int]$VolumeInGb = 120,
    [string]$Name = 'msm-audit',
    [double]$HourlyUsd = 1.19
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$root       = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$secretsDir = Join-Path $root 'scripts\secrets'

Import-Module (Join-Path $root 'scripts\provider\RunPodApi.psm1')      -Force -DisableNameChecking
Import-Module (Join-Path $root 'scripts\provider\ProviderStatus.psm1') -Force -DisableNameChecking

# Publish the intent BEFORE creating anything, so a crash mid-create is visible.
$intentFile = Join-Path $root 'runtime\provider-monitor\provision-intent.json'
@{
    intent_at   = (Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ')
    provider    = 'runpod'
    gpu_type_id = $GpuTypeId
    image       = $ImageName
    note        = 'A pod may exist from this moment. If run-state.instance_id is null and this file is recent, sweep the account manually.'
} | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $intentFile -Encoding utf8

$result = & "$secretsDir\Invoke-WithInfraSecrets.ps1" -Inject @('RUNPOD_API_KEY') -ScriptBlock {
    Set-StrictMode -Off
    Import-Module (Join-Path $root 'scripts\provider\RunPodApi.psm1') -Force -DisableNameChecking

    $body = @{
        name              = $Name
        imageName         = $ImageName
        gpuTypeIds        = @($GpuTypeId)
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
            -Headers (Get-RunPodAuthHeader) -TimeoutSec 120
        return @{ Ok = $true; Pod = $pod }
    }
    catch {
        $detail = ''
        try {
            $stream = $_.Exception.Response.GetResponseStream()
            $reader = New-Object System.IO.StreamReader($stream)
            $detail = $reader.ReadToEnd()
        } catch { }
        return @{ Ok = $false; Error = $_.Exception.Message; Detail = $detail }
    }
}

if (-not $result.Ok) {
    Write-Host "POD CREATION FAILED: $($result.Error)"
    if ($result.Detail) { Write-Host "Detail: $($result.Detail)" }
    # Sweep in case the pod was created despite the error surfacing.
    $sweep = & "$secretsDir\Invoke-WithInfraSecrets.ps1" -Inject @('RUNPOD_API_KEY') -ScriptBlock {
        Import-Module (Join-Path $root 'scripts\provider\RunPodApi.psm1') -Force -DisableNameChecking
        Test-RunPodNoActivePods
    }
    Write-Host "Account sweep after failure: active_pod_count=$($sweep.active_pod_count)"
    if (-not $sweep.clear) { Write-Host "!! POD(S) PRESENT DESPITE FAILURE: $($sweep.pod_ids -join ', ') - terminate manually." }
    throw 'Provisioning failed.'
}

$pod = $result.Pod
Write-Host "pod created: $($pod.id)"

# --- Register with monitor + watchdog immediately -----------------------------
try {
    $state = Register-Instance -InstanceId $pod.id
    Write-Host "registered: hard deadline $($state.hard_deadline_utc)"
}
catch {
    Write-Host "!! REGISTRATION FAILED after pod creation. Terminating to avoid an orphan."
    & "$secretsDir\Invoke-WithInfraSecrets.ps1" -Inject @('RUNPOD_API_KEY') -ScriptBlock {
        Import-Module (Join-Path $root 'scripts\provider\RunPodApi.psm1') -Force -DisableNameChecking
        Stop-RunPodPodHard -PodId $pod.id
    } | ConvertTo-Json -Depth 6 | Write-Host
    throw
}

Remove-Item -LiteralPath $intentFile -Force -ErrorAction SilentlyContinue

# Take an activity lease covering provisioning + model download so the watchdog
# does not mistake a long startup for idleness.
& (Join-Path $root 'scripts\provider\Send-AuditHeartbeat.ps1') -Activity 'provisioning' -BusyMinutes 90 | Out-Null

@{
    created_at = (Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ')
    pod_id     = $pod.id
    gpu        = $GpuTypeId
    image      = $ImageName
    hourly_usd = $HourlyUsd
    volume_gb  = $VolumeInGb
    disk_gb    = $ContainerDiskInGb
} | ConvertTo-Json -Depth 5 |
    Set-Content -LiteralPath (Join-Path $root 'evidence\provider\pod-creation.json') -Encoding utf8

Write-Host 'Waiting for the pod to report a public SSH endpoint...'
$endpoint = & "$secretsDir\Invoke-WithInfraSecrets.ps1" -Inject @('RUNPOD_API_KEY') -ScriptBlock {
    Set-StrictMode -Off
    Import-Module (Join-Path $root 'scripts\provider\RunPodApi.psm1') -Force -DisableNameChecking
    for ($i = 0; $i -lt 60; $i++) {
        Start-Sleep -Seconds 10
        $p = Get-RunPodPod -PodId $pod.id
        if (-not $p) { continue }
        $status = $p.desiredStatus
        $ports  = $p.portMappings
        Write-Host ("  [{0:d2}] status={1} publicIp={2} ports={3}" -f $i, $status, $p.publicIp, ($(if ($ports) { ($ports.PSObject.Properties | ForEach-Object { "$($_.Name)->$($_.Value)" }) -join ',' } else { 'none' })))
        if ($p.publicIp -and $ports -and $ports.PSObject.Properties.Match('22').Count -gt 0) {
            return @{ Ip = $p.publicIp; Port = $ports.'22'; Raw = $p }
        }
    }
    return $null
}

if (-not $endpoint) {
    Write-Host '!! Pod did not expose SSH within 10 minutes. It is registered with the watchdog and will be reaped; investigate.'
    throw 'No SSH endpoint.'
}

Write-Host "SSH endpoint: root@$($endpoint.Ip):$($endpoint.Port)"
@{
    pod_id = $pod.id
    ssh_host = $endpoint.Ip
    ssh_port = $endpoint.Port
    recorded_at = (Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ')
} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $root 'runtime\provider-monitor\ssh-endpoint.json') -Encoding utf8

& (Join-Path $root 'scripts\provider\Update-ProviderStatus.ps1') -Reason 'post-provision'
