# RunPodApi.psm1
#
# Thin client over RunPod's official authenticated APIs.
#
# Every function expects RUNPOD_API_KEY to already be present in the process
# environment, which means every caller must come through
# scripts/secrets/Invoke-WithInfraSecrets.ps1. No function here reads a secret
# file, prints a key, or writes one anywhere.
#
# RunPod exposes two surfaces and both are used deliberately:
#   * GraphQL  (api.runpod.io/graphql) - account balance, pod create/terminate.
#   * REST v1  (rest.runpod.io/v1)     - pod listing and status.

Set-StrictMode -Version Latest

function Get-RunPodAuthHeader {
    if (-not $env:RUNPOD_API_KEY) {
        throw 'RUNPOD_API_KEY is not present in this process. Call through Invoke-WithInfraSecrets.ps1.'
    }
    return @{ Authorization = "Bearer $($env:RUNPOD_API_KEY)" }
}

function Invoke-RunPodGraphQL {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][string]$Query,
        [hashtable]$Variables,
        [int]$TimeoutSec = 30
    )
    $payload = @{ query = $Query }
    if ($Variables) { $payload.variables = $Variables }
    $body = $payload | ConvertTo-Json -Depth 10 -Compress

    $res = Invoke-RestMethod -Uri 'https://api.runpod.io/graphql' -Method Post `
        -ContentType 'application/json' -Body $body -Headers (Get-RunPodAuthHeader) -TimeoutSec $TimeoutSec

    $errs = $res.PSObject.Properties.Match('errors')
    if ($errs.Count -gt 0 -and $res.errors) {
        throw "RunPod GraphQL error: $(($res.errors | ForEach-Object { $_.message }) -join '; ')"
    }
    return $res.data
}

function Get-RunPodBalance {
    <#
    .SYNOPSIS
        Exact provider-reported account balance in USD.
    .OUTPUTS
        PSCustomObject: balance_usd, spend_per_hr_usd, retrieved_at
    #>
    [CmdletBinding()]
    param()
    $d = Invoke-RunPodGraphQL -Query 'query { myself { clientBalance currentSpendPerHr } }'
    [pscustomobject]@{
        balance_usd      = [double]$d.myself.clientBalance
        spend_per_hr_usd = [double]$d.myself.currentSpendPerHr
        retrieved_at     = (Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ')
    }
}

function Get-RunPodPods {
    <#
    .SYNOPSIS
        All pods currently on the account. An empty array means no paid GPU.
    #>
    [CmdletBinding()]
    param([int]$TimeoutSec = 30)
    $r = Invoke-RestMethod -Uri 'https://rest.runpod.io/v1/pods' -Headers (Get-RunPodAuthHeader) -TimeoutSec $TimeoutSec
    if ($null -eq $r) { return @() }
    return @($r)
}

function Get-RunPodPod {
    <#
    .SYNOPSIS
        One pod by ID. Returns $null when the pod no longer exists, which is the
        expected result after a successful termination.
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][string]$PodId,
        [int]$TimeoutSec = 30
    )
    try {
        return Invoke-RestMethod -Uri "https://rest.runpod.io/v1/pods/$PodId" -Headers (Get-RunPodAuthHeader) -TimeoutSec $TimeoutSec
    }
    catch {
        $status = $null
        if ($_.Exception.PSObject.Properties.Match('Response').Count -gt 0 -and $_.Exception.Response) {
            try { $status = [int]$_.Exception.Response.StatusCode } catch { }
        }
        if ($status -eq 404) { return $null }
        throw
    }
}

function Stop-RunPodPodHard {
    <#
    .SYNOPSIS
        Terminate a pod permanently and verify it is gone.
    .DESCRIPTION
        This is the cleanup primitive. It is deliberately idempotent: an already
        absent pod is reported as success, so the watchdog and the normal
        shutdown path can both call it without racing.
    .OUTPUTS
        PSCustomObject: pod_id, action, verified_absent, attempts, timestamp, detail
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][string]$PodId,
        [int]$MaxAttempts = 5,
        [int]$VerifyDelaySeconds = 5
    )

    $detail = @()
    $attempt = 0

    while ($attempt -lt $MaxAttempts) {
        $attempt++
        try {
            # REST delete is the documented terminate path.
            Invoke-RestMethod -Uri "https://rest.runpod.io/v1/pods/$PodId" -Method Delete `
                -Headers (Get-RunPodAuthHeader) -TimeoutSec 45 | Out-Null
            $detail += "attempt ${attempt}: REST DELETE issued"
        }
        catch {
            $status = $null
            if ($_.Exception.PSObject.Properties.Match('Response').Count -gt 0 -and $_.Exception.Response) {
                try { $status = [int]$_.Exception.Response.StatusCode } catch { }
            }
            if ($status -eq 404) {
                $detail += "attempt ${attempt}: REST DELETE returned 404, pod already absent"
            }
            else {
                $detail += "attempt ${attempt}: REST DELETE failed ($status); trying GraphQL podTerminate"
                try {
                    Invoke-RunPodGraphQL -Query 'mutation Term($input: PodTerminateInput!) { podTerminate(input: $input) }' `
                        -Variables @{ input = @{ podId = $PodId } } -TimeoutSec 45 | Out-Null
                    $detail += "attempt ${attempt}: GraphQL podTerminate issued"
                }
                catch {
                    $detail += "attempt ${attempt}: GraphQL podTerminate failed"
                }
            }
        }

        Start-Sleep -Seconds $VerifyDelaySeconds

        $still = $null
        try { $still = Get-RunPodPod -PodId $PodId } catch { $still = 'unknown' }

        if ($null -eq $still) {
            return [pscustomobject]@{
                pod_id          = $PodId
                action          = 'terminated'
                verified_absent = $true
                attempts        = $attempt
                timestamp       = (Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ')
                detail          = $detail
            }
        }
        $detail += "attempt ${attempt}: pod still present after verify delay"
    }

    return [pscustomobject]@{
        pod_id          = $PodId
        action          = 'termination-unverified'
        verified_absent = $false
        attempts        = $attempt
        timestamp       = (Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ')
        detail          = $detail
    }
}

function Test-RunPodNoActivePods {
    <#
    .SYNOPSIS
        Shutdown verification: true when the account has no pod at all.
    #>
    [CmdletBinding()]
    param()
    $pods = Get-RunPodPods
    # Both naming conventions are emitted deliberately. Stop-AuditRun.ps1 reads
    # `no_active_pods` / `active_pod_ids`, this function originally returned only
    # `clear` / `pod_ids`, and the whole module runs under
    # Set-StrictMode -Version Latest, where reading a missing property THROWS.
    #
    # On 2026-07-31 that mismatch aborted a teardown at the account-sweep step,
    # so it never reached the final step that stands the monitor and watchdog
    # down. Two provider-monitor loops then polled the API every 240s for ~19h
    # against a pod terminated hours earlier. The pod itself was correctly
    # terminated first, so nothing was billed - but a teardown that throws
    # halfway is exactly the failure this script exists to prevent.
    $count = @($pods).Count
    $ids   = @($pods | ForEach-Object { $_.id })
    [pscustomobject]@{
        active_pod_count = $count
        pod_ids          = $ids
        active_pod_ids   = $ids
        clear            = ($count -eq 0)
        no_active_pods   = ($count -eq 0)
        checked_at       = (Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ')
    }
}

Export-ModuleMember -Function Get-RunPodAuthHeader, Invoke-RunPodGraphQL, Get-RunPodBalance,
    Get-RunPodPods, Get-RunPodPod, Stop-RunPodPodHard, Test-RunPodNoActivePods
