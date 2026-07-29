<#
.SYNOPSIS
    Capture the raw provider API responses behind docs/01-provider-comparison.md.

.DESCRIPTION
    Queries the official authenticated APIs only. Account records are reduced to
    the billing-relevant fields; no email, address, key material or other
    personal account content is written to disk.
#>
[CmdletBinding()]
param(
    [string]$EvidenceDir = (Join-Path $PSScriptRoot '..\..\evidence\provider')
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$secretsDir  = (Resolve-Path (Join-Path $PSScriptRoot '..\secrets')).Path
$EvidenceDir = (Resolve-Path $EvidenceDir).Path
$ts = (Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ')

# ----------------------------------------------------------- Vast offers ----
$vastOffers = & "$secretsDir\Invoke-WithInfraSecrets.ps1" -Inject @('VAST_API_KEY') -ScriptBlock {
    Set-StrictMode -Off
    $qObj = @{
        verified          = @{ eq = $true }
        rentable          = @{ eq = $true }
        num_gpus          = @{ eq = 1 }
        gpu_ram           = @{ gte = 79000 }
        reliability2      = @{ gte = 0.995 }
        disk_space        = @{ gte = 150 }
        direct_port_count = @{ gte = 2 }
        datacenter        = @{ eq = $true }
        type              = 'on-demand'
        order             = @( ,@('dph_total','asc') )
        limit             = 50
    }
    $r = Invoke-RestMethod -Uri 'https://console.vast.ai/api/v0/search/asks/' -Method Put `
        -ContentType 'application/json' -Body (@{ q = $qObj } | ConvertTo-Json -Depth 10) `
        -Headers @{ Authorization = "Bearer $($env:VAST_API_KEY)" } -TimeoutSec 60
    $r.offers | Select-Object id, gpu_name, gpu_ram, num_gpus, dph_total, storage_cost,
        reliability2, inet_down, inet_up, disk_space, cuda_max_good, geolocation,
        verified, datacenter, direct_port_count, rentable
}
@{ captured_at = $ts; filter = 'verified datacenter, on-demand, 1 GPU, >=79GB VRAM, reliability>=0.995, disk>=150GB, direct ports>=2'; offer_count = @($vastOffers).Count; offers = $vastOffers } |
    ConvertTo-Json -Depth 8 | Out-File (Join-Path $EvidenceDir 'vast-offers.json') -Encoding utf8

# --------------------------------------------------------- RunPod offers ----
$runpodTypes = & "$secretsDir\Invoke-WithInfraSecrets.ps1" -Inject @('RUNPOD_API_KEY') -ScriptBlock {
    Set-StrictMode -Off
    $q = '{"query":"query { gpuTypes { id displayName memoryInGb secureCloud communityCloud maxGpuCount lowestPrice(input: {gpuCount: 1}) { uninterruptablePrice minimumBidPrice } } }"}'
    $r = Invoke-RestMethod -Uri 'https://api.runpod.io/graphql' -Method Post -ContentType 'application/json' `
        -Body $q -Headers @{ Authorization = "Bearer $($env:RUNPOD_API_KEY)" } -TimeoutSec 60
    $r.data.gpuTypes | Select-Object id, displayName, memoryInGb, secureCloud, communityCloud, maxGpuCount,
        @{n='onDemandPrice';e={$_.lowestPrice.uninterruptablePrice}},
        @{n='spotPrice';e={$_.lowestPrice.minimumBidPrice}}
}
@{ captured_at = $ts; source = 'RunPod GraphQL gpuTypes'; types = $runpodTypes } |
    ConvertTo-Json -Depth 8 | Out-File (Join-Path $EvidenceDir 'runpod-gpu-types.json') -Encoding utf8

# --------------------------------------------------------- Account state ----
# Billing-relevant fields only. No email, address, or key material.
$vastAcct = & "$secretsDir\Invoke-WithInfraSecrets.ps1" -Inject @('VAST_API_KEY') -ScriptBlock {
    Set-StrictMode -Off
    $r = Invoke-RestMethod -Uri 'https://console.vast.ai/api/v0/users/current/' `
        -Headers @{ Authorization = "Bearer $($env:VAST_API_KEY)" } -TimeoutSec 30
    @{ balance = $r.balance; credit = $r.credit; total_spend = $r.total_spend
       can_pay = $r.can_pay; has_billing = $r.has_billing; has_rented = $r.has_rented }
}
$runpodAcct = & "$secretsDir\Invoke-WithInfraSecrets.ps1" -Inject @('RUNPOD_API_KEY') -ScriptBlock {
    Set-StrictMode -Off
    $r = Invoke-RestMethod -Uri 'https://api.runpod.io/graphql' -Method Post -ContentType 'application/json' `
        -Body '{"query":"query { myself { clientBalance currentSpendPerHr } }"}' `
        -Headers @{ Authorization = "Bearer $($env:RUNPOD_API_KEY)" } -TimeoutSec 30
    @{ clientBalance = $r.data.myself.clientBalance; currentSpendPerHr = $r.data.myself.currentSpendPerHr }
}
@{ captured_at = $ts
   note = 'Billing-relevant account fields only. No personal account content is recorded.'
   vast = $vastAcct; runpod = $runpodAcct } |
    ConvertTo-Json -Depth 6 | Out-File (Join-Path $EvidenceDir 'account-state.json') -Encoding utf8

Write-Host "Vast qualifying offers : $(@($vastOffers).Count)"
Write-Host "RunPod GPU types       : $(@($runpodTypes).Count)"
Write-Host "Evidence written to evidence/provider/"
