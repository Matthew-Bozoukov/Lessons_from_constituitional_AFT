<#
.SYNOPSIS
    Validate every credential against a harmless read-only official endpoint.

.DESCRIPTION
    Records ONLY provider, timestamp, HTTP status and success/failure. Response
    bodies are never stored, never printed and never inspected beyond the
    minimum needed to classify success. Account identifiers, balances and other
    body content are deliberately discarded here; the provider monitor reads
    balances separately and records them as figures, not as raw bodies.

    Infrastructure credentials go through Invoke-WithInfraSecrets.ps1 and the
    Anthropic credential through Invoke-WithPetriSecrets.ps1, so the two never
    share a process.

.OUTPUTS
    Writes evidence/credentials/credential-validation.json and .md
#>
[CmdletBinding()]
param(
    [string]$EvidenceDir = (Join-Path $PSScriptRoot '..\..\evidence\credentials')
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$EvidenceDir = (Resolve-Path $EvidenceDir).Path
$secretsDir  = $PSScriptRoot

function New-Result {
    param($Provider, $Endpoint, $Method, $Status, $Ok, $Note)
    [pscustomobject]@{
        provider    = $Provider
        endpoint    = $Endpoint
        method      = $Method
        timestamp   = (Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ')
        http_status = $Status
        result      = if ($Ok) { 'success' } else { 'failure' }
        note        = $Note
    }
}

function Invoke-StatusOnly {
    <#  Returns the HTTP status code and nothing from the body. #>
    param([string]$Uri, [hashtable]$Headers, [string]$Method = 'GET')
    try {
        $resp = Invoke-WebRequest -Uri $Uri -Headers $Headers -Method $Method `
            -UseBasicParsing -TimeoutSec 30 -ErrorAction Stop
        return @{ Status = [int]$resp.StatusCode; Ok = ([int]$resp.StatusCode -ge 200 -and [int]$resp.StatusCode -lt 300) }
    }
    catch {
        $status = $null
        if ($_.Exception.PSObject.Properties.Name -contains 'Response' -and $_.Exception.Response) {
            try { $status = [int]$_.Exception.Response.StatusCode } catch { $status = $null }
        }
        return @{ Status = $status; Ok = $false; Error = $_.Exception.GetType().Name }
    }
}

$results = @()

# ---------------------------------------------------------------- Vast.ai ----
$vast = & "$secretsDir\Invoke-WithInfraSecrets.ps1" -Inject @('VAST_API_KEY') -ScriptBlock {
    $u = 'https://console.vast.ai/api/v0/users/current/'
    try {
        $r = Invoke-WebRequest -Uri $u -Headers @{ Authorization = "Bearer $($env:VAST_API_KEY)" } `
            -UseBasicParsing -TimeoutSec 30 -ErrorAction Stop
        @{ Status = [int]$r.StatusCode; Ok = $true }
    } catch {
        $s = $null
        if ($_.Exception.Response) { try { $s = [int]$_.Exception.Response.StatusCode } catch {} }
        @{ Status = $s; Ok = $false; Error = $_.Exception.Message.Substring(0, [Math]::Min(120, $_.Exception.Message.Length)) }
    }
}
$results += New-Result 'vast.ai' 'GET /api/v0/users/current/' 'GET' $vast.Status $vast.Ok `
    'Read-only account endpoint. Response body discarded.'

# ----------------------------------------------------------------- RunPod ----
$runpod = & "$secretsDir\Invoke-WithInfraSecrets.ps1" -Inject @('RUNPOD_API_KEY') -ScriptBlock {
    $u = 'https://rest.runpod.io/v1/pods'
    try {
        $r = Invoke-WebRequest -Uri $u -Headers @{ Authorization = "Bearer $($env:RUNPOD_API_KEY)" } `
            -UseBasicParsing -TimeoutSec 30 -ErrorAction Stop
        @{ Status = [int]$r.StatusCode; Ok = $true }
    } catch {
        $s = $null
        if ($_.Exception.Response) { try { $s = [int]$_.Exception.Response.StatusCode } catch {} }
        @{ Status = $s; Ok = $false; Error = $_.Exception.Message.Substring(0, [Math]::Min(120, $_.Exception.Message.Length)) }
    }
}
$results += New-Result 'runpod' 'GET /v1/pods' 'GET' $runpod.Status $runpod.Ok `
    'Read-only pod listing. Response body discarded.'

# ------------------------------------------------------------ Hugging Face ---
$hf = & "$secretsDir\Invoke-WithInfraSecrets.ps1" -Inject @('HF_TOKEN') -ScriptBlock {
    $u = 'https://huggingface.co/api/whoami-v2'
    try {
        $r = Invoke-WebRequest -Uri $u -Headers @{ Authorization = "Bearer $($env:HF_TOKEN)" } `
            -UseBasicParsing -TimeoutSec 30 -ErrorAction Stop
        @{ Status = [int]$r.StatusCode; Ok = $true }
    } catch {
        $s = $null
        if ($_.Exception.Response) { try { $s = [int]$_.Exception.Response.StatusCode } catch {} }
        @{ Status = $s; Ok = $false; Error = $_.Exception.Message.Substring(0, [Math]::Min(120, $_.Exception.Message.Length)) }
    }
}
$results += New-Result 'huggingface' 'GET /api/whoami-v2' 'GET' $hf.Status $hf.Ok `
    'Read-only identity endpoint. Response body discarded.'

# --------------------------------------------------------------- Anthropic ---
# Isolated wrapper. This is the only place an Anthropic credential is used here.
$anthropic = & "$secretsDir\Invoke-WithPetriSecrets.ps1" -ScriptBlock {
    $u = 'https://api.anthropic.com/v1/models?limit=1'
    try {
        $r = Invoke-WebRequest -Uri $u -Headers @{
            'x-api-key'         = $env:ANTHROPIC_API_KEY
            'anthropic-version' = '2023-06-01'
        } -UseBasicParsing -TimeoutSec 30 -ErrorAction Stop
        @{ Status = [int]$r.StatusCode; Ok = $true }
    } catch {
        $s = $null
        if ($_.Exception.Response) { try { $s = [int]$_.Exception.Response.StatusCode } catch {} }
        @{ Status = $s; Ok = $false; Error = $_.Exception.Message.Substring(0, [Math]::Min(120, $_.Exception.Message.Length)) }
    }
}
$results += New-Result 'anthropic' 'GET /v1/models?limit=1' 'GET' $anthropic.Status $anthropic.Ok `
    'Read-only model listing. Response body discarded.'

# ------------------------------------------------------- SSH key key-shape ---
# Determine whether MSM_SSH_PRIVATE_KEY holds inline PEM material or a path to
# a key file, without revealing either. Only the classification is recorded.
$ssh = & "$secretsDir\Invoke-WithInfraSecrets.ps1" -Inject @('MSM_SSH_PRIVATE_KEY') -ScriptBlock {
    $v = $env:MSM_SSH_PRIVATE_KEY
    $isPem = $v -match '-----BEGIN [A-Z ]*PRIVATE KEY-----'
    # A Windows path, a POSIX path, or a ~-relative path. Checked before any
    # attempt to treat the value as key material.
    $looksPath = (-not $isPem) -and ($v -match '^[A-Za-z]:|^[~/\\.]')
    $pathExists = $false
    $parentExists = $false
    if ($looksPath) {
        $expanded = if ($v.StartsWith('~')) { Join-Path $HOME $v.Substring(1).TrimStart('/', '\') } else { $v }
        $pathExists = Test-Path -LiteralPath $expanded
        $parent = Split-Path -Parent $expanded
        if ($parent) { $parentExists = Test-Path -LiteralPath $parent }
    }
    @{ IsPem = [bool]$isPem; LooksPath = [bool]$looksPath; PathExists = [bool]$pathExists; ParentExists = [bool]$parentExists }
}
$sshForm =
    if ($ssh.IsPem) { 'inline PEM key material' }
    elseif ($ssh.LooksPath -and $ssh.PathExists) { 'path to a key file that exists' }
    elseif ($ssh.LooksPath -and $ssh.ParentExists) { 'path to a key file that DOES NOT EXIST (parent directory exists)' }
    elseif ($ssh.LooksPath) { 'path to a key file that DOES NOT EXIST (parent directory also missing)' }
    else { 'unrecognized form' }
$sshOk = $ssh.IsPem -or ($ssh.LooksPath -and $ssh.PathExists)
$results += New-Result 'ssh-key' 'local key-shape classification' 'local' $null $sshOk `
    "MSM_SSH_PRIVATE_KEY form: $sshForm. Neither the path nor any key material was read or displayed. SSH is required only at GPU provisioning time; every earlier phase is unaffected."

# --------------------------------------------------------------- Persist ----
$payload = [pscustomobject]@{
    generated_at = (Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ')
    policy       = 'Only provider, endpoint, timestamp, HTTP status and success/failure are recorded. No response body, account identifier or balance is stored in this file.'
    checks       = $results
}
$jsonPath = Join-Path $EvidenceDir 'credential-validation.json'
$payload | ConvertTo-Json -Depth 6 | Out-File -FilePath $jsonPath -Encoding utf8

$md = @()
$md += '# Credential validation'
$md += ''
$md += "Generated: $($payload.generated_at)"
$md += ''
$md += 'Each credential was exercised against one harmless read-only official endpoint.'
$md += 'Only the fields below were recorded. No response body, account identifier or'
$md += 'balance value is stored here, and no credential value was printed, logged or'
$md += 'written to disk at any point.'
$md += ''
$md += '| Provider | Endpoint | Timestamp (UTC) | HTTP status | Result |'
$md += '| --- | --- | --- | --- | --- |'
foreach ($r in $results) {
    $st = if ($null -eq $r.http_status) { 'n/a' } else { $r.http_status }
    $md += "| $($r.provider) | ``$($r.endpoint)`` | $($r.timestamp) | $st | **$($r.result)** |"
}
$md += ''
$md += '## Notes'
$md += ''
foreach ($r in $results) { $md += "- **$($r.provider)** - $($r.note)" }
$md += ''
$md += '## Raw artifact'
$md += ''
$md += '- [credential-validation.json](./credential-validation.json)'
$md -join "`n" | Out-File -FilePath (Join-Path $EvidenceDir 'credential-validation.md') -Encoding utf8

$results | Format-Table provider, http_status, result -AutoSize
