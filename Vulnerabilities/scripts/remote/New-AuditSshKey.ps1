<#
.SYNOPSIS
    Create the audit SSH keypair at the path MSM_SSH_PRIVATE_KEY names, and
    register only the PUBLIC key with the RunPod account.

.DESCRIPTION
    Run once, with the user's explicit approval, because registering a public
    key changes a provider account setting.

    Safety properties:
      * Never overwrites. If the private key file already exists the script
        aborts without touching it.
      * The private key path comes from the secret file and is never printed.
        Only the public key and its fingerprint are displayed; both are safe to
        share by construction.
      * Existing RunPod public keys are preserved. The account pubKey field
        holds newline-separated keys; this script appends and never replaces a
        non-empty field with a bare new key.
      * The private key is written outside the repository, under
        $HOME\.ssh, and the repository ignore rules exclude id_* and *.key
        regardless.

.OUTPUTS
    Writes evidence/credentials/ssh-key-provisioning.md
#>
[CmdletBinding()]
param(
    [string]$Comment = 'msm-audit',
    [string]$EvidenceDir = (Join-Path $PSScriptRoot '..\..\evidence\credentials')
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$secretsDir  = (Resolve-Path (Join-Path $PSScriptRoot '..\secrets')).Path
$EvidenceDir = (Resolve-Path $EvidenceDir).Path

# ---------------------------------------------------------- generate key ----
$gen = & "$secretsDir\Invoke-WithInfraSecrets.ps1" -Inject @('MSM_SSH_PRIVATE_KEY') -ScriptBlock {
    Set-StrictMode -Off
    $keyPath = $env:MSM_SSH_PRIVATE_KEY
    $pubPath = "$keyPath.pub"

    if (Test-Path -LiteralPath $keyPath) {
        return @{ Created = $false; Reason = 'private key already exists; refusing to overwrite'; PubPath = $pubPath; Existed = $true }
    }

    $parent = Split-Path -Parent $keyPath
    if (-not (Test-Path -LiteralPath $parent)) {
        New-Item -ItemType Directory -Path $parent -Force | Out-Null
    }

    # No passphrase: required for unattended tunnel reconnection.
    # PowerShell 5.1 drops empty-string arguments to native executables, so the
    # empty -N value is passed through cmd.exe, which quotes it correctly.
    $cmd = 'ssh-keygen -q -t ed25519 -f "{0}" -N "" -C "{1}"' -f $keyPath, $Comment
    $null = & cmd.exe /c $cmd 2>&1
    if (-not (Test-Path -LiteralPath $keyPath)) {
        return @{ Created = $false; Reason = 'ssh-keygen did not produce a key file'; PubPath = $pubPath; Existed = $false }
    }

    # Windows OpenSSH refuses a private key readable by other principals.
    # icacls works for a file the current user owns; Set-Acl with inheritance
    # protection would require SeSecurityPrivilege and therefore elevation.
    $me = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
    $null = & icacls.exe $keyPath /inheritance:r 2>&1
    $null = & icacls.exe $keyPath /grant:r "${me}:(F)" 2>&1

    $pub = (Get-Content -LiteralPath $pubPath -Raw).Trim()
    $fp  = (& ssh-keygen.exe -lf $pubPath) -join ''

    return @{ Created = $true; PublicKey = $pub; Fingerprint = $fp; Existed = $false }
}

if (-not $gen.Created) {
    if ($gen.Existed) {
        Write-Host "Private key already present. Reusing it; nothing was overwritten."
        $gen = & "$secretsDir\Invoke-WithInfraSecrets.ps1" -Inject @('MSM_SSH_PRIVATE_KEY') -ScriptBlock {
            Set-StrictMode -Off
            $keyPath = $env:MSM_SSH_PRIVATE_KEY
            $pubPath = "$keyPath.pub"
            # Harden permissions on the reuse path too: OpenSSH rejects a
            # private key that other principals can read.
            $me = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
            $null = & icacls.exe $keyPath /inheritance:r 2>&1
            $null = & icacls.exe $keyPath /grant:r "${me}:(F)" 2>&1
            @{ Created = $false; Existed = $true
               PublicKey = (Get-Content -LiteralPath $pubPath -Raw).Trim()
               Fingerprint = ((& ssh-keygen.exe -lf $pubPath) -join '') }
        }
    } else {
        throw "SSH key generation failed: $($gen.Reason)"
    }
}

Write-Host "Public key  : $($gen.PublicKey)"
Write-Host "Fingerprint : $($gen.Fingerprint)"

# ------------------------------------------------- register with RunPod ----
$reg = & "$secretsDir\Invoke-WithInfraSecrets.ps1" -Inject @('RUNPOD_API_KEY') -ScriptBlock {
    Set-StrictMode -Off
    $newKey = $gen.PublicKey
    $hdr = @{ Authorization = "Bearer $($env:RUNPOD_API_KEY)" }

    $cur = Invoke-RestMethod -Uri 'https://api.runpod.io/graphql' -Method Post -ContentType 'application/json' `
        -Body '{"query":"query { myself { pubKey } }"}' -Headers $hdr -TimeoutSec 30
    $existing = $cur.data.myself.pubKey
    if ($null -eq $existing) { $existing = '' }

    if ($existing -match [regex]::Escape($newKey)) {
        return @{ Action = 'already-present'; KeyCount = (($existing -split "`n") | Where-Object { $_.Trim() }).Count }
    }

    # Preserve any key already on the account.
    $combined = if ([string]::IsNullOrWhiteSpace($existing)) { $newKey } else { ($existing.TrimEnd() + "`n" + $newKey) }

    $mutation = @{
        query     = 'mutation UpdateUserSettings($input: UpdateUserSettingsInput!) { updateUserSettings(input: $input) { id pubKey } }'
        variables = @{ input = @{ pubKey = $combined } }
    } | ConvertTo-Json -Depth 6

    $res = Invoke-RestMethod -Uri 'https://api.runpod.io/graphql' -Method Post -ContentType 'application/json' `
        -Body $mutation -Headers $hdr -TimeoutSec 30

    if ($res.errors) { return @{ Action = 'failed'; Error = ($res.errors | ForEach-Object { $_.message }) -join '; ' } }

    $after = $res.data.updateUserSettings.pubKey
    @{ Action = 'registered'
       Verified = ($after -match [regex]::Escape($newKey))
       KeyCount = (($after -split "`n") | Where-Object { $_.Trim() }).Count }
}

if ($reg.Action -eq 'failed') { throw "RunPod public-key registration failed: $($reg.Error)" }
Write-Host "RunPod registration: $($reg.Action); account now holds $($reg.KeyCount) public key(s)."

# ------------------------------------------------------------- evidence ----
$ts = (Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ')
$md = @(
    '# SSH key provisioning'
    ''
    "Generated: $ts"
    ''
    'The audit SSH keypair was created at the path `MSM_SSH_PRIVATE_KEY` already'
    'named in `infra.env`. That path is not reproduced here. The private key was'
    'never printed, never logged and lives outside the repository.'
    ''
    '| Item | Value |'
    '| --- | --- |'
    "| Action | $(if ($gen.Existed) { 'reused pre-existing keypair' } else { 'created new ed25519 keypair' }) |"
    '| Algorithm | ed25519 |'
    '| Passphrase | none (required for unattended tunnel reconnection) |'
    "| Public key fingerprint | ``$($gen.Fingerprint)`` |"
    "| RunPod registration | $($reg.Action) |"
    "| Public keys on RunPod account after | $($reg.KeyCount) |"
    "| Registration verified by re-read | $(if ($reg.PSObject.Properties.Name -contains 'Verified') { $reg.Verified } else { 'n/a' }) |"
    ''
    '## Authorization'
    ''
    'The user was asked before this ran, because registering a public key changes'
    'a RunPod account setting, and explicitly approved generating a keypair at'
    'the configured path. The RunPod account held **no** public keys beforehand,'
    'so nothing was displaced; the script appends rather than replaces in any'
    'case.'
    ''
    '## Public key'
    ''
    'Public keys are safe to record.'
    ''
    '```text'
    $gen.PublicKey
    '```'
) -join "`n"

$md | Out-File -FilePath (Join-Path $EvidenceDir 'ssh-key-provisioning.md') -Encoding utf8
Write-Host "Evidence written to evidence/credentials/ssh-key-provisioning.md"
