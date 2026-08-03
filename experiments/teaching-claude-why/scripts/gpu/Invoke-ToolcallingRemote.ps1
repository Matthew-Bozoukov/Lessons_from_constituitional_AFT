<#
.SYNOPSIS
    Run a bash script on the tool-calling SFT pod, or upload/download files.

.DESCRIPTION
    Same technique as the sibling experiment's Invoke-Remote.ps1 - write the script to a
    temp file with LF endings, scp it, execute it by path, so no shell quoting survives
    the PowerShell -> ssh.exe -> bash journey - but reading THIS run's endpoint file
    rather than the shared runtime\provider-monitor one another task is using.

    The SSH key path comes from MSM_SSH_PRIVATE_KEY and is never printed.
#>
[CmdletBinding(DefaultParameterSetName = 'Text')]
param(
    [Parameter(Mandatory, ParameterSetName = 'Text')][string]$ScriptText,
    [Parameter(Mandatory, ParameterSetName = 'File')][string]$ScriptFile,

    [Parameter(Mandatory, ParameterSetName = 'Upload')][string]$UploadFrom,
    [Parameter(Mandatory, ParameterSetName = 'Upload')][string]$UploadTo,
    [Parameter(Mandatory, ParameterSetName = 'Download')][string]$DownloadFrom,
    [Parameter(Mandatory, ParameterSetName = 'Download')][string]$DownloadTo,

    [switch]$Detach,
    [string]$RemoteName,
    [int]$TimeoutSec = 0
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$expRoot    = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$vulnRoot   = (Resolve-Path (Join-Path $expRoot '..\vulnerabilities')).Path
$secretsDir = Join-Path $vulnRoot 'scripts\secrets'
$ep = Get-Content -LiteralPath (Join-Path $expRoot 'runtime\toolcalling\ssh-endpoint.json') -Raw | ConvertFrom-Json

if (-not $RemoteName) { $RemoteName = 'rexec-' + [guid]::NewGuid().ToString('N').Substring(0, 8) + '.sh' }

$mode = $PSCmdlet.ParameterSetName
$payload = $null
if ($mode -eq 'Text') { $payload = $ScriptText }
elseif ($mode -eq 'File') { $payload = [System.IO.File]::ReadAllText((Resolve-Path $ScriptFile).Path) }

$localTmp = $null
if ($payload) {
    $localTmp = Join-Path $env:TEMP $RemoteName
    # LF only: bash rejects CRLF shebangs and mis-parses trailing \r.
    [System.IO.File]::WriteAllText($localTmp, ($payload -replace "`r`n", "`n"), (New-Object System.Text.UTF8Encoding($false)))
}

& (Join-Path $secretsDir 'Invoke-WithInfraSecrets.ps1') -Inject @('MSM_SSH_PRIVATE_KEY') -ScriptBlock {
    Set-StrictMode -Off
    $k = $env:MSM_SSH_PRIVATE_KEY
    $common = @('-i', $k, '-o', 'StrictHostKeyChecking=no', '-o', 'UserKnownHostsFile=/dev/null',
                '-o', 'LogLevel=ERROR', '-o', 'ServerAliveInterval=30', '-o', 'ServerAliveCountMax=6')
    $sshArgs = $common + @('-p', "$($ep.port)", "root@$($ep.ip)")
    $scpArgs = $common + @('-P', "$($ep.port)")

    switch ($mode) {
        'Upload'   { & scp.exe @scpArgs $UploadFrom "root@$($ep.ip):$UploadTo"; return }
        'Download' { & scp.exe @scpArgs "root@$($ep.ip):$DownloadFrom" $DownloadTo; return }
    }

    & scp.exe @scpArgs $localTmp "root@$($ep.ip):/workspace/$RemoteName" | Out-Null

    if ($Detach) {
        # setsid + nohup + closed stdin so the process survives the SSH session.
        & ssh.exe @sshArgs "mkdir -p /workspace/logs; chmod +x /workspace/$RemoteName; cd /workspace; setsid nohup bash /workspace/$RemoteName > /workspace/logs/$RemoteName.out 2>&1 < /dev/null & sleep 1; echo DETACHED:/workspace/logs/$RemoteName.out"
    }
    elseif ($TimeoutSec -gt 0) {
        & ssh.exe @sshArgs "chmod +x /workspace/$RemoteName; timeout $TimeoutSec bash /workspace/$RemoteName"
    }
    else {
        & ssh.exe @sshArgs "chmod +x /workspace/$RemoteName; bash /workspace/$RemoteName"
    }
}

if ($localTmp) { Remove-Item -LiteralPath $localTmp -Force -ErrorAction SilentlyContinue }
