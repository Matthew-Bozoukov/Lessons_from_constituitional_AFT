<#
.SYNOPSIS
    Run a bash script on the audit pod, or upload/download files.

.DESCRIPTION
    Quoting a non-trivial shell command through PowerShell -> ssh.exe -> bash
    mangles it (PowerShell strips quotes before ssh ever sees them). This helper
    sidesteps that entirely: the script is written to a temp file with LF line
    endings, uploaded with scp, and executed by path. No shell quoting is
    involved anywhere.

    The SSH key path comes from MSM_SSH_PRIVATE_KEY and is never printed.

.EXAMPLE
    .\Invoke-Remote.ps1 -ScriptText 'nvidia-smi; df -h /workspace'

.EXAMPLE
    .\Invoke-Remote.ps1 -ScriptFile .\scripts\remote\serve.sh -Detach
#>
[CmdletBinding(DefaultParameterSetName = 'Text')]
param(
    [Parameter(Mandatory, ParameterSetName = 'Text')][string]$ScriptText,
    [Parameter(Mandatory, ParameterSetName = 'File')][string]$ScriptFile,

    [Parameter(ParameterSetName = 'Upload')][string]$UploadFrom,
    [Parameter(ParameterSetName = 'Upload')][string]$UploadTo,
    [Parameter(ParameterSetName = 'Download')][string]$DownloadFrom,
    [Parameter(ParameterSetName = 'Download')][string]$DownloadTo,

    [switch]$Detach,
    [string]$RemoteName,
    [int]$TimeoutSec = 0
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$root = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$ep = Get-Content -LiteralPath (Join-Path $root 'runtime\provider-monitor\ssh-endpoint.json') -Raw | ConvertFrom-Json

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

& (Join-Path $root 'scripts\secrets\Invoke-WithInfraSecrets.ps1') -Inject @('MSM_SSH_PRIVATE_KEY') -ScriptBlock {
    Set-StrictMode -Off
    $k = $env:MSM_SSH_PRIVATE_KEY
    $common = @('-i', $k, '-o', 'StrictHostKeyChecking=no', '-o', 'ServerAliveInterval=30', '-o', 'ServerAliveCountMax=6')
    $sshArgs = $common + @('-p', "$($ep.ssh_port)", "root@$($ep.ssh_host)")
    $scpArgs = $common + @('-P', "$($ep.ssh_port)")

    switch ($mode) {
        'Upload' {
            & scp.exe @scpArgs $UploadFrom "root@$($ep.ssh_host):$UploadTo"
            return
        }
        'Download' {
            & scp.exe @scpArgs "root@$($ep.ssh_host):$DownloadFrom" $DownloadTo
            return
        }
    }

    & scp.exe @scpArgs $localTmp "root@$($ep.ssh_host):/workspace/$RemoteName" | Out-Null

    if ($Detach) {
        # setsid + nohup + closed stdin so the process survives the SSH session.
        & ssh.exe @sshArgs "chmod +x /workspace/$RemoteName; cd /workspace; setsid nohup bash /workspace/$RemoteName > /workspace/logs/$RemoteName.out 2>&1 < /dev/null & sleep 1; echo DETACHED:/workspace/logs/$RemoteName.out"
    }
    else {
        if ($TimeoutSec -gt 0) {
            & ssh.exe @sshArgs "chmod +x /workspace/$RemoteName; timeout $TimeoutSec bash /workspace/$RemoteName"
        }
        else {
            & ssh.exe @sshArgs "chmod +x /workspace/$RemoteName; bash /workspace/$RemoteName"
        }
    }
}

if ($localTmp) { Remove-Item -LiteralPath $localTmp -Force -ErrorAction SilentlyContinue }
