# ProviderStatus.psm1
#
# Live credit and resource indicator.
#
# Every number this module publishes carries an explicit basis:
#   exact-provider-reported : read directly from the provider's authenticated API
#   locally-calculated      : derived from provider figures plus local clocks
#   estimated               : derived from an assumption that may not hold
#   unavailable             : the provider does not expose it and nothing is invented
#
# Artifacts written to runtime/provider-monitor/:
#   run-state.json    run configuration and identity (source of truth for cost)
#   status.json       machine-readable current status, every field labelled
#   status.md         human-readable current status
#   status-history.jsonl  append-only, one JSON object per refresh
#   monitor.pid       monitor process state
#
# RUNPOD_API_KEY must already be in the process environment; callers come
# through Invoke-WithInfraSecrets.ps1.

Set-StrictMode -Version Latest

$script:ProviderRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$script:MonitorDir   = Join-Path $script:ProviderRoot 'runtime\provider-monitor'
$script:WatchdogDir  = Join-Path $script:ProviderRoot 'runtime\watchdog'

function Get-MonitorPath { param([string]$Name) Join-Path $script:MonitorDir $Name }

function New-Field {
    <# A published number plus the basis on which it may be trusted. #>
    param($Value, [ValidateSet('exact-provider-reported','locally-calculated','estimated','unavailable')][string]$Basis, [string]$Unit)
    [pscustomobject]@{ value = $Value; basis = $Basis; unit = $Unit }
}

function Initialize-RunState {
    <#
    .SYNOPSIS
        Create run-state.json before any paid resource exists.
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][string]$Provider,
        [Parameter(Mandatory)][string]$Gpu,
        [Parameter(Mandatory)][double]$HourlyUsd,
        [Parameter(Mandatory)][double]$MaxGpuSpendUsd,
        [Parameter(Mandatory)][double]$MaxWallClockHours,
        [Parameter(Mandatory)][double]$IdleShutdownMinutes,
        [double]$StartingBalanceUsd = [double]::NaN,
        [string]$StartingBalanceBasis = 'unavailable',
        [double]$StorageHourlyUsd = 0.0
    )

    if (-not (Test-Path $script:MonitorDir)) { New-Item -ItemType Directory -Path $script:MonitorDir -Force | Out-Null }

    $state = [ordered]@{
        run_id                  = 'msm-audit-' + (Get-Date).ToUniversalTime().ToString('yyyyMMdd-HHmmss')
        provider                = $Provider
        gpu                     = $Gpu
        hourly_usd              = $HourlyUsd
        storage_hourly_usd      = $StorageHourlyUsd
        instance_id             = $null
        instance_created_at     = $null
        instance_state          = 'not-provisioned'
        terminated_at           = $null
        starting_balance_usd    = $(if ([double]::IsNaN($StartingBalanceUsd)) { $null } else { $StartingBalanceUsd })
        starting_balance_basis  = $StartingBalanceBasis
        budget                  = [ordered]@{
            max_gpu_spend_usd     = $MaxGpuSpendUsd
            max_wall_clock_hours  = $MaxWallClockHours
            idle_shutdown_minutes = $IdleShutdownMinutes
        }
        hard_deadline_utc       = $null
        created_at              = (Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ')
    }

    $state | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Get-MonitorPath 'run-state.json') -Encoding utf8
    return [pscustomobject]$state
}

function Get-RunState {
    [CmdletBinding()] param()
    $p = Get-MonitorPath 'run-state.json'
    if (-not (Test-Path -LiteralPath $p)) { return $null }
    return (Get-Content -LiteralPath $p -Raw | ConvertFrom-Json)
}

function Set-RunState {
    [CmdletBinding()] param([Parameter(Mandatory)]$State)
    $State | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Get-MonitorPath 'run-state.json') -Encoding utf8
}

function Register-Instance {
    <#
    .SYNOPSIS
        Record a newly created paid resource and set its hard deadline.
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][string]$InstanceId,
        [string]$CreatedAtUtc
    )
    $state = Get-RunState
    if (-not $state) { throw 'run-state.json missing. Call Initialize-RunState first.' }
    if (-not $CreatedAtUtc) { $CreatedAtUtc = (Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ') }

    $state.instance_id         = $InstanceId
    $state.instance_created_at = $CreatedAtUtc
    $state.instance_state      = 'provisioned'
    $state.hard_deadline_utc   = ([datetime]::Parse($CreatedAtUtc).ToUniversalTime().AddHours($state.budget.max_wall_clock_hours)).ToString('yyyy-MM-ddTHH:mm:ssZ')
    Set-RunState -State $state
    return $state
}

function Test-ModelServerHealth {
    <# vLLM through the local end of the SSH tunnel. #>
    [CmdletBinding()] param([string]$Uri = 'http://127.0.0.1:8000/v1/models', [int]$TimeoutSec = 5)
    try {
        $r = Invoke-WebRequest -Uri $Uri -UseBasicParsing -TimeoutSec $TimeoutSec -ErrorAction Stop
        if ([int]$r.StatusCode -eq 200) { return 'healthy' }
        return "unhealthy-http-$([int]$r.StatusCode)"
    } catch { return 'unreachable' }
}

function Test-TunnelHealth {
    <# Local forwarded port listening implies the SSH tunnel is up. #>
    [CmdletBinding()] param([int]$Port = 8000)
    try {
        $c = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
        if ($c) { return 'listening' }
        return 'down'
    } catch { return 'unknown' }
}

function Get-WatchdogState {
    [CmdletBinding()] param()
    $hb = Join-Path $script:WatchdogDir 'watchdog.state.json'
    if (-not (Test-Path -LiteralPath $hb)) { return 'not-running' }
    try {
        $s = Get-Content -LiteralPath $hb -Raw | ConvertFrom-Json
        $age = ((Get-Date).ToUniversalTime() - [datetime]::Parse($s.last_heartbeat_utc).ToUniversalTime()).TotalSeconds
        $alive = $false
        if ($s.pid) { $alive = [bool](Get-Process -Id $s.pid -ErrorAction SilentlyContinue) }
        if (-not $alive) { return 'dead (pid gone)' }
        if ($age -gt 300) { return "stale ($([int]$age)s since heartbeat)" }
        return "armed (deadline $($s.hard_deadline_utc))"
    } catch { return 'unreadable' }
}

function Write-ProviderStatus {
    <#
    .SYNOPSIS
        Refresh every figure, write all status artifacts, return the compact line.
    .PARAMETER Reason
        Why this refresh happened, e.g. 'periodic', 'post-provision',
        'post-model-download', 'post-audit', 'pre-cleanup', 'post-cleanup'.
    #>
    [CmdletBinding()]
    param(
        [string]$Reason = 'periodic',
        [double]$AnthropicSpentUsd = 0.0,
        [double]$MaxAnthropicSpendUsd = 0.0,
        [switch]$SkipHealthChecks
    )

    $now   = (Get-Date).ToUniversalTime()
    $nowS  = $now.ToString('yyyy-MM-ddTHH:mm:ssZ')
    $state = Get-RunState
    if (-not $state) { throw 'run-state.json missing. Call Initialize-RunState first.' }

    # ---- provider balance -------------------------------------------------
    $balanceField = New-Field -Value $null -Basis 'unavailable' -Unit 'USD'
    $lastRefresh  = 'never'
    $apiOk = $false
    try {
        $bal = Get-RunPodBalance
        $balanceField = New-Field -Value ([math]::Round($bal.balance_usd, 4)) -Basis 'exact-provider-reported' -Unit 'USD'
        $lastRefresh  = $bal.retrieved_at
        $apiOk = $true
    } catch {
        # Do not invent a balance. Keep the previous successful refresh time.
        $prev = $null
        $sp = Get-MonitorPath 'status.json'
        if (Test-Path -LiteralPath $sp) {
            try { $prev = Get-Content -LiteralPath $sp -Raw | ConvertFrom-Json } catch { }
        }
        if ($prev -and $prev.last_successful_api_refresh) { $lastRefresh = $prev.last_successful_api_refresh }
    }

    # ---- instance state ---------------------------------------------------
    $instanceState = $state.instance_state
    $instanceId    = $state.instance_id
    if ($instanceId -and $Reason -ne 'post-cleanup') {
        try {
            $pod = Get-RunPodPod -PodId $instanceId
            if ($null -eq $pod) { $instanceState = 'absent' }
            elseif ($pod.PSObject.Properties.Match('desiredStatus').Count -gt 0) { $instanceState = [string]$pod.desiredStatus }
            else { $instanceState = 'present' }
        } catch { $instanceState = 'query-failed' }
    }

    # ---- elapsed and cost -------------------------------------------------
    $elapsedHours = 0.0
    $elapsedText  = '00:00'
    if ($state.instance_created_at) {
        $start = [datetime]::Parse($state.instance_created_at).ToUniversalTime()
        $end   = if ($state.terminated_at) { [datetime]::Parse($state.terminated_at).ToUniversalTime() } else { $now }
        $span  = $end - $start
        if ($span.Ticks -lt 0) { $span = [timespan]::Zero }
        $elapsedHours = $span.TotalHours
        $elapsedText  = '{0:00}:{1:00}' -f [int]$span.TotalHours, $span.Minutes
    }

    $hourlyTotal = [double]$state.hourly_usd + [double]$state.storage_hourly_usd
    $infraCost   = [math]::Round($elapsedHours * $hourlyTotal, 4)
    $costField   = New-Field -Value $infraCost -Basis 'locally-calculated' -Unit 'USD'

    $gpuRemaining = [math]::Round([double]$state.budget.max_gpu_spend_usd - $infraCost, 4)
    $gpuRemainingField = New-Field -Value $gpuRemaining -Basis 'locally-calculated' -Unit 'USD'

    # Provider-side delta is only meaningful with a trustworthy starting balance.
    $spendVsBalanceField = New-Field -Value $null -Basis 'unavailable' -Unit 'USD'
    if ($null -ne $state.starting_balance_usd -and $balanceField.basis -eq 'exact-provider-reported') {
        $spendVsBalanceField = New-Field -Value ([math]::Round([double]$state.starting_balance_usd - [double]$balanceField.value, 4)) `
            -Basis 'locally-calculated' -Unit 'USD'
    }

    $wallRemainingField = New-Field -Value $null -Basis 'unavailable' -Unit 'hours'
    if ($state.hard_deadline_utc) {
        $rem = ([datetime]::Parse($state.hard_deadline_utc).ToUniversalTime() - $now).TotalHours
        $wallRemainingField = New-Field -Value ([math]::Round([math]::Max($rem, 0), 3)) -Basis 'locally-calculated' -Unit 'hours'
    }

    # ---- health -----------------------------------------------------------
    $serverHealth = 'not-checked'; $tunnelHealth = 'not-checked'
    if (-not $SkipHealthChecks) {
        $serverHealth = Test-ModelServerHealth
        $tunnelHealth = Test-TunnelHealth
    }
    $watchdog = Get-WatchdogState

    $anthropicRemaining = if ($MaxAnthropicSpendUsd -gt 0) { [math]::Round($MaxAnthropicSpendUsd - $AnthropicSpentUsd, 4) } else { $null }

    # ---- publish ----------------------------------------------------------
    $status = [ordered]@{
        generated_at                 = $nowS
        refresh_reason               = $Reason
        run_id                       = $state.run_id
        provider                     = $state.provider
        gpu                          = $state.gpu
        hourly_price_usd             = New-Field -Value $hourlyTotal -Basis 'exact-provider-reported' -Unit 'USD/h'
        instance_id                  = $instanceId
        instance_state               = $instanceState
        provider_balance             = $balanceField
        starting_provider_balance    = New-Field -Value $state.starting_balance_usd -Basis $state.starting_balance_basis -Unit 'USD'
        provider_balance_delta       = $spendVsBalanceField
        elapsed_runtime              = New-Field -Value ([math]::Round($elapsedHours, 4)) -Basis 'locally-calculated' -Unit 'hours'
        elapsed_runtime_text         = $elapsedText
        estimated_infrastructure_cost = $costField
        experiment_gpu_budget_remaining = $gpuRemainingField
        wall_clock_remaining         = $wallRemainingField
        hard_deadline_utc            = $state.hard_deadline_utc
        model_server_health          = $serverHealth
        ssh_tunnel_health            = $tunnelHealth
        last_successful_api_refresh  = $lastRefresh
        api_refresh_ok_this_cycle    = $apiOk
        cleanup_watchdog_state       = $watchdog
        anthropic_spent_usd          = New-Field -Value ([math]::Round($AnthropicSpentUsd, 4)) -Basis 'locally-calculated' -Unit 'USD'
        anthropic_budget_remaining   = New-Field -Value $anthropicRemaining -Basis $(if ($null -eq $anthropicRemaining) { 'unavailable' } else { 'locally-calculated' }) -Unit 'USD'
    }

    $status | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Get-MonitorPath 'status.json') -Encoding utf8
    ($status | ConvertTo-Json -Depth 8 -Compress) | Add-Content -LiteralPath (Get-MonitorPath 'status-history.jsonl') -Encoding utf8

    # ---- compact line -----------------------------------------------------
    $balText = if ($balanceField.basis -eq 'unavailable') { 'provider balance unavailable' } else { "provider balance `$$($balanceField.value)" }
    $gpuTag  = if ($instanceState -in @('not-provisioned','absent','TERMINATED')) { '[NO GPU]' } else { '[GPU ACTIVE]' }
    $line = "$gpuTag $($state.provider) | $($state.gpu) | `$$hourlyTotal/h | $balText | experiment GPU budget `$$gpuRemaining remaining | elapsed $elapsedText | vLLM $serverHealth"

    # ---- markdown ---------------------------------------------------------
    $fmt = {
        param($f)
        if ($null -eq $f.value) { return "unavailable _(nothing invented)_" }
        return "$($f.value) $($f.unit) _($($f.basis))_"
    }
    $md = @(
        '# Provider status'
        ''
        '```text'
        $line
        '```'
        ''
        "Generated: $nowS  (refresh reason: $Reason)"
        ''
        'Every figure below is labelled with its basis: **exact-provider-reported**,'
        '**locally-calculated**, **estimated**, or **unavailable**. Nothing is invented;'
        'an unavailable figure is reported as unavailable.'
        ''
        '| Field | Value |'
        '| --- | --- |'
        "| Provider | $($state.provider) |"
        "| GPU | $($state.gpu) |"
        "| Hourly price | $(& $fmt $status.hourly_price_usd) |"
        "| Instance ID | $(if ($instanceId) { '`' + $instanceId + '`' } else { 'none' }) |"
        "| Instance state | $instanceState |"
        "| Provider balance | $(& $fmt $balanceField) |"
        "| Starting provider balance | $(& $fmt $status.starting_provider_balance) |"
        "| Provider balance delta | $(& $fmt $spendVsBalanceField) |"
        "| Elapsed runtime | $elapsedText ($(& $fmt $status.elapsed_runtime)) |"
        "| Estimated infrastructure cost | $(& $fmt $costField) |"
        "| Experiment GPU budget remaining | $(& $fmt $gpuRemainingField) |"
        "| Wall-clock remaining | $(& $fmt $wallRemainingField) |"
        "| Hard deadline | $(if ($state.hard_deadline_utc) { $state.hard_deadline_utc } else { 'not set (no instance)' }) |"
        "| Model-server health | $serverHealth |"
        "| SSH-tunnel health | $tunnelHealth |"
        "| Last successful API refresh | $lastRefresh |"
        "| Cleanup-watchdog state | $watchdog |"
        "| Anthropic spend | $(& $fmt $status.anthropic_spent_usd) |"
        "| Anthropic budget remaining | $(& $fmt $status.anthropic_budget_remaining) |"
        ''
        '## Hard limits'
        ''
        "- GPU spend cap: `$$($state.budget.max_gpu_spend_usd)"
        "- Wall-clock cap: $($state.budget.max_wall_clock_hours) h"
        "- Idle shutdown: $($state.budget.idle_shutdown_minutes) min"
        ''
        '## Raw artifacts'
        ''
        '- [status.json](./status.json)'
        '- [status-history.jsonl](./status-history.jsonl) (append-only)'
        '- [run-state.json](./run-state.json)'
    ) -join "`n"
    $md | Set-Content -LiteralPath (Get-MonitorPath 'status.md') -Encoding utf8

    return $line
}

Export-ModuleMember -Function Initialize-RunState, Get-RunState, Set-RunState, Register-Instance,
    Write-ProviderStatus, Test-ModelServerHealth, Test-TunnelHealth, Get-WatchdogState, Get-MonitorPath
