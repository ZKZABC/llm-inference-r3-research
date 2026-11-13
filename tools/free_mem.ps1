#Requires -Version 5.1
<#
.SYNOPSIS
    为 WSL 腾内存：先诊断，再决定要不要清理。

.DESCRIPTION
    默认只读诊断，不动任何东西。加 -Apply 才执行清理。

    ── 三条必须知道的事实 ──────────────────────────────
    1. Windows 的 "Available" 已经包含 standby 缓存。
       所以"清缓存"不会让可用内存变多，只会改变构成。
       市面上所谓"内存清理大师"基本是安慰剂。
    2. .wslconfig 的 memory=NNGB 是「上限」不是「预留」。
       WSL 按需增长，不会一开机就吃掉 12GB。
    3. 真正能腾出内存的只有两件事：关进程、wsl --shutdown。
       后者能释放 vmmem 占用的全部内存（最立竿见影）。

.PARAMETER Apply
    执行清理：关闭白名单进程 + wsl --shutdown

.PARAMETER Force
    跳过交互确认（配合 -Apply 使用）

.PARAMETER Top
    显示前 N 个内存大户，默认 15

.EXAMPLE
    .\free_mem.ps1
        只看不动

.EXAMPLE
    .\free_mem.ps1 -Apply
        清理（会先列清单让你确认）
#>
param(
    [switch]$Apply,
    [switch]$Force,
    [int]$Top = 15
)

$ErrorActionPreference = 'Stop'

# 可以安全关闭的进程（浏览器 / 聊天 / 游戏启动器 / 音乐）
# 刻意不包含：vmmemWSL（关了 WSL 就死）、explorer、WorkBuddy、任何系统进程
$Killable = @(
    'msedge', 'chrome', 'firefox', 'brave', 'opera', 'vivaldi',
    'WeChat', 'Weixin', 'QQ', 'TIM', 'DingTalk', 'WXWork',
    'steam', 'steamwebhelper', 'EpicGamesLauncher', 'Battle.net',
    'Spotify', 'cloudmusic', 'QQMusic', 'KuGou',
    'Notion', 'Slack', 'Discord', 'Telegram', 'Teams'
)

# 绝对不能碰的
$Protected = @(
    'System', 'Idle', 'Registry', 'Memory Compression', 'csrss', 'wininit',
    'winlogon', 'services', 'lsass', 'smss', 'svchost', 'dwm', 'explorer',
    'vmmemWSL', 'vmwp', 'WorkBuddy', 'CodeBuddy'
)

function Get-Snapshot {
    $os = Get-CimInstance Win32_OperatingSystem
    $totalMB = $os.TotalVisibleMemorySize / 1KB
    $freeMB  = $os.FreePhysicalMemory / 1KB
    $availMB = (Get-Counter '\Memory\Available MBytes').CounterSamples.CookedValue
    $commit  = (Get-Counter '\Memory\Committed Bytes').CounterSamples.CookedValue / 1GB
    $standby = (Get-Counter '\Memory\Standby Cache Normal Priority Bytes').CounterSamples.CookedValue / 1GB
    [pscustomobject]@{
        TotalGB   = [math]::Round($totalMB / 1024, 2)
        FreeGB    = [math]::Round($freeMB / 1024, 2)
        AvailGB   = [math]::Round($availMB / 1024, 2)
        UsedGB    = [math]::Round(($totalMB - $freeMB) / 1024, 2)
        CommitGB  = [math]::Round($commit, 2)
        StandbyGB = [math]::Round($standby, 2)
    }
}

function Show-Report {
    $s = Get-Snapshot

    Write-Host (" ")
    Write-Host ("=" * 62) -ForegroundColor Cyan
    Write-Host " Windows 内存诊断" -ForegroundColor Cyan
    Write-Host ("=" * 62) -ForegroundColor Cyan

    Write-Host ("  物理内存      {0,7:N2} GB" -f $s.TotalGB)
    Write-Host ("  可用          {0,7:N2} GB  (已含 standby 缓存)" -f $s.AvailGB) -ForegroundColor Green
    Write-Host ("  已用          {0,7:N2} GB" -f $s.UsedGB)
    Write-Host ("  提交          {0,7:N2} GB" -f $s.CommitGB) -ForegroundColor $(if ($s.CommitGB -gt $s.TotalGB) { 'Yellow' } else { 'Gray' })
    Write-Host ("  standby 缓存  {0,7:N2} GB  (可回收，不用管)" -f $s.StandbyGB) -ForegroundColor DarkGray

    if ($s.CommitGB -gt $s.TotalGB) {
        Write-Host "  注意：提交内存已超过物理内存，系统正在用 pagefile 换页" -ForegroundColor Yellow
    }

    # WSL 虚拟机
    Write-Host "`n  WSL 虚拟机 (vmmemWSL)" -ForegroundColor Cyan
    $vm = Get-Process vmmemWSL -ErrorAction SilentlyContinue
    if ($vm) {
        Write-Host ("    {0,7:N2} GB" -f ($vm.WorkingSet64 / 1GB))
    } else {
        Write-Host "    未运行" -ForegroundColor DarkGray
    }

    # .wslconfig
    $cfg = Join-Path $env:USERPROFILE '.wslconfig'
    Write-Host "`n  .wslconfig" -ForegroundColor Cyan
    if (Test-Path $cfg) {
        $body = ((Get-Content $cfg) -join "`n") -replace "`n", "`n    "
        Write-Host "    $body" -ForegroundColor DarkGray
    } else {
        Write-Host "    不存在（WSL 默认用主机内存的 50%）" -ForegroundColor Yellow
    }

    # Top 进程
    Write-Host "`n  内存占用 Top $Top" -ForegroundColor Cyan
    Write-Host ("    {0,-26} {1,8}  {2}" -f '进程', '内存', '标记')

    $procs = Get-Process | Sort-Object WorkingSet64 -Descending | Select-Object -First $Top
    foreach ($p in $procs) {
        $tag = ''
        $color = 'Gray'
        if ($Protected -contains $p.ProcessName) { $tag = '系统/关键'; $color = 'DarkGray' }
        elseif ($Killable -contains $p.ProcessName) { $tag = '可关闭'; $color = 'Yellow' }
        Write-Host ("    {0,-26} {1,7:N2}G  {2}" -f $p.ProcessName, ($p.WorkingSet64 / 1GB), $tag) -ForegroundColor $color
    }

    # 结论与建议
    $killableSum = (Get-Process | Where-Object { $Killable -contains $_.ProcessName } |
                    Measure-Object WorkingSet64 -Sum).Sum / 1GB

    Write-Host ("`n" + "=" * 62) -ForegroundColor Cyan
    Write-Host " 判断" -ForegroundColor Cyan
    Write-Host ("=" * 62) -ForegroundColor Cyan

    # WSL 跑 14B MoE 的峰值需求约 10.7 GB
    $need = 10.7
    $gap  = $need - $s.AvailGB

    if ($gap -le 0) {
        Write-Host ("  WSL 峰值需求约 {0:N1} GB，当前可用 {1:N1} GB → 够用，无需清理" -f $need, $s.AvailGB) -ForegroundColor Green
    } else {
        Write-Host ("  WSL 峰值需求约 {0:N1} GB，当前可用 {1:N1} GB → 缺口 {2:N1} GB" -f $need, $s.AvailGB, $gap) -ForegroundColor Yellow
        Write-Host (" ")
        Write-Host "  可选的腾内存手段（按效果排序）：" -ForegroundColor Cyan
        $vmGB = if ($vm) { $vm.WorkingSet64 / 1GB } else { 0 }
        Write-Host ("    1. wsl --shutdown          释放 {0:N1} GB（立即，最有效）" -f $vmGB)
        Write-Host ("    2. 关闭浏览器/聊天软件      约 {0:N1} GB" -f $killableSum)
        Write-Host  "    3. 清缓存（standby）        0 GB（别被骗，可用内存已含缓存）" -ForegroundColor DarkGray
        Write-Host (" ")
        Write-Host "  更实际的思路：降低 WSL 的需求，而不是硬腾 12GB" -ForegroundColor Cyan
        Write-Host "    · 关掉 Windows 里吃显存的程序 → 显存空闲变多 → offload 需求变小"
        Write-Host "    · 减小 --max-model-len（512 → 256）"
        Write-Host "    · .wslconfig 设 memory=11GB + swap=8GB（不是 12，留一点给 Windows）"
    }
}

function Invoke-Cleanup {
    Write-Host (" ")
    Write-Host ("=" * 62) -ForegroundColor Yellow
    Write-Host " 清理（只动白名单进程 + wsl --shutdown）" -ForegroundColor Yellow
    Write-Host ("=" * 62) -ForegroundColor Yellow

    $targets = Get-Process | Where-Object { $Killable -contains $_.ProcessName }

    if ($targets) {
        Write-Host (" ")
        Write-Host "  将要关闭：" -ForegroundColor Yellow
        $targets | Group-Object ProcessName | ForEach-Object {
            $mem = ($_.Group | Measure-Object WorkingSet64 -Sum).Sum / 1GB
            Write-Host ("    {0,-24} {1} 个进程  {2,6:N2} GB" -f $_.Name, $_.Count, $mem)
        }
        $freed = ($targets | Measure-Object WorkingSet64 -Sum).Sum / 1GB
        Write-Host ("    合计可释放约 {0:N2} GB" -f $freed) -ForegroundColor Yellow

        if (-not $Force) {
            $ans = Read-Host "`n  确认关闭这些进程？[y/N]"
            if ($ans -notmatch '^[yY]') {
                Write-Host "  已跳过进程关闭" -ForegroundColor DarkGray
                $targets = $null
            }
        }
        if ($targets) {
            $targets | Stop-Process -Force -ErrorAction SilentlyContinue
            Write-Host "  已关闭" -ForegroundColor Green
        }
    } else {
        Write-Host "  没有可关闭的白名单进程" -ForegroundColor DarkGray
    }

    # wsl --shutdown：这一步释放 vmmem 全部内存
    Write-Host (" ")
    Write-Host "  正在执行 wsl --shutdown（释放 WSL 虚拟机内存）..." -ForegroundColor Yellow
    & wsl.exe --shutdown 2>$null
    Start-Sleep -Seconds 3
    Write-Host "  完成" -ForegroundColor Green
    Write-Host "  提示：下次进 WSL 用 wsl -d Ubuntu-26.04 重新启动" -ForegroundColor DarkGray
}

# ── 主流程 ──
Show-Report

if ($Apply) {
    Invoke-Cleanup
    Write-Host (" ")
    Write-Host "清理后状态：" -ForegroundColor Cyan
    $s = Get-Snapshot
    Write-Host ("  可用 {0:N2} GB（清理前对比见上）" -f $s.AvailGB) -ForegroundColor Green
} else {
    Write-Host (" ")
    Write-Host "  以上只是诊断。要执行清理加 -Apply：" -ForegroundColor DarkGray
    Write-Host "    .\tools\free_mem.ps1 -Apply" -ForegroundColor DarkGray
}
