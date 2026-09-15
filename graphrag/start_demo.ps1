<#
=============================================================================
  GraphRAG 规范知识图谱 Demo · 一键启动（Windows PowerShell）
=============================================================================
  一条命令起全套：Neo4j（后台）→ 等待 bolt 就绪 → 检查/写入 demo 图谱
                  → 启动 demo 前端服务（单路图谱 / 双路向量+图谱）
  本脚本只属于独立 demo，不涉及 rag2 / agent 落地链路。

  用法（项目根目录或本目录均可执行）：
    .\graphrag\start_demo.ps1                一键启动（Neo4j + 建图 + 前端，前台）
    .\graphrag\start_demo.ps1 -Background    前端也放后台 + 自动开浏览器 + 健康检查
    .\graphrag\start_demo.ps1 -Neo4jOnly     只起 Neo4j（并等 bolt 就绪）
    .\graphrag\start_demo.ps1 -Reload        强制把 demo 图谱重新写入 Neo4j
    .\graphrag\start_demo.ps1 -NoLoad        跳过“图谱是否已入库”检查
    .\graphrag\start_demo.ps1 -Open          启动后自动打开浏览器
    .\graphrag\start_demo.ps1 -Status        查看端口/图库/进程状态
    .\graphrag\start_demo.ps1 -Stop          停止 demo 前端 + Neo4j

  参数：
    -Py <python.exe>   指定解释器（默认自动探测 torch_gpu / 托管 python）
    -Port 7870         demo 前端端口
    -Timeout 150       Neo4j 就绪等待上限（秒）
    -Install           缺依赖时自动 pip install -r requirements.txt

  前置：
    ★ JDK 11（Neo4j 4.4.8 要 11，不是 17+）。缺 Java 时先设：
        $env:NEO4J_JAVA = "C:\path\to\jdk-11\bin\java.exe"
    ★ pip 里有 fastapi / uvicorn / neo4j（双路模式额外需要 torch + sentence-transformers）
    ★ Neo4j 默认账号 neo4j / neo4j123456（可用 $env:NEO4J_PASSWORD 覆盖）

  首次若提示“禁止运行脚本”，执行一次：
    Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
=============================================================================
#>
[CmdletBinding()]
param(
    [switch]$Background,
    [switch]$Neo4jOnly,
    [switch]$Reload,
    [switch]$NoLoad,
    [switch]$Open,
    [switch]$Status,
    [switch]$Stop,
    [switch]$Help,
    [string]$Py = '',
    [int]$Port = 7870,
    [int]$Timeout = 150,
    [switch]$Install
)

$ErrorActionPreference = 'Stop'

# 净化宿主注入的 PYTHONPATH（vendor\shim 会拦截导入，造成“依赖缺失”假故障）
if ($env:PYTHONPATH -and ($env:PYTHONPATH -like '*vendor\shim*' -or $env:PYTHONPATH -like '*vendor/shim*')) {
    $env:PYTHONPATH = ''
}
# 限制 BLAS 线程数（双路模式要加载 torch，避免 OpenBLAS 内存分配失败）
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'

$Here = Split-Path -Parent $MyInvocation.MyCommand.Path     # ...\graphrag
$Root = Split-Path -Parent $Here                            # 项目根
Set-Location $Root

$Neo4jHome  = Join-Path $Root 'neo4j-community-4.4.8'
$StartNeo4j = Join-Path $Neo4jHome 'start_neo4j.py'
$AppPy      = Join-Path $Here 'app.py'
$LoadPy     = Join-Path $Here 'scripts\load_neo4j.py'
$AppPidFile = Join-Path $Here '.demo_app.pid'
$AppLog     = Join-Path $Here 'demo_app.log'
$AppErrLog  = Join-Path $Here 'demo_app.err.log'

function Write-Head($m) { Write-Host ''; Write-Host "== $m ==" -ForegroundColor Cyan }
function Write-Ok($m)   { Write-Host "  [ok] $m" -ForegroundColor Green }
function Write-Info($m) { Write-Host "  $m" }
function Write-Warn2($m){ Write-Host "  [!] $m" -ForegroundColor Yellow }
function Write-Err2($m) { Write-Host "  [x] $m" -ForegroundColor Red }

function Show-Usage {
    Write-Host @'
GraphRAG demo 一键启动 —— 用法

  .\graphrag\start_demo.ps1                一键启动（Neo4j + 建图 + 前端，前台）
  .\graphrag\start_demo.ps1 -Background    前端也放后台 + 自动开浏览器 + 健康检查
  .\graphrag\start_demo.ps1 -Neo4jOnly     只起 Neo4j（并等 bolt 就绪）
  .\graphrag\start_demo.ps1 -Reload        强制把 demo 图谱重新写入 Neo4j
  .\graphrag\start_demo.ps1 -NoLoad        跳过“图谱是否已入库”检查
  .\graphrag\start_demo.ps1 -Open          启动后自动打开浏览器
  .\graphrag\start_demo.ps1 -Status        查看端口 / 图库 / 依赖状态
  .\graphrag\start_demo.ps1 -Stop          停止 demo 前端 + Neo4j
  .\graphrag\start_demo.ps1 -Help          本帮助

参数： -Py <python.exe>  -Port 7870  -Timeout 150  -Install

前置： JDK 11（Neo4j 4.4.8 要 11，不是 17+）；pip 有 fastapi / uvicorn / neo4j
       缺 Java 时先设： $env:NEO4J_JAVA = "C:\path\to\jdk-11\bin\java.exe"
'@
}

# ------------------------------------------------------------------ 解释器探测
function Resolve-Python {
    $cands = @()
    if ($Py) { $cands += $Py }
    $cands += @(
        (Join-Path $env:USERPROFILE 'anaconda3\envs\torch_gpu\python.exe')
        'C:\Users\<用户名>\anaconda3\envs\torch_gpu\python.exe'
        (Join-Path $env:USERPROFILE '.workbuddy\binaries\python\envs\default\Scripts\python.exe')
        'C:\Users\<用户名>\.workbuddy\binaries\python\versions\3.13.12\python.exe'
        'C:\Users\<用户名>\.workbuddy\binaries\python\envs\default\Scripts\python.exe'
    )
    foreach ($c in $cands) { if ($c -and (Test-Path $c)) { return $c } }
    $g = Get-Command python -ErrorAction SilentlyContinue
    if ($g) { return $g.Source }
    return $null
}

$Python = Resolve-Python
if (-not $Python) {
    Write-Err2 '未找到 Python 解释器。请用 -Py "C:\path\to\python.exe" 指定。'
    exit 1
}

# 静默执行 python 片段。
# 坑：native 命令往 stderr 写内容时，PS 5.1 会包成 NativeCommandError 进错误流；
# 在 $ErrorActionPreference='Stop' 下会被升级为终止性错误，把脚本打断。
# 这里临时放宽 EAP 并丢弃 stderr，只回传 stdout 文本与退出码。
function Invoke-PyQuiet([string]$code) {
    $prev = $ErrorActionPreference
    $ErrorActionPreference = 'SilentlyContinue'
    try {
        $raw = & $Python -c $code 2>$null
        $rc = $LASTEXITCODE
        $text = (@($raw) -join "`n").Trim()
        return [pscustomobject]@{ ok = ($rc -eq 0); text = $text }
    } finally { $ErrorActionPreference = $prev }
}

function Test-Module($name) {
    return (Invoke-PyQuiet "import $name").ok
}

# ------------------------------------------------------------------ 端口/进程
function Test-Port([int]$p) {
    try {
        $cli = New-Object System.Net.Sockets.TcpClient
        $iar = $cli.BeginConnect('127.0.0.1', $p, $null, $null)
        $okc = $iar.AsyncWaitHandle.WaitOne(400, $false)
        $connected = ($okc -and $cli.Connected)
        $cli.Close()
        return $connected
    } catch { return $false }
}

function Get-PortOwner([int]$p) {
    $c = Get-NetTCPConnection -LocalPort $p -State Listen -ErrorAction SilentlyContinue
    if ($c) { return ($c.OwningProcess | Select-Object -Unique) }
    return @()
}

function Stop-ByPort([int]$p, [string]$name) {
    $owners = Get-PortOwner $p
    if (-not $owners) { Write-Info "端口 $p 无监听（$name 未在跑）"; return $false }
    $done = $false
    foreach ($procId in $owners) {
        try {
            Stop-Process -Id $procId -Force -ErrorAction Stop
            Write-Ok "已停止 $name（端口 $p，PID $procId）"
            $done = $true
        } catch { Write-Warn2 "停止 PID $procId 失败：$_" }
    }
    return $done
}

# ------------------------------------------------------------------ Neo4j
function Test-Neo4jBolt {
    if (-not (Test-Port 7687)) { return $false }
    return (Invoke-PyQuiet "import sys;sys.path.insert(0,'graphrag');from src.neo4j_store import Neo4jStore;s=Neo4jStore();s.close()").ok
}

function Wait-Neo4j([int]$Seconds = 150) {
    $t0 = Get-Date
    Write-Info '等待 Neo4j bolt(7687) 就绪 ...'
    while (((Get-Date) - $t0).TotalSeconds -lt $Seconds) {
        if (Test-Neo4jBolt) { Write-Host ''; return $true }
        Write-Host '.' -NoNewline -ForegroundColor DarkGray
        Start-Sleep -Seconds 2
    }
    Write-Host ''
    return $false
}

function Get-GraphNodeCount {
    $r = Invoke-PyQuiet "import sys;sys.path.insert(0,'graphrag');from src.neo4j_store import Neo4jStore;s=Neo4jStore();print(s.stats()['nodes']);s.close()"
    if (-not $r.ok -or -not $r.text) { return -1 }
    $last = ($r.text -split "`n")[-1].Trim()
    $n = 0
    if ([int]::TryParse($last, [ref]$n)) { return $n }
    return -1
}

function Start-Neo4j {
    if (Test-Neo4jBolt) { Write-Ok 'Neo4j 已在运行（bolt://localhost:7687）'; return $true }
    if (-not (Test-Path $StartNeo4j)) { Write-Err2 "找不到 $StartNeo4j"; return $false }

    Write-Head '启动 Neo4j（后台）'
    & $Python $StartNeo4j --background
    if ($LASTEXITCODE -ne 0) {
        Write-Err2 'Neo4j 启动失败。多为缺 JDK 11（Neo4j 4.4.8 要 11，不是 17+）。'
        Write-Info '  修复：装 JDK 11 后设  $env:NEO4J_JAVA = "C:\path\to\jdk-11\bin\java.exe"'
        Write-Info '  日志： neo4j-community-4.4.8\neo4j.stdout.log'
        return $false
    }
    if (Wait-Neo4j $Timeout) { Write-Ok 'Neo4j 就绪'; return $true }
    Write-Head ''
    Write-Err2 "等待 $Timeout 秒仍不可达。看日志： $Neo4jHome\neo4j.stdout.log"
    return $false
}

# ------------------------------------------------------------------ 建图
function Invoke-LoadGraph {
    Write-Head 'demo 图谱入库检查'
    $n = Get-GraphNodeCount
    if ($n -lt 0) {
        Write-Warn2 '无法读取图库统计（Neo4j 未就绪或缺 neo4j 驱动）。跳过建图。'
        Write-Info ('  装驱动： & "{0}" -m pip install neo4j' -f $Python)
        return
    }
    if ($Reload -or $n -eq 0) {
        if ($n -eq 0) { Write-Info ' 图库为空，写入 demo 图谱 ...' } else { Write-Info ' -Reload：强制重写 demo 图谱 ...' }
        & $Python $LoadPy
        if ($LASTEXITCODE -ne 0) { Write-Err2 '图谱写入失败，见上方输出' }
        else { Write-Ok '图谱已写入（应见：56 节点 / 96 关系）' }
    } else {
        Write-Ok "图谱已入库：$n 节点（要强制重建加 -Reload）"
    }
}

# ------------------------------------------------------------------ 前端
function Show-DemoCheatSheet {
    Write-Head '演示速查'
    Write-Info "前端      : http://127.0.0.1:$Port/"
    Write-Info "Neo4j     : http://localhost:7474   （neo4j / neo4j123456）"
    Write-Info '单路演示  : 选「单路 · 图谱」，问 → 深基坑开挖前要做哪些安全准备？'
    Write-Info '双路演示  : 选「双路 · 向量+图谱」，勾/取消「启用向量端」对比命中来源（图谱/向量/双源）'
    Write-Info '图谱展示  : Neo4j Browser 里粘贴'
    Write-Info '            MATCH (n)-[r]->(m) RETURN n,r,m LIMIT 300'
    Write-Info '            MATCH p=(h:HazardCategory)-[:REGULATED_BY]->(:Standard) RETURN p'
    Write-Info '            MATCH p=(h:HazardCategory)-[:HAS_METRIC]->(:Metric)-[:HAS_THRESHOLD]->(:Threshold)-[:TRIGGERS]->(:Obligation) RETURN p'
}

function Start-App {
    $appArgs = @($AppPy)
    if ($Background) {
        if (Test-Port $Port) { Write-Warn2 "端口 $Port 已被占用，跳过启动（先 -Stop 或换 -Port）"; return $true }
        Write-Head '启动 demo 服务（后台）'
        $p = Start-Process -FilePath $Python -ArgumentList $appArgs -WorkingDirectory $Root `
             -PassThru -WindowStyle Hidden -RedirectStandardOutput $AppLog -RedirectStandardError $AppErrLog
        "$($p.Id)" | Set-Content -Path $AppPidFile -Encoding ascii
        $t0 = Get-Date
        while (((Get-Date) - $t0).TotalSeconds -lt 60) {
            if (Test-Port $Port) { break }
            Start-Sleep -Milliseconds 800
        }
        if (Test-Port $Port) {
            Write-Ok "服务已就绪（PID $($p.Id)）"
            try {
                $h = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/api/health" -TimeoutSec 5
                if ($h.ok) { Write-Ok ("Neo4j 连通，图库 {0} 节点 / {1} 关系" -f $h.stats.nodes, $h.stats.edges) }
                else { Write-Warn2 ("健康检查未通过：" + $h.error) }
            } catch { Write-Warn2 "健康检查失败：$_" }
        } else {
            Write-Err2 "服务未在 60 秒内监听 $Port，看日志： $AppErrLog"
            return $false
        }
    } else {
        Write-Head '启动 demo 服务（前台，Ctrl+C 停止）'
        Write-Info '提示：Neo4j 在后台继续运行；全停用  .\graphrag\start_demo.ps1 -Stop'
        Show-DemoCheatSheet
        & $Python @appArgs
        return ($LASTEXITCODE -eq 0)
    }
    return $true
}

function Open-Urls {
    try { Start-Process "http://127.0.0.1:$Port/" | Out-Null; Write-Ok "已打开前端：http://127.0.0.1:$Port/" } catch { Write-Warn2 "打开前端失败：$_" }
    try { Start-Process 'http://localhost:7474' | Out-Null; Write-Ok '已打开 Neo4j Browser：http://localhost:7474' } catch { Write-Warn2 "打开 Neo4j Browser 失败：$_" }
}

# ------------------------------------------------------------------ 状态 / 停止
function Show-Status {
    Write-Head 'GraphRAG demo 状态'
    Write-Info ("解释器  : {0}" -f $Python)
    foreach ($pair in @(@(7687, 'Neo4j Bolt'), @(7474, 'Neo4j HTTP'), @($Port, 'Demo 前端'))) {
        if (Test-Port $pair[0]) {
            $owners = Get-PortOwner $pair[0]
            Write-Ok ("{0} ({1}) : 监听中，PID {2}" -f $pair[1], $pair[0], ($owners -join ','))
        } else {
            Write-Warn2 ("{0} ({1}) : 未监听" -f $pair[1], $pair[0])
        }
    }
    if (Test-Path $AppPidFile) {
        $apid = (Get-Content $AppPidFile -ErrorAction SilentlyContinue | Select-Object -First 1)
        $alive = $false
        if ($apid) { $alive = [bool](Get-Process -Id $apid -ErrorAction SilentlyContinue) }
        if ($alive) { Write-Ok "后台前端进程 : PID $apid（存活）" } else { Write-Warn2 "后台前端进程 : PID $apid（已退出；日志 $AppErrLog）" }
    }
    $n = Get-GraphNodeCount
    if ($n -ge 0) { Write-Ok "图库 : $n 节点 / $(if ($n -gt 0) { '已建图' } else { '空，需建图' })" }
    else { Write-Warn2 '图库 : 读不到（Neo4j 未起或缺 neo4j 驱动）' }

    Write-Head '依赖'
    foreach ($m in @('fastapi', 'uvicorn', 'neo4j')) {
        if (Test-Module $m) { Write-Ok "$m 可用" } else { Write-Warn2 "$m 缺失： & `"$Python`" -m pip install $m" }
    }
    if (Test-Module 'torch') { Write-Ok 'torch 可用（双路向量端可开）' } else { Write-Warn2 'torch 缺失：双路模式将不可用，仅单路图谱' }
}

function Invoke-Stop {
    Write-Head '停止 GraphRAG demo'
    [void](Stop-ByPort $Port 'Demo 前端')
    if (Test-Path $AppPidFile) { Remove-Item $AppPidFile -Force -ErrorAction SilentlyContinue }
    [void](Stop-ByPort 7687 'Neo4j Bolt')
    [void](Stop-ByPort 7474 'Neo4j HTTP')
    Start-Sleep -Seconds 1
    foreach ($p in @(7687, 7474, $Port)) {
        if (Test-Port $p) { Write-Warn2 "端口 $p 仍在监听（可能需手动结束进程）" }
    }
    Write-Info '提示：Neo4j 被强杀后下次启动会自动做事务恢复，属正常现象。'
}

# ------------------------------------------------------------------ 依赖体检
function Test-Deps {
    $missing = @()
    foreach ($m in @('fastapi', 'uvicorn', 'neo4j')) { if (-not (Test-Module $m)) { $missing += $m } }
    if ($missing.Count -eq 0) { return $true }
    Write-Warn2 ("缺依赖：{0}" -f ($missing -join ', '))
    if ($Install) {
        Write-Head '安装依赖'
        & $Python -m pip install -r (Join-Path $Root 'requirements.txt')
        if ($LASTEXITCODE -ne 0) { Write-Err2 'pip install 失败，检查网络/镜像'; return $false }
        Write-Ok '依赖安装完成'
        return $true
    }
    Write-Info ('修复： & "{0}" -m pip install {1}' -f $Python, ($missing -join ' '))
    Write-Info '   或： .\graphrag\start_demo.ps1 -Install'
    return $false
}

# ------------------------------------------------------------------ 主流程
if ($Help) { Show-Usage; exit 0 }

Write-Head '环境'
Write-Ok "解释器：$Python"
$ver = Invoke-PyQuiet "import sys;print(sys.version.split()[0])"
if ($ver.ok) { Write-Info "版本：$($ver.text)" }
if ($Python -like '*torch_gpu*') { Write-Info '后端：torch_gpu（双路向量端可用）' }

if ($Status) { Show-Status; exit 0 }
if ($Stop)   { Invoke-Stop; exit 0 }

if (-not (Test-Path $Neo4jHome)) { Write-Err2 "找不到 Neo4j 目录：$Neo4jHome"; exit 1 }
if (-not (Test-Path $AppPy))     { Write-Err2 "找不到 demo 服务：$AppPy"; exit 1 }

if ($Neo4jOnly) {
    # 只起 Neo4j 时不该被 python 依赖挡住（服务本身只需 Java）
    if (-not (Test-Module 'neo4j')) { Write-Warn2 '缺 neo4j 驱动：Neo4j 服务能起，但建图/图库统计不可用' }
} else {
    if (-not (Test-Deps)) { exit 1 }
    if (-not (Test-Module 'torch')) { Write-Warn2 '无 torch：双路模式的向量端不可用，单路图谱不受影响' }
}

if (-not (Start-Neo4j)) { exit 1 }

if ($Neo4jOnly) {
    Write-Head '仅 Neo4j 模式'
    Write-Ok 'Neo4j 已就绪：Bolt 7687 / Browser http://localhost:7474（neo4j / neo4j123456）'
    Write-Info '未启动前端。要起前端： .\graphrag\start_demo.ps1'
    exit 0
}
if (-not $NoLoad) { Invoke-LoadGraph }

$ok = Start-App
if ($ok -and $Background) { Show-DemoCheatSheet; if ($Open) { Open-Urls } }
exit 0
