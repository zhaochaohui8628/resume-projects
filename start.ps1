<#
=============================================================================
  施工方案合规审查 · 快速启动脚本（Windows PowerShell）
=============================================================================
  用法（在项目根目录执行）：
    .\start.ps1                     环境自检 + 启动 Web UI（默认）
    .\start.ps1 check               只做体检：依赖 / 数据 / 模型 / 冒烟检索
    .\start.ps1 ui                  只启动 Web UI（http://127.0.0.1:7860/）
    .\start.ps1 test                跑全量回归测试（agent+rag2+ner2+grpo）
    .\start.ps1 review .\方案.txt    CLI 合规自查一份方案 -> agent/outputs/report.md
    .\start.ps1 demo                用内置样例方案演示 CLI 自查
    .\start.ps1 help                打印帮助

  可选参数：
    -Port 8000                      改 UI 端口（默认 7860）
    -Py "C:\...\python.exe"         手动指定 Python 解释器
    -Install                        体检/启动前缺依赖时自动 pip install
    -Llm                            review 时走真实 DeepSeek（需先设 DEEPSEEK_API_KEY）

  首次使用若提示"禁止运行脚本"，先执行一次：
    Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
=============================================================================
#>
[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [ValidateSet('start', 'check', 'ui', 'test', 'review', 'demo', 'help')]
    [string]$Action = 'start',

    [Parameter(Position = 1)]
    [string]$File = '',

    [int]$Port = 7860,
    [string]$Py = '',
    [switch]$Install,
    [switch]$Llm
)

$ErrorActionPreference = 'Stop'

# 宿主注入的 PYTHONPATH（例如 workbuddy 的 vendor\shim）会带一个 sitecustomize.py 拦截部分导入，
# 表现为 torch/transformers "ModuleNotFoundError: No module named 'typing_extensions'" 之类的假故障。
# 这里做一次净化，避免在注入环境里误判"依赖缺失"。
if ($env:PYTHONPATH -and ($env:PYTHONPATH -like '*vendor\shim*' -or $env:PYTHONPATH -like '*vendor/shim*')) {
    $env:PYTHONPATH = ''
}

# 限制 BLAS / OpenMP 线程数。
# 原因：本机 16 核 + 16G 内存，OpenBLAS 默认按核数分配线程缓冲，在「加载 torch +
# sentence-transformers + FAISS 索引」时会报 "Memory allocation still failed after 10 retries"
# 或 faiss `std::bad_alloc`。设为 1 后内存占用线性下降，检索/自检/NER 均正常。
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'

# 中文输出统一 UTF-8：Python 侧写 UTF-8，而 PowerShell 5.1 默认按系统代码页(GBK)解码 native
# stdout → 实测出现「鐩綍濂戠害」这类乱码。两边都钉成 UTF-8 后一致。
$env:PYTHONIOENCODING = 'utf-8'
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch { }

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

function Write-Head($m) { Write-Host ''; Write-Host "== $m ==" -ForegroundColor Cyan }
function Write-Ok($m) { Write-Host "  [ok] $m" -ForegroundColor Green }
function Write-Info($m) { Write-Host "  $m" }
function Write-Warn2($m) { Write-Host "  [!] $m" -ForegroundColor Yellow }
function Write-Err2($m) { Write-Host "  [x] $m" -ForegroundColor Red }

function Show-Help {
    Write-Head '施工方案合规审查 · 快速启动'
    Write-Host @'
  .\start.ps1                    环境自检 + 启动 Web UI（默认）
  .\start.ps1 check              只做体检（依赖/数据/模型/冒烟检索）
  .\start.ps1 ui                 只启动 Web UI
  .\start.ps1 test               跑全量回归测试
  .\start.ps1 review <方案文件>   CLI 合规自查 -> agent/outputs/report.md
  .\start.ps1 demo               用内置样例方案演示 CLI 自查
  .\start.ps1 help               本帮助

  参数： -Port <n>  -Py <python路径>  -Install  -Llm

  [注意] 只有 -Install 会改动 Python 环境（pip install 依赖）；不带它时脚本只读，不改环境。
  [注意] 本脚本会自动设置 PYTHONPATH 净化与 OPENBLAS/OMP 线程数=1（见文件头注释）。

  直接用 Python 手动跑（等价）：
    & $PY agent/src/fastapi_app.py
    & $PY agent/scripts/run_check.py agent/tests/fixtures/sample_plan.txt
'@
}

# ------------------------------------------------------------------ 解释器探测
function Invoke-PyQuiet([string]$exe, [string]$code) {
    # 探测型 python 调用：native 命令往 stderr 写内容时，PS 5.1 会包成 NativeCommandError，
    # 在 $ErrorActionPreference='Stop' 下被升级为终止性错误（`2>$null` 拦不住）→
    # 缺依赖时脚本会"莫名中断"而不是返回 false。这里临时放宽 EAP 再丢弃 stderr。
    $prev = $ErrorActionPreference
    $ErrorActionPreference = 'SilentlyContinue'
    try {
        $null = & $exe -c $code 2>$null
        return ($LASTEXITCODE -eq 0)
    } finally { $ErrorActionPreference = $prev }
}

function Resolve-Python {
    $cands = @()
    if ($Py) { $cands += $Py }
    $cands += @(
        'C:\Users\<用户名>\anaconda3\envs\torch_gpu\python.exe'
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

# 限制 BLAS 线程数为 1：避免 numpy(OpenBLAS) 按 CPU 核数分配内存，
# 在加载 torch + sentence-transformers 时报 "OpenBLAS: Memory allocation failed"。
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'

if ($Action -eq 'help') { Show-Help; exit 0 }

Write-Head '环境'
Write-Ok "解释器：$Python"
$verLine = & $Python -c "import sys;print(sys.version.split()[0])" 2>$null
Write-Info "版本：$verLine"
if ($Python -like '*envs\torch_gpu*') { Write-Info '后端：torch_gpu（CUDA 可用）' }
elseif ($Python -like '*anaconda3\python.exe') { Write-Warn2 'base 环境是 CPU 版 torch，建议改用 torch_gpu 环境' }

# ------------------------------------------------------------------ 依赖检查
function Test-Module($name) {
    return (Invoke-PyQuiet $Python "import $name")
}

function Install-Deps {
    Write-Head '安装依赖'
    $req = Join-Path $Root 'requirements.txt'
    & $Python -m pip install -r $req
    if ($LASTEXITCODE -ne 0) { Write-Err2 'pip install 失败，请检查网络/镜像。'; exit 1 }
    Write-Ok '依赖安装完成'
}

# ------------------------------------------------------------------ 产物预检
function Test-Artifacts {
    Write-Head '关键产物'
    $need = @(
        @{ p = 'rag2/data/corpus/clauses.jsonl'; d = '条款语料' }
        @{ p = 'rag2/data/index/units.jsonl'; d = '检索单元' }
        @{ p = 'rag2/data/index/bm25.json'; d = 'BM25 索引' }
        @{ p = 'agent/data/rules/abolished_clauses.json'; d = 'C1 废止条文规则库' }
        @{ p = 'agent/data/rules/hazardous_work_types.json'; d = 'C2 危大阈值规则库' }
        @{ p = 'agent/data/rules/required_sections.json'; d = 'C4 九章要素规则库' }
        @{ p = 'agent/src/ui/index.html'; d = 'UI 前端' }
    )
    $missing = 0
    foreach ($n in $need) {
        if (Test-Path (Join-Path $Root $n.p)) { Write-Ok $n.d }
        else { Write-Err2 ("缺失：{0}（{1}）" -f $n.p, $n.d); $missing++ }
    }

    $idx = Join-Path $Root 'rag2/data/index'
    $tower = ''
    foreach ($t in @('dual_mix', 'dual_gold', 'tower_base')) {
        if ((Test-Path (Join-Path $idx "$t.faiss")) -and (Test-Path (Join-Path $Root "rag2/data/models/$t/doc_encoder"))) {
            $tower = $t; break
        }
        if ($t -eq 'tower_base' -and (Test-Path (Join-Path $idx "$t.faiss"))) { $tower = $t; break }
    }
    if ($tower) { Write-Ok "向量索引/双塔：$tower" }
    else { Write-Warn2 '未找到可用的 .faiss 索引，检索将不可用（判档路仍可用）' }

    if (Test-Path (Join-Path $Root 'ner2/models/s2_crf_param_v3')) { Write-Ok 'NER 模型：s2_crf_param_v3' }
    else { Write-Warn2 '未找到 ner2 模型，实体抽取将退化为纯规则层' }

    if (Test-Path (Join-Path $Root 'rag2/data/models/cross_v2_ep4')) { Write-Ok '精排模型：cross_v2_ep4' }
    else { Write-Warn2 '未找到 CrossEncoder 精排产物，检索将跳过精排' }

    return $missing
}

function Test-ApiKey {
    if ($env:DEEPSEEK_API_KEY) { Write-Ok 'DEEPSEEK_API_KEY 已设置（LLM 环节走真实推理）' }
    else { Write-Warn2 '未设 DEEPSEEK_API_KEY：LLM 路由/汇总走 mock；检索/规则/判档/NER 不受影响' }
}

# ------------------------------------------------------------------ 动作：check
function Invoke-Check {
    Write-Head '环境自检（env/bootstrap.py）'
    $a = @((Join-Path $Root 'env/bootstrap.py'), 'check')
    if ($Install) { $a += '--install' }
    & $Python @a
    $code = $LASTEXITCODE
    [void](Test-Artifacts)
    Write-Head '自检结论'
    switch ($code) {
        0 { Write-Ok '全通过（退出码 0）' }
        2 { Write-Warn2 '可用但有告警（退出码 2），见上方 [warn] 项' }
        default { Write-Err2 "有致命问题（退出码 $code），见上方输出" }
    }
    exit $code
}

# ------------------------------------------------------------------ 动作：test
function Invoke-Test {
    if (-not (Test-Module 'pytest')) {
        Write-Err2 '缺 pytest，无法跑测试。'
        Write-Info ('请确认后手动安装：  & "{0}" -m pip install pytest' -f $Python)
        exit 1
    }
    Write-Head '全量回归测试（分模块跑）'
    Write-Info '说明：四个 tests/ 目录同名，pytest 一次全给会互相覆盖收集，故逐个跑。'
    $failed = 0
    foreach ($t in @('agent/tests', 'rag2/tests', 'ner2/tests', 'grpo/tests')) {
        Write-Host ''
        Write-Host "-- $t" -ForegroundColor DarkCyan
        & $Python -m pytest $t -q
        if ($LASTEXITCODE -ne 0) { $failed++ }
    }
    Write-Head '测试结论'
    if ($failed -eq 0) { Write-Ok '全部模块通过' }
    else { Write-Err2 ('{0} 个模块存在失败，见上方摘要' -f $failed) }
    exit $failed
}

# ------------------------------------------------------------------ 动作：review / demo
function Invoke-Review($target) {
    if (-not $target) {
        Write-Err2 '请给出方案文件，例如： .\start.ps1 review .\方案.txt'
        exit 1
    }
    if (-not (Test-Path $target)) { Write-Err2 "文件不存在：$target"; exit 1 }
    Write-Head "合规自查：$target"
    $a = @((Join-Path $Root 'agent/scripts/run_check.py'), $target)
    if ($Llm) { $a += '--deepseek' }
    & $Python @a
    $code = $LASTEXITCODE
    $rep = Join-Path $Root 'agent/outputs/report.md'
    if (Test-Path $rep) { Write-Ok "报告已生成：$rep" }
    exit $code
}

# ------------------------------------------------------------------ 动作：ui / start
function Invoke-Ui {
    if (-not (Test-Module 'fastapi')) {
        Write-Warn2 '缺 fastapi。请先执行： .\start.ps1 check -Install'
        if ($Install) { Install-Deps } else { exit 1 }
    }
    if (-not (Test-Module 'uvicorn')) {
        Write-Warn2 '缺 uvicorn。请先执行： .\start.ps1 check -Install'
        if ($Install) { Install-Deps } else { exit 1 }
    }
    $env:AGENT_UI_PORT = "$Port"
    Write-Head 'Web UI'
    Write-Info "地址：http://127.0.0.1:$Port/   （浏览器将自动打开；Ctrl+C 停止）"
    Write-Info '提示：上传方案支持 .txt / .docx / .pdf'
    & $Python (Join-Path $Root 'agent/src/fastapi_app.py')
    exit $LASTEXITCODE
}

# ------------------------------------------------------------------ 分发
switch ($Action) {
    'check' { Invoke-Check }
    'test' { Invoke-Test }
    'review' { Invoke-Review $File }
    'demo' { Invoke-Review 'agent/tests/fixtures/sample_plan.txt' }
    'ui' { Invoke-Ui }
    'start' {
        Write-Head '快速自检'
        [void](Test-Artifacts)
        Test-ApiKey
        if (-not (Test-Module 'fastapi')) {
            Write-Warn2 '缺 fastapi/uvicorn ——UI 无法启动。'
            Write-Info '执行： .\start.ps1 check -Install   或   & $PY -m pip install -r requirements.txt'
            exit 1
        }
        Invoke-Ui
    }
}
