<#
Stage 2 of docs/BENCHMARK_PLAN_100.md — the local models over the 100-contract
sample, one python process per model so a crash in one does not take the rest
down. Every contract is saved as it completes, so re-running skips finished work
and resumes where it stopped.

  pwsh scripts/run_bench100.ps1                       # bge contexts (default)
  pwsh scripts/run_bench100.ps1 -Tag minilm100 -Embed sentence-transformers/all-MiniLM-L6-v2 -Chroma ./data/chroma_db_minilm100

Progress: tail output/benchmarks/locals_<Tag>.log
#>
param(
  [string]$Tag    = 'bge100',
  [string]$Embed  = 'BAAI/bge-base-en-v1.5',
  [string]$Chroma = './data/chroma_db_bge'
)

$ErrorActionPreference = 'Continue'
$env:OLLAMA_MODELS   = 'D:\ollama_models'
$env:EMBED_MODEL     = $Embed
$env:CHROMA_DB_PATH  = $Chroma
$log = "output/benchmarks/locals_$Tag.log"

# KV cache quantization is a *server* setting. Exporting it in this process does
# nothing to a server that is already running, so the 24B phase restarts one.
function Restart-Ollama([bool]$KvQuant) {
  # llama-server holds the VRAM, and force-killing its ollama parent orphans it
  # rather than reaping it. Miss these and each phase leaks a model's worth of
  # GPU memory, until the 24B cannot allocate its 11.8 GiB of weights.
  Get-Process -Name 'ollama', 'ollama app', 'llama-server' -ErrorAction SilentlyContinue |
    Stop-Process -Force
  Start-Sleep -Seconds 5
  if ($KvQuant) {
    $env:OLLAMA_FLASH_ATTENTION = '1'
    $env:OLLAMA_KV_CACHE_TYPE   = 'q8_0'
  } else {
    Remove-Item Env:OLLAMA_FLASH_ATTENTION -ErrorAction SilentlyContinue
    Remove-Item Env:OLLAMA_KV_CACHE_TYPE   -ErrorAction SilentlyContinue
  }
  Start-Process -FilePath 'ollama' -ArgumentList 'serve' -WindowStyle Hidden `
                -RedirectStandardError "output/benchmarks/ollama_server_$Tag.log"
  foreach ($i in 1..30) {
    try { Invoke-RestMethod http://localhost:11434/api/tags -TimeoutSec 3 | Out-Null; return }
    catch { Start-Sleep -Seconds 2 }
  }
  throw 'ollama server did not come up'
}

$models = @(
  @{ id = 'llama3.1:8b-instruct-q8_0'; kv = $false },
  @{ id = 'qwen3:14b-q4_K_M';          kv = $false },
  @{ id = 'hf.co/bartowski/Mistral-Small-24B-Instruct-2501-GGUF:IQ4_XS'; kv = $true }
)

foreach ($m in $models) {
  "=== $(Get-Date -Format s)  $($m.id)  kv_q8=$($m.kv) ===" | Tee-Object $log -Append
  Restart-Ollama $m.kv
  python -u scripts/benchmark_models.py --contracts 100 --seed 42 --run-tag $Tag `
         --skip-deepseek --models $m.id 2>&1 | Tee-Object $log -Append
  # 100% GPU or it spilled to CPU and the timings are meaningless
  (ollama ps | Out-String) | Tee-Object $log -Append
}

"=== $(Get-Date -Format s)  report ===" | Tee-Object $log -Append
python -u scripts/benchmark_models.py --report --run-tag $Tag 2>&1 | Tee-Object $log -Append
