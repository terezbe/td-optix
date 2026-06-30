<#
  compile_ptx.ps1 — compile the OptiX device code to PTX (only needed when
  demo_programs.cu or LaunchParamsDemo.h change). Outputs demo_programs.ptx in the
  source dir AND stages a copy into Release\ (the plugin loads it relative to itself).

  After this, hot-reload picks up the new PTX with no DLL rebuild:
      op(...).par.unloadplugin = False ; op(...).par.reinitpulse.pulse()
#>
[CmdletBinding()]
param(
  [string]$CudaRoot = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8",
  [string]$Ccbin    = "",            # host cl.exe dir for nvcc; auto-detected if empty
  # Virtual arch for the PTX. compute_75 (Turing) is the portable baseline: PTX JIT-compiles
  # UPWARD, so one build runs on Turing/Ampere(30xx)/Ada(40xx)/Blackwell with no perf loss
  # (the driver JITs to the GPU's native SASS at load). Do NOT use sm_89 — that locks it to Ada.
  [string]$Arch     = "compute_75"
)
$ErrorActionPreference = 'Stop'
$src     = Join-Path $PSScriptRoot '..\phase2\OptixDemoTOP' | Resolve-Path
$nvcc    = Join-Path $CudaRoot 'bin\nvcc.exe'
$optixInc = Join-Path $src '..\optix-dev\include'

if (-not (Test-Path $nvcc)) { throw "nvcc not found: $nvcc" }

# Auto-detect a host compiler nvcc accepts (prefer VS2019 v142, then VS2022).
if (-not $Ccbin) {
  $cands = @(
    "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Tools\MSVC\*\bin\Hostx64\x64",
    "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\*\bin\Hostx64\x64"
  )
  foreach ($c in $cands) { $hit = Get-ChildItem $c -Directory -ErrorAction SilentlyContinue | Select-Object -First 1; if ($hit) { $Ccbin = $hit.FullName; break } }
}
if (-not $Ccbin) { throw "No host cl.exe dir found; pass -Ccbin '<...\\Hostx64\\x64>'" }

Push-Location $src
try {
  Write-Host "nvcc -ptx -arch=$Arch (ccbin: $Ccbin)" -ForegroundColor Cyan
  & $nvcc -ptx "-arch=$Arch" -allow-unsupported-compiler `
      -ccbin "$Ccbin" -I "$optixInc" `
      demo_programs.cu -o demo_programs.ptx
  if ($LASTEXITCODE -ne 0) { throw "nvcc failed (exit $LASTEXITCODE)" }
  Copy-Item demo_programs.ptx 'Release\demo_programs.ptx' -Force
  Write-Host "OK -> demo_programs.ptx (staged into Release\)" -ForegroundColor Green
} finally { Pop-Location }
