<#
  build_dll.ps1 — reproducible host-DLL build for OptixDemoTOP (Release|x64, v142).
  This is the exact command verified working on the dev machine (2026-06-29).

  Host-only edits (OptixDemoTOP.cpp / .h / RRDenoiser.cpp) -> run this.
  Device edits (demo_programs.cu / LaunchParamsDemo.h) -> run compile_ptx.ps1 first.

  TouchDesigner holds a lock on the loaded DLL. Release it before building:
      op('/project1/optixDemo/PT_Render/render').par.unloadplugin = True
  ...then after this script:
      op(...).par.unloadplugin = False ; op(...).par.reinitpulse.pulse()
#>
[CmdletBinding()]
param(
  [string]$MSBuild = "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\MSBuild\Current\Bin\MSBuild.exe",
  [string]$CudaRoot = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
)
$ErrorActionPreference = 'Stop'
$proj = Join-Path $PSScriptRoot '..\phase2\OptixDemoTOP\OptixDemoTOP.sln' | Resolve-Path

if (-not (Test-Path $MSBuild))  { throw "MSBuild not found: $MSBuild (install VS2019 BuildTools + v142, or pass -MSBuild)" }
if (-not (Test-Path $CudaRoot)) { throw "CUDA 12.8 not found: $CudaRoot (install CUDA 12.8, or pass -CudaRoot)" }

Write-Host "Building $proj (Release|x64, v142)..." -ForegroundColor Cyan
& $MSBuild $proj `
  -p:Configuration=Release -p:Platform=x64 `
  -p:CudaToolkitIncludeDir="$CudaRoot\include" `
  -p:CudaToolkitLibDir="$CudaRoot\lib\x64" `
  -nologo -m -v:minimal
if ($LASTEXITCODE -ne 0) { throw "MSBuild failed (exit $LASTEXITCODE)" }

$dll = Join-Path $PSScriptRoot '..\phase2\OptixDemoTOP\Release\OptixDemoTOP.dll' | Resolve-Path
Write-Host "OK -> $dll" -ForegroundColor Green
Get-Item $dll | Select-Object Length, LastWriteTime
