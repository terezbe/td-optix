<#
  stage_runtime.ps1 — assemble a self-contained, shareable runtime folder.

  Produces  dist\PT_Render\  containing ONLY the files a recipient needs at runtime:
      OptixDemoTOP.dll      the plugin (loads the rest relative to itself)
      demo_programs.ptx     OptiX device pipeline
      nvngx_dlssd.dll       DLSS Ray Reconstruction runtime (NVIDIA redistributable)
      cudart64_12.dll       CUDA 12.x runtime
  Drop PT_Render.tox into the same folder (see scripts note) and the bundle is portable:
  the plugin resolves PTX + nvngx beside its own DLL; the .tox resolves the DLL beside itself.

  OptiX itself ships inside the NVIDIA display driver (no redistributable DLL).
#>
[CmdletBinding()]
param(
  [string]$CudaRoot = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8",
  [string]$OutDir   = ""   # default: <repo>\dist\PT_Render
)
$ErrorActionPreference = 'Stop'
$repo    = Join-Path $PSScriptRoot '..' | Resolve-Path
$release = Join-Path $repo 'phase2\OptixDemoTOP\Release'
if (-not $OutDir) { $OutDir = Join-Path $repo 'dist\PT_Render' }

$items = @(
  @{ src = Join-Path $release 'OptixDemoTOP.dll';  req = $true  },
  @{ src = Join-Path $release 'demo_programs.ptx'; req = $true  },
  @{ src = Join-Path $release 'nvngx_dlssd.dll';   req = $true  },
  @{ src = Join-Path $CudaRoot 'bin\cudart64_12.dll'; req = $true }
)

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
Write-Host "Staging runtime -> $OutDir" -ForegroundColor Cyan
foreach ($i in $items) {
  if (-not (Test-Path $i.src)) {
    if ($i.req) { throw "MISSING required runtime file: $($i.src)" } else { continue }
  }
  Copy-Item $i.src $OutDir -Force
  $f = Get-Item (Join-Path $OutDir (Split-Path $i.src -Leaf))
  Write-Host ("  {0,-22} {1,12:N0} bytes" -f $f.Name, $f.Length) -ForegroundColor Green
}
Write-Host "Done. Add PT_Render.tox to this folder to complete the shareable bundle." -ForegroundColor Cyan
Write-Host "Recipient: RTX GPU (Ampere+), NVIDIA driver >= 590, Windows x64, TouchDesigner 2025.32820+." -ForegroundColor DarkGray
