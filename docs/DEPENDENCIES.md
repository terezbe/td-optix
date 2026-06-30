# Dependencies & System Requirements — PT_Render (OptixDemoTOP)

What you need to **run** the shareable bundle, and what you need to **rebuild** it.

---

## A. Runtime requirements (to USE the `.tox` bundle)

| Requirement | Minimum | Notes |
|---|---|---|
| GPU | NVIDIA **RTX** (Turing / Ampere / **Ada** / Blackwell) | PTX built for `compute_75` (Turing) → JIT-compiles up to **all** RTX archs. Non-RTX GPUs cannot run OptiX RTX path tracing. |
| Driver | **≥ 590** for DLSS Ray Reconstruction; any recent driver for the OptiX-denoiser path | If the driver is too old for DLSS-RR, the renderer **auto-falls back** to the OptiX AI denoiser (then raw) — see below. |
| CUDA runtime | 12.x (`cudart64_12.dll`, **bundled**) | Built against CUDA 12.8. |
| OptiX | **none to install** — ships inside the NVIDIA display driver | The plugin loads OptiX entry points from the driver at runtime. |
| OS | Windows x64 | Only Release\|x64 is built. |
| Host app | TouchDesigner **2025.32820+** | C++ TOP, CUDA execute mode. |
| VRAM | resolution-dependent (~11×W·H·float4 + DLSS internal buffers) | 8 GB comfortably covers 1080p; 4K needs more. |

### Bundled runtime files (assembled by `scripts/stage_runtime.ps1` into `dist/PT_Render/`)
- `OptixDemoTOP.dll` — the plugin. Resolves the two files below **relative to its own location**.
- `demo_programs.ptx` — OptiX device pipeline (must sit beside the DLL).
- `nvngx_dlssd.dll` — DLSS Ray Reconstruction runtime (NVIDIA-signed redistributable; must sit beside the DLL).
- `cudart64_12.dll` — CUDA 12.x runtime.
- `PT_Render.tox` — the renderer COMP. Resolves the DLL **relative to where the `.tox` was loaded from**.

> **Graceful degradation:** Denoiser = `Rr` (DLSS) → if DLSS is unavailable (non-RTX driver/old driver) or fails a frame, the renderer automatically uses the **OptiX AI denoiser**, and only then the raw accumulated path-trace. Users without DLSS still get a denoised image, not raw noise. Read `rrInit` / `rrResult` on an Info CHOP pointed at the `render` node to see which path is active.

---

## B. Build requirements (to REBUILD from source)

| Component | Version | Where |
|---|---|---|
| CUDA Toolkit | **12.8** | `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8` |
| MSVC toolset | **v142** (VS2019 BuildTools) | project `PlatformToolset` = v142 |
| MSBuild | VS2019 BuildTools | `…\2019\BuildTools\MSBuild\Current\Bin\MSBuild.exe` |
| OptiX SDK | **9.1** (headers only) | `phase2/optix-dev/include` |
| DLSS SDK (NGX) | 310.5.3+ | headers `phase2/DLSS/include`, lib `phase2/DLSS/lib/...` |
| GPU arch (PTX) | `compute_75` (portable; JITs up to all RTX — **not** `sm_89`, which locks to Ada) | `scripts/compile_ptx.ps1` (`-Arch`). DLL is host-only (arch-independent). |

### Build steps (scripts encode the verified commands)
1. **Device → PTX** (only when `demo_programs.cu` / `LaunchParamsDemo.h` change):
   `pwsh scripts/compile_ptx.ps1`  → writes `demo_programs.ptx` and stages it into `Release/`.
2. **Host → DLL** (host-only edits): release the TD lock, then
   `pwsh scripts/build_dll.ps1`  → `phase2/OptixDemoTOP/Release/OptixDemoTOP.dll`.
   Hot-reload over MCP: `unloadplugin=True` → build → `unloadplugin=False` + `reinitpulse.pulse()`.
3. **Stage the bundle:** `pwsh scripts/stage_runtime.ps1` → `dist/PT_Render/` with all 4 runtime files.

See `docs/BUILD-AND-SHIP.md` for the full build/reload/ship detail.

---

## C. Licensing (open-source release — verified 2026-06-29)

The NVIDIA SDK license forbids using the SDK "in any manner that would cause it to become subject to an open source software license," forbids redistributing SDK **headers/libs** standalone, and (for DLSS) requires attribution + pre-release notification. To stay clean as an **MIT open-source** project:

- **Do NOT commit the NVIDIA SDKs** to the public repo: no OptiX headers (`optix-dev/`), no DLSS SDK (`DLSS/`), no `nvsdk_ngx_*.lib`. Builders download them from NVIDIA and set local paths (see §B).
- **DLSS is OPTIONAL and user-supplied.** Do **not** bundle `nvngx_dlssd.dll` in the public release. The renderer's **default denoiser is the OptiX AI denoiser** (ships in the driver), and the graceful `RR→OptiX→raw` fallback means it works fully without DLSS. Users who want DLSS-RR download `nvngx_dlssd.dll` themselves (from NVIDIA/DLSS) and drop it beside the plugin — with NVIDIA's required attribution if they redistribute it.
- **CUDA runtime** (`cudart64_12.dll`) **is** redistributable under the CUDA EULA (Attachment A) — may ship in releases.
- **OptiX** runtime ships inside the NVIDIA display driver; only headers are used at build time (not redistributed).
- The NGX **Project/App ID** is an R&D UUID — fine here; a **registered** ID + NVIDIA notification + in-UI attribution are required only for a **commercial** product, which this is not.

> Not legal advice — verify against the current [DLSS](https://github.com/NVIDIA/DLSS), OptiX, and CUDA EULAs before distributing.
