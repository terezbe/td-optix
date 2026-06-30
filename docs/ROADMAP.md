# Roadmap & Checkpoint (done / in-progress / backlog)

A living checklist for **OptixDemoTOP** — a standalone OptiX 9.1 + CUDA 12.8 path tracer hosted in a TouchDesigner CPlusPlus TOP (`/project1/optixDemo/render`), whose goal is to feel like a native TD "Render TOP": assemble a scene from familiar TD pieces (Camera/Light COMPs, materials, environment, textures). This is a documentation checkpoint for an eventual ship — the project is still in active development.

> **How to read status tags:** ✅ = verified working in code I read. ⚠️ = works but with a caveat. 🔬 = investigated then shelved. 🚧 = in progress / blocked. ⬜ = backlog / not started. `[unverified]` = a claim I could not confirm directly against the engine source.

---

## At-a-glance

| Area | Status |
|---|---|
| HDR float output + resolution + Nvidia Denoise TOP | ✅ Done |
| Smooth normals (per-object) | ✅ Done |
| Native C++ POP geometry route | 🔬 Investigated, **shelved** (froze TD ×3) |
| G-buffer AOVs + motion vectors + in-plugin denoisers (OptiX AI / TAA / DLSS RR) | ✅ Done |
| Triangle-soup live geometry (no index buffer) + POP texture bridge | ✅ Done |
| Seamless live play + per-object materials via baked attributes | ✅ Done |
| Environment: gradient / HDRI / importance-sampled env NEE / physical sky / PT Environment COMP | ✅ Done |
| Light COMPs → NEE (point/cone/distant) via param transport | ✅ Done |
| **Base color map (UV + triplanar)** | 🚧 **In progress — blocked by "the freeze"** |
| Shipping path fix (DLL-relative paths + self-contained bundle) | ✅ Done (2026-06-29) |
| **FBX import** — FBX COMP → SOP-to-POP → triangle-soup bridge (matID flatten); design in `docs/RESEARCH-fbx-import.md` | ⬜ **Next version (v2)** — planned |
| PT Lights UX rethink, camera DOF, more texture maps, consolidation | ⬜ Backlog |

A note on the SVGF design paper (`PHASE3-optix-x-svgf.md`): that GLSL-TOP SVGF denoiser (temporal / variance / à-trous) and the `e2DArray` multi-AOV output were **designed** there but are **not present in the shipping OptixDemoTOP source** — see [Discrepancies](#discrepancies-context-vs-code). What actually ships in OptixDemoTOP is a single `e2D` RGBA32Float output plus three in-plugin denoisers (OptiX AI, TAA, DLSS Ray Reconstruction).

---

## DONE (chronological — what each step delivered)

### Step 1a/1b — HDR float output + resolution param
- The TOP outputs **linear HDR**, not tonemapped 8-bit. `execute()` creates the output as `OP_PixelFormat::RGBA32Float`, `texDim = e2D` (`OptixDemoTOP.cpp:422-423`). Tonemapping/gamma happen **downstream** in TD (a `tonemap` pixel shader), so the denoisers receive linear HDR.
- Output resolution is driven by the TOP's resolution (`W`/`H` read each cook; all per-frame device buffers reallocate on a size change — `OptixDemoTOP.cpp:459-483`).

### Step 1 — Nvidia Denoise TOP wiring
- The plugin emits the AOVs a denoiser wants alongside noisy color: a first-hit **albedo** buffer and first-hit **world-normal** buffer (`LaunchParamsDemo.h:36-37`, written by the device program). These can feed TD's native **Nvidia Denoise TOP** downstream.
- The same AOVs are also consumed by the **in-plugin** OptiX AI denoiser (see Phase 3 below), so denoising can happen either inside the plugin or downstream.

### Step 1c — smooth normals
- Per-object **smooth normals**: the triangle closest-hit barycentric-interpolates a per-vertex normal (`triN`, `LaunchParamsDemo.h:83`) when an N attribute is wired (input 3), falling back to the flat geometric normal otherwise.
- Made **per-material** via a toggle: the PT Material `Smoothnormals` flag is encoded as `type += 10` in the Mat attribute; the shader decodes `smoothObj = (rawtype >= 10)` and gates the interpolation per object. `[unverified]` exact encoding line — confirmed in the seamless-pop plan notes, consistent with the `type` decode in `demo_programs.cu`.

### Step 3 — native C++ POP geometry route → 🔬 INVESTIGATED, then SHELVED
- The plan was GPU-direct dynamic **topology**: a CPlusPlus **POP** reads the index buffer + positions as CUDA device buffers, shared to the OptiX TOP. The hand-off worked in isolation but **TD froze 3×**.
- Root-cause research (`step3-research-notes.md`, `PHASE3-optix-x-svgf.md §1`): partly the Windows console **Quick-Edit** freeze (a known CudaPOP gotcha), partly likely **CUDA-context contention** (POP `getBuffer(CUDA)` vs the OptiX context every frame).
- **Decision: shelve the C++ POP route**; pivot to the stable **triangle-soup via texture** bridge (below). The C++ POP code is preserved but not in the active path.

### Phase 3 — OptiX G-buffer / AOVs + motion vectors + in-plugin denoisers
- The raygen/closest-hit programs emit a per-pixel **G-buffer**: albedo, world normal, linear depth, **motion vectors** (`params.flow`, written at `demo_programs.cu:400`), plus the RR-specific layers (`rrDepth`, `rrMotion`, spec albedo, roughness, hit distance — `demo_programs.cu:409-410`, `LaunchParamsDemo.h:65-74`).
- Motion vectors come from the previous-frame camera basis (`prev_eye/u/v/w`, `LaunchParamsDemo.h:52-56`); members `myPrevEye/U/V/W` + `myHavePrev` (`OptixDemoTOP.h:112-113`).
- **Three in-plugin denoisers**, selected by one **Denoiser menu** (None / OptiX / TAA / RR — `OptixDemoTOP.cpp:382`, `setupParameters` menu at ~`:742`):
  - **OptiX AI denoiser** (temporal, with albedo+normal+flow guide layers) — `OptixDemoTOP.cpp:687-700`.
  - **TAA** (temporal reprojection accumulation through camera motion) — `taaPrevCol`/`taaCurCol`, `LaunchParamsDemo.h:60-63`.
  - **DLSS Ray Reconstruction** (NGX CUDA, via `RRDenoiser`) — `OptixDemoTOP.cpp:681-685`, wrapper in `RRDenoiser.h/.cpp`.
  - Denoiser-specific params gray out via `inputs->enablePar(name,bool)` in `execute()` (C++-side, since Python `.enable` can't touch plugin params). `[unverified]` exact enablePar lines — described in the seamless-pop plan; the menu-driven read is confirmed in code.
- **Output AOV mechanism actually shipped:** a single `e2D` RGBA32Float output with separate device buffers per AOV fed to the denoisers — **not** the `e2DArray` layered output proposed in the SVGF paper.

### Phase 3 — SVGF GLSL denoiser → ⚠️ DESIGNED, NOT in OptixDemoTOP
- `PHASE3-optix-x-svgf.md` fully specs a GLSL-TOP **SVGF** pipeline (temporal reprojection → variance prefilter → 5-level à-trous, albedo demodulation, separate direct/indirect). I found **no SVGF / à-trous / variance code in OptixDemoTOP** (`demo_programs.cu`, `OptixDemoTOP.cpp`). The shipping denoisers are OptiX AI / TAA / DLSS RR (above). Treat SVGF as a design doc / future track, not a delivered feature of this plugin. The SVGF work referenced in CONTEXT most likely lived on the **CudaTOP** track (see the consolidation note in [Backlog](#backlog--pending)).

### Phase 4 R1 — triangle-soup renderer (variable-size, no index buffer)
- Live geometry rides in as a **positions texture** (one texel per vertex, RGBA32F). Each cook the plugin reads input 0, `cudaMemcpy2DFromArrayAsync` into a `float4` vert buffer, and builds a **no-index triangle GAS** (`buildInputTriGAS(N)`, `OptixDemoTOP.h:144`; copy at `OptixDemoTOP.cpp:498-506`).
- Uses **FAST_TRACE BVH + REFIT**: refit when the vert count is unchanged (`myTriGasRefits`), full rebuild on count change, forced full rebuild every 30 refits to keep BVH quality (`OptixDemoTOP.h:74-76`). `[unverified]` exact build-flag lines (inside `buildInputTriGAS`, not read here) — flag intent confirmed by the member comments.
- Pipeline has **two hitgroups** (procedural sphere CH = SBT record 0, `__closesthit__tri` = record 1); raygen traces `useInput ? record 1 : record 0` with the active GAS.

### Phase 4 R2 — POP → triangle-soup texture bridge
- TD-side adapter chain (from the seamless-pop plan):
  ```
  POP scene → Facet POP (operation=unique)   # un-shares an indexed mesh into a triangle SOUP
            → poptoTOP (attribscope=P, layout=square, format=rgba32float, rgbamode=custom)
            → render input 0
  ```
- **Gotcha (cost a debug round):** `poptoTOP.par.rgbamode` defaults to `pactive`, which bakes **P regardless of `attribscope`** → must set `rgbamode='custom'` for non-P attributes (Color/Mat/N). P bakes correctly under either mode.
- **Gotcha:** the renderer reads input as TRIANGLES (every 3 soup verts = 1 tri); quads/n-gons (e.g. Box POP) mis-render → triangulate before the merge.

### Phase 4 R3 — seamless live play (deform / move / add, zero rebuilds)
- The GAS rebuilds **every cook** from whatever's in the input texture (realloc only when vert count changes), so add/remove/deform "just works" with no DLL reload and no manual rebuild. Pure deformation (same count) uses REFIT. This is what makes it feel like wiring a SOP into a native Render TOP.
- `Useinput` toggle + `Numverts` int param select input mode and the real (un-padded) vertex count (`OptixDemoTOP.cpp:393-394`, params at `:761-762`).

### Phase 4 R4 — live per-object materials via texture
- Per-vertex material attributes baked alongside positions and read in the triangle CH:
  - **Color** (input 1 → `triCd`) = base/emission color.
  - **Mat** (input 2 → `triMat`) = `vec4(type, roughness, ior, emitStrength)`; type 0=lambert, 1=metal, 2=glass, 3=emitter (`LaunchParamsDemo.h:81-82`, decode at `demo_programs.cu:442,462-473`).
  - **N** (input 3 → `triN`) = smooth normal.
- Defaults when an attribute isn't wired: grey lambert (`color = V(0.82,0.80,0.78)` at `demo_programs.cu:442`).
- TD-side **PT Material COMP** (`/project1/optixDemo/PTMat`) wraps an Attribute POP that bakes Color + Mat from custom params (Type / Color / Roughness / Ior / Emitstrength / Smoothnormals). Drop one per object; copy per material.

### Environment 1c — background-visibility modes
- `Backgroundmode` menu: **Environment | Solid | Transparent** + `Bgcolor` RGB (`OptixDemoTOP.cpp:406-408`, params `:773-776`; device `bgMode`/`bgSolid`, `LaunchParamsDemo.h:130-131`).
- Branch is on **primary-ray miss only**; secondary-bounce misses always use the real sky so lighting is identical across modes. Transparent mode plumbs a real **alpha matte** (path-traced `.w`), and since the denoisers reconstruct RGB only, the host does a strided copy to restore the matte before the blit (`OptixDemoTOP.cpp:702-703`). Switching modes resets accumulation (`myPrevBgMode`, `OptixDemoTOP.h:115`).

### Environment 2 — HDRI input
- `maxInputs` includes an **equirectangular HDRI** on **input 4**, wrapped as a `cudaTextureObject` (cached in `myHdriTex`/`myHdriArray`, recreated only when the array ptr changes — `OptixDemoTOP.cpp:509-520`). `skyDome` HDRI branch maps direction → equirect UV; `Skymode` menu (Gradient / Hdri / Physical) + `Hdrirot` (`OptixDemoTOP.cpp:409-412`).

### Environment 3 — HDRI importance-sampled env NEE (sinθ CDF)
- Host builds a **2D CDF** (luminance × sinθ) from the HDRI (`buildEnvCDF`, `OptixDemoTOP.h:145`; members `myEnvCondCdf`/`myEnvMargCdf`/`myEnvFunc`/`myEnvFuncInt`, `OptixDemoTOP.h:128-133`), throttled to ~every 15 cooks or a `Rebuildenv` pulse (`OptixDemoTOP.cpp:536-544`).
- Device `sampleEnv` binary-searches the CDFs → dir + pdf, casts an occlusion ray, returns `f·L·cos/pdf` (`demo_programs.cu:172`, called at `:337`). Gated by `Envimportance` (default ON, HDRI-only).
- **Note on "MIS":** rather than full multiple-importance-sampling weights, the code reuses a `lastSpec` trick — diffuse hits do env-NEE and their bounce-miss adds 0 (env owned by NEE); specular bounces still see the full env. So it's importance-sampled NEE with a double-count guard, not textbook MIS weighting. `[unverified]` whether a formal MIS weight is applied anywhere.

### Environment 4 — physical sky
- Analytic daylight sky driven by `Turbidity` (`OptixDemoTOP.cpp:414`), `skyMode==2`. **The implementation in `demo_programs.cu` is Preetham** (`preethamSky`, `demo_programs.cu:78`, comment "analytic daylight sky (no solar disk)"), with Perez A–E coefficients + zenith + norm uploaded per cook (`LaunchParamsDemo.h:126-129`). The analytic sun NEE + sun disk are layered on top.
- **Discrepancy:** CONTEXT/memory says "Hosek-Wilkie/Preetham" — the shipping code is **Preetham only** (no Hosek-Wilkie radiance term found).

### Environment 5 — PT Environment COMP
- `/project1/optixDemo/PTEnv` base COMP with ~24 custom params (Mode, Sky Zenith/Horizon/Strength, Sun Dir/Color/Strength/Angle, Turbidity, Background, Bg Color, HDRI Rotation, Importance, Hdrifile, HDRI source) grouped under section headers (Sky / Sun / HDRI / Environment Map / Background). The renderer's Environment-page params are expression-bound to PTEnv, so editing PTEnv drives the renderer (like the Camera COMP). Real CC0 HDRI pre-loaded (`phase2/hdri/venice_sunset_2k.hdr`); HDRI source can be a File or any TOP. `[unverified]` exact PTEnv param list (a TD-side COMP, not in the C++ source) — described in the seamless-pop plan; the renderer-side Environment params it binds to **are** confirmed in `setupParameters` (`OptixDemoTOP.cpp:766-783`).
  - ⚠️ One PTEnv polish (the Sky-Mode "fold" that merges the File/TOP source sub-menu into the Mode menu) was applied but **not yet saved** — it reverts on TD restart and needs re-applying.

### Light COMPs integration (point / cone / distant → NEE via param transport)
- **Final design = parameter transport, not a TOP input.** The renderer has a `Lightdata` **string** param (CSV: count, then 13 floats/light) + a `Lightintensity` master float on a "Lights" page (`OptixDemoTOP.cpp:415-416`, params `:786-787`). The C++ `getParString` parses it → host `PTLight[]` → device buffer (`OptixDemoTOP.cpp:549-570`), and device `sampleLights` does NEE with soft shadows (`demo_programs.cu:205`, called at `:339`).
- TD side: a **PTLights** base COMP holds native Light COMPs + an **Execute DAT** (`onFrameStart`) that serializes each light (type, color×dimmer, world pos/dir, cone, size) into `render.par.Lightdata`.
- **Why param transport:** feeding a Python **Script TOP** into a CUDA input **hard-freezes** TD (GIL / cook-thread re-entrancy — see [the freeze](#the-freeze)). The texture-bridge attempt for lights froze TD ×3; this design has none of those inputs. `PTLight` struct: `LaunchParamsDemo.h:20-29`.

---

## IN PROGRESS

### 🚧 Base color map (UV + triplanar projection, auto-fallback) — BLOCKED by "the freeze"
The feature is **fully coded** on both host and device, but cannot be exercised safely because the input it relies on triggers a deadlock.

- **Device side (works):** `LaunchParamsDemo.h:87-91` (`baseColorTex`, `useBaseColor`, `projScale`, `texMode`); shading at `demo_programs.cu:443-460`:
  - `texMode`: 0 = auto (UV if the geometry has non-degenerate UVs, else projection), 1 = force UV, 2 = force projection (`demo_programs.cu:453`).
  - UV path samples `tex2D(baseColorTex, uu, vv)`; the **triplanar** fallback (`demo_programs.cu:250-260`) blends 3 axis projections by the normal, tiling every `projScale` world units.
  - The map **modulates** the base color (`color = mul(color, tx)`), it doesn't replace it.
- **Host side (works):** input 5 acquired and wrapped as a wrap/linear/normalized `cudaTextureObject` (`OptixDemoTOP.cpp:438-440, 523-534`); params `Texmode` (menu) + `Projscale` on a "Texture" page (`OptixDemoTOP.cpp:418-420`, params `:792-793`); `Showuv` debug toggle.

#### The freeze
Feeding certain TD TOPs into the renderer's CUDA inputs **hard-hangs TD's main thread** (white window, "not responding"), with **no GPU TDR** in the Windows event log → a pure **CPU/main-thread deadlock**, not a GPU timeout.

- **Confirmed safe:** poptoTOPs (geometry attribute textures, inputs 0–3); the HDRI Movie-File-In on input 4.
- **Confirmed deadlock:** (a) a Python **Script TOP** wired to a CUDA input (GIL / cook re-entrancy — already solved for lights by moving to the `Lightdata` string param); (b) a raw **GLSL TOP** at input 5 (base-color map); (c) that **same GLSL TOP routed through a Null TOP** — so the "an Out/Null TOP fixes GL→CUDA interop" theory is **disproven**.

#### Leading hypothesis — an *acquisition* gap, not a connection gap (grounded in the code I read)
First, a correction to an earlier framing: this is **not** a non-contiguous *connection* problem. TouchDesigner's UI/Python connection model enforces **contiguous** input connections — you cannot wire input N while input N-1 is empty (verified empirically in this project: optional inputs had to be reordered for exactly this reason). And per the TD SDK header (`CPlusPlus_Common.h`, `getNumInputs` docs ~line 2255-2260), `getNumInputs()` returns the **count** of connected inputs; under TD's contiguity that count equals (highest-connected-index + 1). So the `getNumInputs()>5` gate at `OptixDemoTOP.cpp:438` is equivalent to `getInputTOP(5)!=nullptr` and is **not** a live connection bug. (The per-index `getInputTOP(i)!=nullptr` check is still preferred — clearer, SDK-correct, and gap-robust — but the existing gate is functionally fine.)

The real asymmetry is in **which connected inputs `execute()` actually acquires** (`OptixDemoTOP.cpp:428-447`):

| Input | `getCUDAArray` called when | Code |
|---|---|---|
| 0 pos | `useInput && numInputs>0` | `:428,434` |
| 1 Cd | `useInput && numInputs>1` | `:429,435` |
| 2 Mat | `useInput && numInputs>2` | `:430,436` |
| 3 N | `useInput && numInputs>3` | `:431,437` |
| **5 base color** | **`numInputs>5`** (NOT gated by sky mode) | **`:438,440`** |
| 6 UV | `useInput && numInputs>6` | `:441,443` |
| **4 HDRI** | **`skyHdri && numInputs>4`** (skipped when sky mode ≠ HDRI) | **`:445,447`** |

The smoking gun: whenever the renderer is **not** in HDRI sky mode (`skyHdri` false — e.g. Physical-sky mode), `execute()` **skips `getCUDAArray` on the connected input 4 (HDRI)** at `:445` while **still calling `getCUDAArray` on input 5 (base color)** at `:438`. In the real freeze (incident 6) input 4 (PTEnv Out) **was connected** and input 5 (a test TOP) **was connected** ⇒ `getNumInputs()=6` ⇒ input 5 acquired, input 4 connected-but-**not**-acquired. The acquired set was `{0,1,2,3,5}` — a CONNECTED input (4) deliberately skipped while a higher index (5) is acquired. That **acquisition asymmetry** is the leading suspect, versus the one demonstrably-working texture input (the HDRI), which is acquired only when it's the active sky.

**Why this wedges silently:** `getCUDAArray` runs on TD's **MAIN thread** (`CPlusPlus_Common.h` ~line 676: "All CUDA operations must occur on the MAIN thread"). A hang inside that call wedges the same thread that would otherwise write a file log — which is why the freeze produces a white "not responding" window with **no GPU TDR** and often no durable log line.

**Caveat / honesty:** this is a strong, code-grounded **hypothesis**, **not yet proven** to be the deadlock mechanism. A competing data point is that a Python Script TOP froze even as the *only* extra input, and a raw GLSL TOP froze on input 5 even when routed through a Null TOP — so the GL→CUDA interop angle (raw generator sources needing GL→CUDA interop on the main thread) is not excluded. `[unverified]` causation.

#### Decisive experiment (minimal freeze risk, maximal discrimination)
The primary diagnostic must be **out-of-process** because the hang is on the same thread that writes a file log:

1. **Instrument `execute()`.** Emit `OutputDebugString` immediately **before** and **after** each `getCUDAArray` call, labeled with the input index and per-index connectivity computed via `getInputTOP(i)!=nullptr` (NOT `getNumInputs()>k`). Add a flush-per-line file log as a durable secondary. Read the breadcrumbs live in **DebugView**. Note: an input's `cudaArray` pointer is null until `beginCUDAOperations()`, so the AFTER-acquire log must **not** treat a null array pointer as a failure. Rebuild once.
2. **Run the SAFE control (CUDA-resident source).** Keep input 4 connected (the existing PTEnv Out) + set **Physical** sky (so input 4 is skipped) + wire a **known-safe poptoTOP** (CUDA-resident, non-GL) into input 5. This yields `getNumInputs()=6`, input 4 connected-but-skipped, input 5 acquired — the exact acquisition asymmetry, but with a CUDA-resident source.
   - **If TD does NOT freeze** ⇒ the acquisition-gap hypothesis is **refuted**; the cause is the GL-backed generator source (raw GLSL/Null TOP needs GL→CUDA interop on the main thread). Fix path: route base color through a CUDA-resident TOP, or load textures host-side from file.
   - **If TD DOES freeze** ⇒ the acquisition asymmetry itself is the cause, independent of GL interop; the DebugView trace's last line names the exact `getCUDAArray` that wedged. Fix path: always acquire connected inputs (don't skip input 4), or restructure acquisition.

Either outcome is decisive.

---

## BACKLOG / PENDING

- **⬜ Rethink PT Lights UX.** The param-transport light system works but reads opaque: native Light COMPs are nested *inside* PTLights (invisible at top level) and data rides a hidden auto-filled `Lightdata` string. Want a clearer surface (lights visible at top level / readable status / less magic). Also expose the per-type intensity balance as a visible knob rather than the built-in factor.
- **⬜ Camera DOF.** The TD Camera COMP has **no aperture/focus** (TD's raster renderer can't do DOF), so the renderer's `Aperture` + focus currently sit on the render TOP = split-brain. Fix per the "param locality" principle: add `Aperture`/`Focus` to the Camera COMP (or a **PT Camera** wrapper) and read them there. (DOF itself is plumbed: `aperture` + thin-lens basis exist, `LaunchParamsDemo.h:45-50`; `Aperture` param at `OptixDemoTOP.cpp:755`.)
- **⬜ Shipping fix — relative PTX/DLL paths.** Two **hardcoded absolute paths** block shipping: the PTX (`OptixDemoTOP.cpp:19` → `.../demo_programs.ptx`) and the RR/NGX DLL dir (`OptixDemoTOP.cpp:457`). Make these relative to the plugin (e.g. resolve next to the DLL).
- **⬜ Other texture maps (roughness / metallic / normal / emission).** Designed to **reuse the base-color plumbing** (same input-acquisition + `cudaTextureObject` + UV/triplanar sampling). Blocked behind the same freeze; unblocking base color unblocks these.
- **⬜ Broader seamless POP/SOP renderer plan.** Instancing via OptiX IAS (per-point transforms), SDF/volume textures, depth/displacement map as geometry, more attribute channels — the full "native Render TOP" vision in `seamless-pop-renderer-plan.md`.
- **⬜ CudaTOP-vs-OptixDemoTOP consolidation.** The project has two diverged renderers: **OptixDemoTOP** (has DLSS RR, cleaner code, the seamless triangle-soup input — the chosen base) and **CudaTOP** (`/project1/PT/cudatest`, has the older SVGF / glass / Geo-COMP scene work, no RR). **Decision recorded: consolidate into OptixDemoTOP**, with CudaTOP as donor/reference. This consolidation is **not yet executed** in code (two plugins still exist).
- **⬜ Min-system-requirements doc.** Capture the verified target: RTX 40-series (Ada, sm_89), driver R590+ (tested 595.97), OptiX 9.1, CUDA 12.8, TD 2025.32820, Windows. `[unverified]` lower bounds (only the dev machine — RTX 4080 Super — is confirmed).

---

## Discrepancies (CONTEXT vs. code)

These are the gaps I found between the running assumptions in CONTEXT/memory and what the engine source actually says — worth flagging before they propagate into ship docs:

1. **SVGF is not in OptixDemoTOP.** `PHASE3-optix-x-svgf.md` describes a GLSL-TOP SVGF denoiser (temporal/variance/à-trous) as a build target. No SVGF/à-trous/variance code exists in `demo_programs.cu` or `OptixDemoTOP.cpp`. The shipping in-plugin denoisers are **OptiX AI, TAA, and DLSS Ray Reconstruction**. SVGF was a design paper (and likely the CudaTOP track), not a delivered OptixDemoTOP feature.
2. **Output is `e2D`, not `e2DArray`.** The Phase 3 paper proposed a layered `e2DArray` G-buffer output; the shipping code uses a single `e2D` RGBA32Float output (`OptixDemoTOP.cpp:422-423`) with separate device AOV buffers fed to the denoisers.
3. **Physical sky is Preetham, not Hosek-Wilkie.** CONTEXT says "Hosek-Wilkie/Preetham"; the implementation is **Preetham only** (`preethamSky`, `demo_programs.cu:78`). No Hosek-Wilkie radiance term found.
4. **Orbit camera now respects the FOV param.** The seamless-pop memory lists a TODO: "in Orbit mode the render uses a HARDCODED fovY=28° and ignores Camfov." The current code computes `fovY` from `camfov` in the orbit branch too (`OptixDemoTOP.cpp:584`, comment "orbit now respects the FOV param too"). That TODO appears **resolved**.
5. **Base color (input 5) is acquired even when the HDRI (input 4) is skipped.** Inputs 0–3 and 6 are gated by `useInput`; input 4 by `skyHdri` (skipped when sky mode ≠ HDRI); input 5 by neither — `getCUDAArray` runs on input 5 (`OptixDemoTOP.cpp:438`) while it is skipped on the *connected* input 4 (`:445`). Note this is an **acquisition** gap, not a connection gap (TD enforces contiguous connections, so `getNumInputs()>5` is equivalent to `getInputTOP(5)!=nullptr` and is not itself a bug). This acquisition asymmetry is the mechanism behind the freeze hypothesis and is the single most actionable code finding here.
6. **"MIS" is NEE + a `lastSpec` double-count guard, not formal MIS weighting.** Env importance sampling is real (sinθ CDF), but I found no balance-heuristic MIS weight — the double-count avoidance is the `lastSpec` trick. Wording in ship docs should say "importance-sampled NEE" rather than "MIS" unless a weight is added.

---

*Checkpoint generated from a read of the live engine source (`OptixDemoTOP.cpp`, `OptixDemoTOP.h`, `LaunchParamsDemo.h`, `demo_programs.cu`) plus the planning docs (`seamless-pop-renderer-plan.md`, `dev-plan.md`, `PHASE3-optix-x-svgf.md`, `step3-research-notes.md`). TD-side COMP/parameter details that live outside the C++ source are marked `[unverified]` where I could not confirm them against code.*
