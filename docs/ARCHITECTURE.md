# Engine Architecture (end to end)

A real-time RTX path tracer (OptiX 9.1 + CUDA) packaged as a TouchDesigner custom `CPlusPlus TOP` (`OptixDemoTOP`) that re-cooks every frame: it acquires TD TOP inputs as CUDA arrays, builds a triangle GAS from live geometry, fills one shared `LaunchParamsDemo` struct, runs `optixLaunch`, denoises, and writes the result back into the output TOP.

> Scope and honesty note: every statement below is grounded in the source files
> `OptixDemoTOP.cpp`, `OptixDemoTOP.h`, `demo_programs.cu`, `LaunchParamsDemo.h`,
> `RRDenoiser.h/.cpp` (read in full) plus the live network dump
> `_td_state_snapshot.txt`. Claims I could not verify from code are marked
> **[unverified]**. Where the running assumptions in the project notes disagree
> with the actual code, that is called out explicitly (see
> [Discrepancies](#discrepancies-vs-the-running-assumptions)).

---

## 1. The big picture

`OptixDemoTOP` is a `TOP_CPlusPlusBase` subclass registered to TD with
`executeMode = TOP_ExecuteMode::CUDA` and `minInputs = 0`, `maxInputs = 7`
(`FillTOPPluginInfo`, `OptixDemoTOP.cpp:83-94`). In the live project it is the
operator at `/project1/optixDemo/render` (`_td_state_snapshot.txt:2`).

It hosts, entirely in-process:

- one **OptiX device context** (`optixInit` + `optixDeviceContextCreate`, lazily
  on the first cook, `OptixDemoTOP.cpp:451`);
- one **OptiX pipeline** of four program groups — raygen, miss, sphere
  closest-hit, triangle closest-hit (`buildAll`, `OptixDemoTOP.cpp:257-270`);
- a **procedural sphere GAS** generated in code (the "Ray Tracing in One Weekend"
  field of spheres, `genScene` + sphere GAS build, `OptixDemoTOP.cpp:51-78`,
  `307-323`);
- a **live triangle GAS** rebuilt/refit from a TD geometry texture each cook
  (`buildInputTriGAS`, `OptixDemoTOP.cpp:336-365`);
- three **denoiser back-ends** (OptiX AI, TAA reprojection, DLSS Ray
  Reconstruction) selected by a menu parameter.

TD drives it by calling `execute()` once per cooked frame.
`getGeneralInfo` sets `cookEveryFrameIfAsked = true` (`OptixDemoTOP.cpp:230`), so
with the node forced to cook it progressively accumulates samples.

```
                    TouchDesigner process
 ┌─────────────────────────────────────────────────────────────────────┐
 │  TD cook graph                                                        │
 │                                                                       │
 │  POP/SOP ─► poptoTOP ─┐                                               │
 │  POP/SOP ─► poptoTOP ─┤ (geometry attr textures, inputs 0-3)          │
 │  HDRI movie/glsl TOP ─┤ (env, input 4)        ┌──────────────────────┐│
 │  base-color TOP ──────┤ (input 5)             │  OptixDemoTOP        ││
 │  UV poptoTOP ─────────┘ (input 6)        ───► │  ::execute()         ││
 │  Light COMPs ─► Execute DAT ─► "Lightdata" string param ─►           ││
 │  Camera COMP ─► expr ─► Eye/Forward/Camfov params ─►                 ││
 │                                               │   (OptiX + CUDA)     ││
 │                                  output TOP ◄─┤                      ││
 │                                               └──────────────────────┘│
 │   downstream: tonemap (glsl) ─► hsvadj ─► out                         │
 └─────────────────────────────────────────────────────────────────────┘
```

The output is **linear HDR `RGBA32Float`** (`OptixDemoTOP.cpp:423`); tonemapping
is a downstream TD GLSL TOP (`tonemap`, `_td_state_snapshot.txt:39`), not part of
the plugin.

---

## 2. The `execute()` pipeline (ordered)

This is the per-frame cook, in source order (`OptixDemoTOP.cpp:367-712`). The
ordering of "acquire CUDA handles **before** `beginCUDAOperations`" is deliberate
and load-bearing.

| # | Stage | Code | Notes |
|---|-------|------|-------|
| 0 | Output size + read all parameters | `:372-420` | `getSuggestedOutputDesc`, then ~40 `getParDouble/getParString` calls. Params are read **before** any CUDA work. |
| 1 | Create the output CUDA array | `:422-426` | `output->createCUDAArray(...RGBA32Float..., stream=myStream)`. Early-out if null. |
| 2 | Acquire input TOPs as CUDA arrays | `:427-447` | `getInputTOP(i)->getCUDAArray(acq, ...)` for inputs 0-6. **Done before `beginCUDAOperations`** (see §2.1). |
| 3 | `beginCUDAOperations` | `:449` | Enters TD's CUDA context. Early-out if it returns false. |
| 4 | One-time OptiX init + `buildAll` | `:451-452` | `optixInit`, context create, then build pipeline + procedural scene + sphere GAS + denoiser (guarded by `myReadyResult==-999`). |
| 5 | One-time NGX RR init | `:456-457` | `myRR.init(...Release dir)` — on TD's CUDA context (see §6.2). |
| 6 | (Re)allocate size-dependent buffers | `:459-483` | On first cook or resolution change: accum/image/AOVs/flow/TAA ping-pong/RR G-buffer + OptiX denoiser state. Resets `myFrameIndex=0`, `myRRReset=true`. |
| 7 | Copy input geometry → device, build/refit tri GAS | `:485-507` | `cudaMemcpy2DFromArrayAsync` of positions (+Cd/Mat/N/UV) into `float4` buffers, then `buildInputTriGAS(N)`. Sets `triMode`. |
| 8 | Wrap HDRI (input 4) as a CUDA texture object | `:509-521` | Rebuilds the tex object only when the bound `cudaArray` changes. |
| 9 | Wrap base-color map (input 5) as a CUDA texture object | `:523-534` | Same change-detection pattern. |
| 10 | (Re)build HDRI importance CDF | `:536-544` | Throttled to ~every 15 cooks (or on pulse / array swap), host-side. |
| 11 | Parse `Lightdata` string → `PTLight[]` device buffer | `:546-570` | 13 floats per light; no TOP input (see §6.1). |
| 12 | Build camera basis | `:572-589` | Camera-COMP-follow or fallback orbit; thin-lens `cam_u/v/w`. |
| 13 | Accumulation reset logic + frame index | `:591-602` | Reset on move / bg-mode change / sky change / `Reset` pulse. |
| 14 | Jitter (Halton) + view/proj matrices for RR | `:604-622` | Per-frame subpixel jitter; row-major world→view and view→clip. |
| 15 | Fill `LaunchParamsDemo lp` | `:624-670` | Every pointer + scalar the device needs (see §5). |
| 16 | Upload params + `optixLaunch` | `:672-674` | `cudaMemcpyAsync` of `lp` to `myParamsBuf`, then `optixLaunch(...,W,H,1)`. |
| 17 | Denoise (RR **or** OptiX AI **or** none) | `:676-701` | TAA is handled **inside** raygen, so this stage only branches RR/OptiX. |
| 18 | Transparent-matte alpha fix-up | `:702-703` | Denoisers reconstruct RGB only → restore path-traced `.w` for `bgMode==2`. |
| 19 | Copy `result` → output CUDA array | `:704` | `cudaMemcpy2DToArrayAsync` into `outInfo->cudaArray`. |
| 20 | `endCUDAOperations` | `:711` | Leaves TD's CUDA context. |

### 2.1 Why input acquisition happens before `beginCUDAOperations`

All `getCUDAArray` calls for inputs 0-6 are made **before** the
`beginCUDAOperations` gate (`OptixDemoTOP.cpp:427-449`). The HDRI (input 4) is
**gated by sky mode**: `hdriIn` is only fetched when `skyHdri` is true
(`:445`). The base-color input (input 5) is fetched whenever `getNumInputs()>5`
(`:438-440`), independent of geometry or sky mode.

This is an **acquisition gap**, not a connection gap. Two facts from the SDK
matter here:

- `getNumInputs()` returns the **count** of connected inputs, not
  (highest-connected-index + 1) — `CPlusPlus_Common.h` warns the first N are not
  guaranteed connected. But TouchDesigner's UI/Python connection model enforces
  **contiguous** connections (you cannot wire input N while input N-1 is empty;
  verified empirically in this project, which is why the optional inputs had to
  be reordered). Under contiguity, `getNumInputs() == highest-connected-index + 1`,
  so the `getNumInputs()>5` base-color gate (`:438`) is *equivalent* to
  `getInputTOP(5)!=nullptr` and is **not** a live bug. The per-index null-check
  idiom used for inputs 0-3 (`const OP_TOPInput* in = inputs->getInputTOP(i);
  if(in){...}`) is still preferable — it is SDK-correct, clearer, and gap-robust.
- A connected input can therefore **always** be acquired; the freeze surface is
  that `execute()` *chooses not to acquire a connected input*. When sky mode ≠
  HDRI, input 4 (HDRI) is **skipped** (`skyHdri` false, `:445`) while input 5
  (base color) is **still acquired** (`:438`). If input 4 is connected-but-skipped
  and input 5 is connected-and-acquired, the acquired set is `{0,1,2,3,5}` — a gap
  over a *connected* input 4. That **acquisition asymmetry** (acquire 5, skip the
  connected 4) is the leading freeze suspect, observed in incident 6 where
  `getNumInputs()=6`.

Because every `getCUDAArray` runs on TD's **main thread** (`CPlusPlus_Common.h`:
"All CUDA operations must occur on the MAIN thread"), a hang in one of these calls
wedges the very thread that would otherwise write a file log — which is why the
freeze presents as a hard hang with no log tail. This remains a **hypothesis**:
nothing in the plugin *causes* a deadlock by design (`getCUDAArray` is called
identically for every input). See
[Discrepancies](#discrepancies-vs-the-running-assumptions).

```
execute() ordering (critical region)

  read params ──► createCUDAArray(out) ──► getCUDAArray(in0..in6)
                                               │   (input 4/HDRI skipped when
                                               │    sky mode ≠ HDRI, even if
                                               │    connected → acquisition gap;
                                               │    all on TD's MAIN thread)
                                               ▼
                                    beginCUDAOperations()   ◄── CUDA ctx enter
                                               │
        OptiX init ─► alloc ─► copy geom ─► build tri GAS ─► tex objects
                  ─► env CDF ─► parse lights ─► camera ─► fill lp
                  ─► optixLaunch ─► denoise ─► copy to out
                                               │
                                    endCUDAOperations()     ◄── CUDA ctx leave
```

---

## 3. The device side (`demo_programs.cu`)

Compiled by `nvcc` to `demo_programs.ptx` and loaded at runtime via
`PTX_PATH` (`OptixDemoTOP.cpp:19`, `237-247`). One `__constant__ LaunchParamsDemo
params` is the only host→device channel (`demo_programs.cu:7`). Pipeline config:
`numPayloadValues=2` (a packed PRD pointer), `numAttributeValues=1`,
primitive flags `SPHERE | TRIANGLE`, `maxTraceDepth=1` link option — i.e. the
path loop is an **iterative** loop in raygen, not recursive `optixTrace` nesting
(`OptixDemoTOP.cpp:243-269`).

### 3.1 Raygen `__raygen__rg` (`:263-413`)

Per pixel, for `spp` samples, an iterative bounce loop up to `maxDepth`:

1. **Jitter / camera ray.** If `rrEnable`, jitter is the deterministic per-frame
   `rrJitterX/Y` (so RR's temporal AA aligns); otherwise a per-sample random
   jitter scaled by the `jitter` param (`:282-288`). Thin-lens DOF when
   `aperture>0` (`:289-294`).
2. **Trace + shade loop** (`:302-347`): `optixTrace` with `OPTIX_RAY_FLAG_NONE`,
   SBT offset `params.useInput?1:0` (sphere CH = record 0, triangle CH = record
   1). PRD carries throughput, emitted, scatter ray, AOVs.
3. **AOV capture** on the first sample's primary hit: albedo, world normal,
   world pos, specular albedo (F0), roughness, and a specular-reflection
   hit-distance for RR (`:310-318`).
4. **Miss handling** is background-mode aware (`:319-330`): primary miss →
   Environment sky / Solid color / Transparent (alpha 0); specular bounce miss →
   full sky; diffuse bounce miss → sky **unless** `envImportance` owns the env
   (then black, to avoid double-counting NEE).
5. **NEE on diffuse hits**, in order: `sampleSun` (analytic sun), `sampleEnv`
   (HDRI importance), `sampleLights` (analytic Light COMPs), `sampleDirect`
   (emissive spheres) — `:334-341`. After a NEE'd diffuse bounce, `addEmit` is
   cleared so the emitter is not double-counted (`:345`).
6. **Russian roulette** after bounce 3 (`:346`).
7. **Firefly clamp on the indirect term only** — direct light + emitters stay
   bright (`:297`, `:304`, `:349-353`). `Ldir` is snapshotted before the first
   indirect bounce.
8. **Accumulation** (`:373-397`): TAA reprojection running-average if
   `taaEnable`, else a plain progressive running-sum in `params.accum`
   (divided by `frameIndex`). Writes `image`, `albedo`, `normal`, `flow`.
9. **RR G-buffer** written when `rrEnable` (`:402-412`): raw per-frame color,
   specular albedo, roughness, **view-space linear Z**, motion vectors, specular
   hit distance.

### 3.2 Closest-hit — procedural spheres `__closesthit__ch` (`:501-557`)

Hardware sphere primitive (`optixGetSphereData`), per-primitive `DemoMaterial`
indexed by `optixGetPrimitiveIndex()`. Material types: `0` lambert (cosine
hemisphere), `1` metal (reflect + fuzz), `2` dielectric (Fresnel/Schlick
reflect-or-refract), `3` emissive (terminal). Shadow rays short-circuit
(`prd->isShadow → occluded=1`, `:504`).

### 3.3 Closest-hit — input triangle soup `__closesthit__tri` (`:416-492`)

No-index triangle soup: `pid` indexes vertices `3*pid .. 3*pid+2` into
`params.triVerts` (`:423`). Flat geometric normal by default; per-object **smooth
normals** when the material's encoded type is `>= 10` (`smoothObj`), barycentric-
interpolating `triN` (`:433-439`). Per-vertex color from `triCd` (`:442`).
Texturing path (`:443-461`): interpolate `triUV`, decide UV-vs-projection by
`texMode` (`0` auto / `1` force-UV / `2` force-projection), sample `baseColorTex`
either as a `tex2D` UV lookup or via `triplanar(...)` world-space projection, and
modulate `color`. Material type is `rawtype % 10` with the same lambert/metal/
glass/emit branches as spheres.

### 3.4 Miss `__miss__ms` (`:494-499`)

Sets `prd->miss=1` and `prd->skyColor = skyColor(rd)`.

### 3.5 Sky / env model (`:67-111`)

- `skyMode==0`: vertical gradient `skyHorizon→skyZenith` + a sharp analytic
  `sunDisk` (`powf(dot, 1500)*40`).
- `skyMode==1`: equirectangular HDRI lookup `tex2D(hdriTex,u,v)` with
  `u = atan2(z,x)/2π + 0.5 + hdriRot`; the sun disk is suppressed (HDRI carries
  its own sun).
- `skyMode==2`: Preetham analytic daylight (`preethamSky`), xyY→linear-sRGB,
  scaled by `0.05*skyStrength`. Perez coefficients are computed **host-side** from
  sun elevation + turbidity and passed in `lp` (`OptixDemoTOP.cpp:645-663`).

### 3.6 NEE estimators (`:113-247`)

| Estimator | Function | What it samples | pdf handling |
|-----------|----------|-----------------|--------------|
| Emissive spheres | `sampleDirect` `:114-143` | power-weighted CDF over emissive spheres, uniform point on the chosen sphere | area / `pdfLight`, `(1/π)·G` |
| Sun | `sampleSun` `:147-169` | analytic directional delta light, optional angular cone for soft shadows | pdf=1, `f·E·cosθ` |
| HDRI env | `sampleEnv` `:172-199` | 2D-CDF importance sample of the equirect HDRI | solid-angle pdf from `envFunc/envFuncInt` |
| Light COMPs | `sampleLights` `:205-247` | point / cone(spot) / distant analytic lights | pdf=1; point/cone use `1/d²` with a built-in `*50` per-type balance |

All four trace a `TERMINATE_ON_FIRST_HIT` occlusion ray with SBT offset
`params.useInput?1:0`.

---

## 4. The triangle-soup geometry bridge (TD POP → GAS)

The renderer ingests live TD geometry as **attribute textures**, not vertex
buffers. In the live network the upstream nodes are `poptoTOP`s
(`soupTex`, `soupCd`, `soupMat`, `soupN`, `soupUV` —
`_td_state_snapshot.txt:30-35`), which bake POP/SOP point attributes into
`RGBA32Float` textures. Each texel is one vertex; three consecutive texels form
one triangle (triangle soup, `OPTIX_INDICES_FORMAT_NONE`,
`OptixDemoTOP.cpp:348`).

| Input | Live wire | TD attr | Device buffer | `lp` field | Meaning |
|-------|-----------|---------|---------------|------------|---------|
| 0 | `soupTex` | P (position) | `myTriVerts` | `triVerts` | xyz used (stride 16) → GAS vertex buffer |
| 1 | `soupCd` | Cd (color) | `myTriCd` | `triCd` | per-vertex base/emission color |
| 2 | `soupMat` | — | `myTriMat` | `triMat` | (type, roughness, ior, emitStrength) |
| 3 | `soupN` | N (normal) | `myTriN` | `triN` | per-vertex smooth normal |
| 6 | `soupUV` *(exists in network, NOT wired in live snapshot)* | Tex (UV) | `myTriUV` | `triUV` | per-vertex UV (xy) |

In the **live snapshot**, `render`'s inputs are
`['soupTex','soupCd','soupMat','soupN','PTEnv','-']`
(`_td_state_snapshot.txt:27`) — i.e. only inputs **0-5** are present in the
input list, input 5 is empty (`'-'`), and **input 6 (`soupUV`) is not wired**.
The `soupUV` `poptoTOP` *exists in the network* (`_td_state_snapshot.txt:35`)
but is not connected to `render` in this capture, so the UV/texturing path
(input 6 → `myTriUV` → `triUV`) is dormant in the snapshot even though the
plugin fully supports it.

Flow per cook (`OptixDemoTOP.cpp:485-507`):

```
 TD POP/SOP points (P, Cd, Mat, N, UV attrs)
        │   poptoTOP  (one texel per vertex, RGBA32Float)
        ▼
 input 0..3 (+6) cudaArrays   ── getCUDAArray ──►  myTri* float4 device buffers
        │   cudaMemcpy2DFromArrayAsync (DeviceToDevice)
        ▼
 buildInputTriGAS(N)
   ├─ vertexFormat FLOAT3, stride 16 (read xyz of each float4)
   ├─ INDICES_FORMAT_NONE  (soup: 3 verts = 1 triangle)
   ├─ FAST_TRACE | ALLOW_UPDATE
   ├─ REFIT (UPDATE) when same vert count & <30 consecutive refits
   └─ full BUILD on count change or every 30th cook (BVH quality)
        ▼
 myTriGasHandle  ──►  lp.handle (when triMode), SBT record 1 (__closesthit__tri)
```

Capacity is the **padded** texture size `inW*inH` (`:488`), reallocated only when
that changes. The active triangle count `N` is `min(Numverts, padded)` rounded
down to a multiple of 3 (`:340`, `:505`). The UV copy is skipped unless the UV
texture's dimensions exactly match the position texture, to avoid an out-of-
bounds read (`:503-504`).

> **Note on safe vs. unsafe inputs.** The project memory marks `poptoTOP`
> geometry inputs (0-3) as confirmed-safe, and a raw GLSL TOP / Python Script TOP
> wired into a CUDA input as confirmed-deadlock. Nothing in the plugin treats a
> GLSL-backed input differently from a poptoTOP-backed one — both go through the
> same `getCUDAArray` + `cudaMemcpy2DFromArray`. The plugin code does not contain
> a fix for the freeze; the only freeze mitigation visible in code is the
> `Lightdata` **string** transport that replaced a Python-Script-TOP CUDA input
> (§6.1).

---

## 5. The shared data contract

`LaunchParamsDemo.h` is `#include`d by both host and device, so the three structs
below are the literal ABI. Selected fields (full list in the header):

### `LaunchParamsDemo` (the per-launch constant)

| Group | Fields | Purpose |
|-------|--------|---------|
| Output buffers | `accum, image, albedo, normal, flow` | HDR accum, final image, AOV guides, motion |
| Dimensions/progress | `width, height, frameIndex, spp, maxDepth` | launch size + progressive state |
| Geometry | `handle, materials, triVerts, triCd, triMat, triN, triUV, useInput, showUV` | sphere **or** tri GAS + attrs |
| Camera | `cam_eye, cam_u, cam_v, cam_w, aperture` | thin-lens basis (`cam_w` length = focus dist) |
| Temporal | `prev_eye/u/v/w, validPrev, flow, taaPrevCol, taaCurCol, taaEnable, taaMaxHist` | motion vectors + TAA reprojection |
| RR G-buffer | `rrEnable, rrColor, rrSpecAlbedo, rrRoughness, rrDepth, rrMotion, rrHitDist, rrJitterX/Y` | DLSS-RR inputs |
| Base color | `baseColorTex, useBaseColor, projScale, texMode` | UV/triplanar texture map |
| Emissive NEE | `neeEnable, numLights, totalPower, lightPos, lightEmit, lightCDF` | sphere-light direct sampling |
| Analytic lights | `ptLights, numPtLights, lightMaster` | Light COMPs (§6.1) |
| Sky/sun | `sunDir, sunColor, sunAngle, skyStrength, skyZenith, skyHorizon, skyMode, hdriTex, hdriRot` | environment |
| Env importance | `envImportance, envW, envH, envCondCdf, envMargCdf, envFunc, envFuncInt` | HDRI 2D-CDF |
| Physical sky | `perezA..E, perezZenith, perezNorm` | Preetham coefficients |
| Background/AA | `bgMode, bgSolid, jitter, fireflyMax` | matte mode, AA, firefly clamp |

### `DemoMaterial` (per sphere) — `LaunchParamsDemo.h:9-17`
`albedo`, `emission`, `type` (0 lambert / 1 metal / 2 dielectric / 3 emissive),
`fuzz`, `ior`.

### `PTLight` (per analytic Light COMP) — `LaunchParamsDemo.h:20-29`
`pos`, `dir`, `radiance`, `type` (0 point / 1 cone / 2 distant), `radius`
(soft-shadow radius or distant angular radians), `cosInner`, `cosOuter` (cone).

---

## 6. Non-TOP data transports (freeze-driven design choices)

### 6.1 Light COMPs via a string parameter, not a TOP

Analytic lights arrive through the **`Lightdata` string parameter**, parsed
host-side into `myPtLights` (`OptixDemoTOP.cpp:546-570`). Format:
`"N  px py pz dx dy dz cr cg cb type radius cosI cosO  ..."` (13 floats per
light). The header comment is explicit about *why*: "feeding a Python Script TOP
into a CUDA input deadlocks TD's cook thread" (`LaunchParamsDemo.h:101-106`). In
the live network this string is filled by an Execute DAT
(`PTLights/lightbake`, `_td_state_snapshot.txt:96`).

### 6.2 NGX init on TD's CUDA context

`myRR.init(...)` is called once, **inside** `begin/endCUDAOperations`, so NGX
binds to TD's current CUDA context (`OptixDemoTOP.cpp:456-457`; `RRDenoiser.cpp:97`
`cuCtxGetCurrent`). `shutdown()` deliberately does **not** call
`NVSDK_NGX_CUDA_Shutdown()` — doing so resets driver state on the shared context
and breaks a reloaded plugin's `optixDeviceContextCreate`
(`RRDenoiser.cpp:148-158`).

---

## 7. The three denoiser options

Selected by the `Denoiser` menu: `None`, `Optix`, `Taa`, `Rr`
(`OptixDemoTOP.cpp:740-742`, `382`). Note that **TAA is not a post-pass** — it is
the accumulation mode inside raygen — so the post-launch branch only chooses
RR vs OptiX vs nothing.

### 7.1 OptiX AI denoiser (`Optix`)

Created in `buildAll` as `OPTIX_DENOISER_MODEL_KIND_TEMPORAL` with albedo +
normal guides (`:327-330`). Per frame (`:687-700`): compute HDR intensity, invoke
with guide layers (albedo, normal, **flow**), `blendFactor = 1 - Denoisestr`,
and the previous denoised frame as temporal history. The denoised result becomes
the output; the previous-output buffer is updated for the next frame.

### 7.2 DLSS Ray Reconstruction (`Rr`) — `RRDenoiser` / NGX

A pure-CUDA NGX wrapper running in **DLAA mode** (output res == render res:
denoise + AA, no upscaling — `RRDenoiser.h:6`). `init` →
`NVSDK_NGX_CUDA_Init_with_ProjectID`; `ensure(W,H)` creates the DLSSD feature
(`Preset_D`, `DLUnified`, unpacked roughness, **linear** depth, HDR,
`MVLowRes`) and per-input `cudaArray`+texture objects (`RRDenoiser.cpp:60-107`).
`evaluate(...)` copies the 8 linear G-buffer inputs into arrays, fills
`NVSDK_NGX_CUDA_DLSSD_Eval_Params` (color, depth, MVs, diffuse/specular albedo,
normals, roughness, specular hit distance, world→view + view→clip matrices,
jitter, reset, MV scale/invert), runs `NGX_CUDA_EVALUATE_DLSSD_EXT`, and copies
the output surface back (`RRDenoiser.cpp:109-146`). The host passes RR jitter as
`-(halton-0.5)` and MV scale from the `Flowscale` param (`OptixDemoTOP.cpp:680-684`).

```
 raygen RR G-buffer (linear device buffers, render res)
   rrColor, albedo, rrSpecAlb, normal, rrRough, rrDepth(viewZ), rrMotion, rrHitDist
        │  RRDenoiser::evaluate  (copy → cudaArray → texObj)
        ▼
   NGX DLSSD (DLAA, Preset D)  +  world→view / view→clip + jitter + reset
        ▼
   myOutSurf ──► myDenoised ──► output
```

### 7.3 GLSL SVGF chain "in the TD network" — **[unverified / not present in the live dump]**

The brief lists a third option as "a GLSL SVGF chain that lives in the TD
network." **No SVGF nodes appear in the live network dump** — the only
downstream GLSL is `tonemap` (and `hsvadj1`, `out1`)
(`_td_state_snapshot.txt:15,24,39`), and the `Denoiser` menu has only
`None/Optix/Taa/Rr` (`OptixDemoTOP.cpp:741`). The plugin **does export the AOVs an
SVGF pass would need** — `flow` (motion), `albedo`, `normal` are all written by
raygen and are available downstream — and a separate phase doc
(`PHASE3-optix-x-svgf.md`) exists, so an external SVGF chain is plausibly a
*planned/separate-network* artifact rather than a shipped, wired part of this
operator. Treat "SVGF is wired in" as unverified until the network shows it.

---

## 8. Camera basis, motion vectors, TAA reprojection

### 8.1 Camera basis (`OptixDemoTOP.cpp:572-589`)

Two modes:

- **Camera COMP follow** (`Usecamera`): `Eye`/`Forward`/`Camfov` are driven by
  expressions bound to a TD Camera COMP. TD's **horizontal** FOV is converted to
  vertical: `fovY = 2·atan(tan(fovH/2)/aspect)`.
- **Fallback orbit**: eye orbits a target at `(0,1,0)` by `myOrbitAngle`.

The thin-lens basis is `right = norm(fwd×up)`, `vup = right×fwd`, then
`cam_u = right·halfW`, `cam_v = vup·halfH`, `cam_w = fwd·focus` (so `|cam_w|` is
the focus distance, `LaunchParamsDemo.h:46-49`).

### 8.2 Motion vectors via previous-camera reprojection (`demo_programs.cu:360-371`)

There is **no per-object motion** — motion is camera reprojection only. The
previous-frame camera basis (`prev_eye/u/v/w`, captured each cook before update,
`OptixDemoTOP.cpp:591-602`) is used to project the current hit's world position
`aoPos` into the previous frame's screen space; `flow = (curPix - prevPix)`. This
single flow buffer feeds the OptiX denoiser (`guide.flow`), TAA, and RR
(`rrMotion`).

### 8.3 TAA reprojection (`demo_programs.cu:373-395`)

When `taaEnable`, raygen reprojects into a ping-pong history buffer
(`taaPrevCol`/`taaCurCol`, swapped each cook by `myTaaParity`,
`OptixDemoTOP.cpp:666-668`). It reads the history at the reprojected pixel, blends
`a = 1/n` where `n` is the capped history count (`taaMaxHist`), and writes the
new running average + count. When TAA is off, raygen instead does plain
progressive accumulation into `params.accum` divided by `frameIndex`.

### 8.4 RR matrices + jitter (`OptixDemoTOP.cpp:604-622`, `:680-684`)

`view16` is a row-major world→view (rows = right/vup/fwd, translation =
`-dot(axis,eye)`); `proj16` is a left-handed view→clip with z∈[0,1], `zn=0.1`,
`zf=1000`. Per-frame Halton(2,3) jitter feeds raygen (deterministic when
`rrEnable`) and is passed to NGX as `-(halton-0.5)`.

---

## 9. Build & runtime facts

- PTX is loaded from an **absolute path** baked into the source
  (`OptixDemoTOP.cpp:19`); the NGX feature DLL dir is likewise absolute
  (`OptixDemoTOP.cpp:457`). Moving the project requires editing these.
- The DLL ships as `Release/OptixDemoTOP.dll` with `Release/nvngx_dlssd.dll`
  beside it (the NGX search path).
- Info CHOP channels expose health: `executeCount, optixOK, ready, frameIndex,
  spheres, rrInit, rrResult, lights, envReady, ptLights`
  (`OptixDemoTOP.cpp:714-727`) — useful first-line diagnostics.
- Target hardware per project notes: RTX 4080 Super (Ada, sm_89), OptiX 9.1,
  CUDA 12.8, TD 2025.32820. **[unverified from this source set]** — versions are
  from the project brief, not from a file I read here.

---

## Discrepancies vs. the running assumptions

1. **Input index map.** The code comments and parameter labels call the HDRI
   "input 5" in some places (`Skymode` menu label *"HDRI (equirect, input 5)"*,
   `OptixDemoTOP.cpp:778`) and the base-color "the 6th input"
   (`:789`), but the **actual `getInputTOP` indices are 0-based**: positions=0,
   Cd=1, Mat=2, N=3, **HDRI=4**, **base-color=5**, **UV=6**
   (`OptixDemoTOP.cpp:428-447`). The CONTEXT brief's "input 5 = base-color,
   input 6 = UV" matches the **code indices**; the in-UI labels are off-by-one
   and should not be trusted. This doc uses the code indices throughout.

2. **The acquisition-gap freeze hypothesis is consistent with the code,
   but not proven by it.** The freeze is *not* a connection gap (impossible under
   TD's contiguous-input model) — it is an **acquisition gap**: `execute()`
   acquires `{0,1,2,3,5}` while **skipping the connected input 4** when sky mode ≠
   HDRI and a base-color input is wired (`:438-447`). `getNumInputs()` is a count,
   not a max-index, but under contiguity `getNumInputs()>5` is equivalent to
   `getInputTOP(5)!=nullptr`, so the gate itself is not a live bug; the asymmetry
   is acquiring 5 while skipping the connected 4. Every `getCUDAArray` runs on
   TD's **main thread**, so a hang there wedges the same thread that would write a
   log. This matches the "leading new hypothesis." However, nothing in the plugin
   *causes* a deadlock by design — `getCUDAArray` is called identically for every
   input — so the doc records this as a hypothesis, not a verified root cause.
   The only freeze fix actually present in code is the `Lightdata` string
   transport (§6.1).

3. **SVGF is not in the live network.** The brief lists a GLSL SVGF chain as a
   third denoiser "that lives in the TD network," but the dump shows no SVGF
   nodes and the menu offers only `None/Optix/Taa/Rr`. The plugin exports the
   AOVs SVGF would consume, so this is plausibly planned/separate, but it is
   **not** a wired, shipped part of `/project1/optixDemo/render` as captured.

4. **"Procedural sphere scene, no external geometry" header comments are stale.**
   Both `OptixDemoTOP.h:1-2` and `demo_programs.cu:1` still describe the engine as
   procedural-only, but the code clearly supports live input triangle geometry,
   textures, Light COMPs, and HDRI environments. The comments predate the
   geometry-bridge work.

5. **Live `PTEnv` is in Physical-sky mode** (`Mode = 'Physical'`,
   `_td_state_snapshot.txt:63`) with input 5 EMPTY (`:53`). That is exactly the
   configuration the base-color freeze forensics describe (sky not HDRI → input 4
   skipped), so the snapshot is a useful reproduction starting point.
