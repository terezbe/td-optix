# Denoisers

The OptixDemoTOP RTX path tracer ships **three** in-plugin denoise / anti-alias
paths, selected by the **Denoiser** menu parameter on the *Denoise* page
(`OptixDemoTOP.cpp:740-741`, options `None | Optix | Taa | Rr`, default `Optix`):

| Mode      | Menu label                | Mechanism                                                           |
|-----------|---------------------------|--------------------------------------------------------------------|
| `None`    | None (raw)                | Raw running-sum path-trace accumulation, no denoise post-pass.     |
| `Optix`   | OptiX AI                  | OptiX AI denoiser (TEMPORAL model, albedo + normal + flow guides). |
| `Taa`     | TAA (temporal)            | Hand-rolled temporal-reprojection accumulation **inside raygen**.  |
| `Rr`      | DLSS Ray Reconstruction   | NVIDIA DLSS-D via NGX (CUDA), DLAA mode (`RRDenoiser`).             |

Selection is read once per cook before `beginCUDAOperations`
(`OptixDemoTOP.cpp:382`). The three booleans `denoise` / `taa` / `rr` are mutually
exclusive (derived from the single menu string). On `Rr` the plain accumulation
path is bypassed; on `Taa` accumulation happens *inside* raygen; on `Optix`/`None`
the raygen running-sum accumulator runs and OptiX optionally post-filters it.

UI gating: irrelevant sliders are greyed out per mode
(`OptixDemoTOP.cpp:383`) — `Denoisestr` only for OptiX, `Maxhist` only for TAA,
`Flowscale`/`Flowinvy` only for RR, and `Jitter` for everything **except** RR
(RR drives its own deterministic jitter).

> **Note:** SVGF is **not** in this plugin. SVGF lives only in the separate
> research-doc / CudaTOP track and is intentionally out of scope here.

---

## Shared G-buffer / AOV writes (raygen)

All three modes are fed by AOVs the raygen program writes on the **first hit**
of each pixel. The per-ray payload carries the guide data
(`demo_programs.cu:30-34`):

```
hitAlbedo    // surface albedo at the hit   (denoiser guide AOV)
hitNormal    // world normal at the hit     (denoiser guide AOV)
hitPos       // world position at the hit   (used for temporal motion vectors)
hitSpecAlb   // specular albedo / F0         (RR guide)
hitRough     // linear roughness            (RR guide)
```

First-hit AOV accumulators are seeded in raygen at `demo_programs.cu:275-278`
and captured from the primary hit at `demo_programs.cu:312`. The always-written
buffers (used by OptiX and as RR inputs) are emitted near the end of raygen:

- `params.image`   — accumulated color (`demo_programs.cu:397`)
- `params.albedo`  — clamped [0,1] albedo guide (`demo_programs.cu:398`)
- `params.normal`  — world normal guide (`demo_programs.cu:399`)
- `params.flow`    — 2D screen-space motion vector, RGBA with z/w = 0 (`demo_programs.cu:400`)

### Motion vectors (flow)

Flow is computed by reprojecting the current pixel's world hit position through
the **previous** frame's camera (`demo_programs.cu:360-371`). If the surface was
visible last frame (`haveProj`), flow = `current_pixel - previous_pixel` in
pixels; otherwise flow is `(0,0)`. This same flow vector feeds OptiX (`guide.flow`),
the TAA reprojection lookup, and the RR motion buffer.

---

## (A) OptiX AI denoiser

**Creation.** Done once during `ensureReady`/build (`OptixDemoTOP.cpp:327-331`):

```c
OptixDenoiserOptions dopt={};
dopt.guideAlbedo=1; dopt.guideNormal=1;
dopt.denoiseAlpha=OPTIX_DENOISER_ALPHA_MODE_COPY;
optixDenoiserCreate(myOptixContext, OPTIX_DENOISER_MODEL_KIND_TEMPORAL, &dopt, &myDenoiser);
```

- **TEMPORAL** model kind (`OptixDemoTOP.cpp:329`) — reuses the previous denoised
  frame for temporal stability.
- **albedo + normal guides** enabled; **flow guide** supplied per-frame.
- Alpha mode `COPY` — the denoiser does not touch the matte; alpha is passed
  through. (For Transparent background, alpha is additionally restored — see
  *Background interaction* below.)

**Memory.** State + scratch are sized via
`optixDenoiserComputeMemoryResources` on resize and set up with
`optixDenoiserSetup` (`OptixDemoTOP.cpp:477-482`). The denoiser uses
`withoutOverlapScratchSizeInBytes` (no tiling/overlap).

**Per-frame invocation** (`OptixDemoTOP.cpp:687-700`):

1. `optixDenoiserComputeIntensity` computes the HDR intensity into `myIntensity`
   (`:690`); set as `dp.hdrIntensity` (`:691`).
2. `dp.blendFactor = 1.0f - denoisestr` (`:692`) — **OptiX Denoise Strength**
   slider `Denoisestr` (`OptixDemoTOP.cpp:743`, default `0.7`, range 0..1).
   Strength `1` = full AI denoise (`blendFactor 0`), strength `0` = raw
   path-trace (`blendFactor 1`).
3. `dp.temporalModeUsePreviousLayers` is set only once a previous denoised frame
   exists (`myHavePrevDenoised`, `:693`).
4. `dp.flowMulX/Y = flowscale` (`:694`) — the same **RR Motion Scale**
   `Flowscale` slider also scales the OptiX flow guide (and can flip it with a
   negative value).
5. Guide layer = albedo + normal + flow (`FLOAT2` flow image) (`:695`).
6. Layer input/output; `previousOutput` = last denoised frame when available,
   else the current input (`:696-697`).
7. `optixDenoiserInvoke` (`:698`); on success the denoised result is copied into
   `myPrevDenoised` for the next frame (`:699`).

---

## (B) Hand-rolled TAA (in-raygen accumulation)

TAA here is **not** a separate post-pass — it is an accumulation *mode inside the
raygen program* (`demo_programs.cu:373-395`), chosen by `params.taaEnable`
(set from the `Taa` menu, `OptixDemoTOP.cpp:665`).

- **Jitter.** When RR is off, raygen jitters the primary ray by `params.jitter`
  around the pixel center (`demo_programs.cu:284`), scaled by the **AA Jitter**
  slider `Jitter` (`OptixDemoTOP.cpp:745`, default `1.0`). When RR is on, the
  deterministic RR jitter replaces it (`demo_programs.cu:283`).
- **Reprojection.** Uses the same `haveProj`/`ppx,ppy` previous-frame screen
  location as the flow computation (`demo_programs.cu:377-383`). It reads the
  previous TAA color buffer at the reprojected texel.
- **Running average that follows the surface** (`demo_programs.cu:384-386`):
  history count `n` grows up to `params.taaMaxHist`; blend weight `a = 1/n`;
  `outc = hist*(1-a) + col*a`. The new color and history count are stored to
  `taaCurCol`.
- **Max history** = **TAA Max History** slider `Maxhist`
  (`OptixDemoTOP.cpp:744`, default `32`, range 1..512). Higher = smoother but
  more ghosting on disocclusion.

**Double-buffering.** Two color buffers ping-pong each frame via `myTaaParity`
(`OptixDemoTOP.cpp:666-668`): one frame's `taaCurCol` becomes the next frame's
`taaPrevCol`.

When TAA is **off**, raygen instead does plain running-sum accumulation into
`params.accum`, dividing by `frameIndex` (`demo_programs.cu:387-394`) — the
"static hero shot" path that also drives `None` and `Optix` modes. `accum.w`
carries the matte sum.

---

## (C) DLSS Ray Reconstruction (NGX, CUDA) — convention-heavy

RR is implemented in `RRDenoiser.{h,cpp}` and wired in `OptixDemoTOP.cpp`. It is
the most fragile path because nearly every input is a convention NVIDIA's NGX
will silently mis-interpret if you get it wrong. **Every detail below is
load-bearing.**

### Mode & feature-create conventions (`RRDenoiser.cpp:85-95`)

- **DLAA mode** — render resolution **==** output resolution
  (`InWidth==InTargetWidth`, `InHeight==InTargetHeight`, `:93`;
  `InPerfQualityValue = NVSDK_NGX_PerfQuality_Value_DLAA`, `:94`). This is
  denoise + anti-alias with **no upscaling**. Documented in `RRDenoiser.h:6`.
- **Preset_D** — `NVSDK_NGX_Parameter_RayReconstruction_Hint_Render_Preset_DLAA`
  set to `..._Preset_D` (`:86-87`).
- **DLUnified** denoise mode — `InDenoiseMode = ..._Denoise_Mode_DLUnified` (`:90`).
- **Unpacked roughness** — `InRoughnessMode = ..._Roughness_Mode_Unpacked` (`:91`)
  → roughness is its own single-channel `float` AOV, not packed into a normal's
  alpha.
- **Linear depth** — `InUseHWDepth = ..._Depth_Type_Linear` (`:92`). The depth
  AOV is **view-space linear Z**, not hardware/NDC depth (see depth AOV below).
- **Feature flags = `IsHDR | MVLowRes`** (`:95`). `IsHDR` because color is raw
  HDR radiance; `MVLowRes` because motion vectors are at render resolution (= low
  res relative to any conceptual high-res target). In DLAA they are the same
  size, but the flag still selects the MV interpretation NGX expects here
  [unverified: exact NGX semantics of MVLowRes in the equal-resolution DLAA case].

### CUDA context sharing with TouchDesigner (`RRDenoiser.cpp:97-99`)

```c
CUcontext ctx = 0; cuCtxGetCurrent(&ctx);   // share whatever context is current (TD's)
ccp.InCUContext = (void*)ctx; ccp.InCUStream = 0;
```

RR runs on **TD's current CUDA context** — `ensure`/`evaluate` must be called
from `execute()` between `beginCUDAOperations`/`endCUDAOperations`
(`RRDenoiser.h:2-3`). NGX gets whatever context `cuCtxGetCurrent` returns; it
does not create its own. Stream `0` (default) is used.

### The 8 G-buffer inputs

The RR feature is fed **8** input AOVs, enumerated in `RRDenoiser.h:37`
(`COLOR, DALB, SALB, NRM, RGH, DEP, MV, HIT`) and allocated as cudaArrays +
point-sampled texture objects in `ensure` (`RRDenoiser.cpp:69-83`). Formats:

| Idx | Name  | Format   | Source AOV                                  | Written at |
|-----|-------|----------|---------------------------------------------|------------|
| 0 | COLOR | `float4` | `rrColor` raw per-frame HDR radiance        | `demo_programs.cu:404` |
| 1 | DALB  | `float4` | `albedo` diffuse albedo guide (shared)      | `demo_programs.cu:398` |
| 2 | SALB  | `float4` | `rrSpecAlbedo` specular albedo / F0         | `demo_programs.cu:405` |
| 3 | NRM   | `float4` | `normal` world normal (shared)              | `demo_programs.cu:399` |
| 4 | RGH   | `float`  | `rrRoughness` linear roughness              | `demo_programs.cu:406` |
| 5 | DEP   | `float`  | `rrDepth` view-space **linear** Z           | `demo_programs.cu:407-409` |
| 6 | MV    | `float2` | `rrMotion` screen-space motion (pixels)     | `demo_programs.cu:410` |
| 7 | HIT   | `float`  | `rrHitDist` specular hit distance           | `demo_programs.cu:411` |

Each frame `evaluate` copies the linear device buffers into these arrays with
`cudaMemcpy2DToArrayAsync` (`RRDenoiser.cpp:119-126`) and hands the **texture
objects** (not raw pointers) to NGX via the eval params
(`RRDenoiser.cpp:128-138`). The single output is a `float4` surface-backed array
(`myOutSurf`/`myOutArr`, `:81-83`) copied back to `outDenoised`
(`RRDenoiser.cpp:144`).

The host-side RR buffers (`myRRColor` etc.) are allocated to match W×H on resize
(`OptixDemoTOP.cpp:470-475`); `rrHitDist` is `cudaMemset` to zero on (re)alloc
(`:475`). They are wired into LaunchParams at `OptixDemoTOP.cpp:669-670`.

#### Depth AOV (linear)

`rrDepth = aoHit ? vz : 10000.0f` where `vz = dot(hitPos - cam_eye, cam_w)`
normalized by `|cam_w|` — i.e. **view-space linear Z** (`demo_programs.cu:407-409`).
Misses are pushed to a far value (`10000.0`). This pairs with
`Depth_Type_Linear` above; feeding NDC/hardware depth here would break RR.

#### Specular hit distance

`rrHitDist` carries the distance the **reflected/refracted** ray travels from the
first hit (`demo_programs.cu:278, 315, 411`), captured for the specular bounce
(`spec0`). RR uses it to reconstruct sharp reflections.

### Jitter sign convention (`OptixDemoTOP.cpp:680`)

The per-frame jitter is a free-running Halton sequence
(`OptixDemoTOP.cpp:604-607`): `hx = halton(jidx,2)`, `hy = halton(jidx,3)`,
`jidx` = `(myExecuteCount & 15)+1`. The **same** `hx,hy` are sent to raygen as the
primary-ray jitter (`lp.rrJitterX/Y`, `OptixDemoTOP.cpp:670`; consumed at
`demo_programs.cu:283`).

But the jitter handed to **NGX** is negated and re-centered:

```c
float jrrx = -(hx-0.5f), jrry = -(hy-0.5f);   // RR jitter convention: -(jitter-0.5)
```

The `-(halton - 0.5)` sign is the documented NGX convention. Passing `hx,hy`
directly (without the `-(·-0.5)` transform) is a classic RR mistake that causes
sub-pixel smearing / wrong AA.

### Motion-vector scale + Y-flip (`OptixDemoTOP.cpp:681-684`, `RRDenoiser.cpp:135-137`)

`evaluate` receives `mvScaleX, mvScaleY, invertY`, plumbed straight into NGX:

```c
ep.InMVScaleX = mvScaleX;            // RRDenoiser.cpp:135
ep.InMVScaleY = mvScaleY;            // RRDenoiser.cpp:136
ep.InIndicatorInvertYAxis = invertY; // RRDenoiser.cpp:137
```

The host passes **`Flowscale` for both X and Y** (`OptixDemoTOP.cpp:684`):

- **`Flowscale`** = **RR Motion Scale** slider (`OptixDemoTOP.cpp:746`, default
  **`-1.0`**, range -2..2, **not** clamped). Because the motion buffer is in raw
  pixels and NGX's expected sign/scale differs, the default `-1` both flips and
  scales the motion vectors. This is the single most likely knob to need
  re-tuning if reflections/edges swim under camera motion.
- **`Flowinvy`** = **RR Flip MV Y** toggle (`OptixDemoTOP.cpp:747`, default
  **`1`** = on), passed as `invertY` (`OptixDemoTOP.cpp:684`). Y-flip for the
  motion-vector convention (screen Y up vs down). [unverified: `InIndicatorInvertYAxis`
  is, by its name, an NGX *indicator/debug* axis hint; whether it remaps the
  evaluated MVs themselves or only the on-screen MV debug indicator is not
  confirmed from the headers read here — treat the Flowinvy toggle as
  experimentally tuned.]

### View / projection matrices (row-major) (`OptixDemoTOP.cpp:608-620`)

`view16` and `proj16` are built host-side each frame and passed as
`pInWorldToViewMatrix` / `pInViewToClipMatrix` (`RRDenoiser.cpp:132`):

- **`view16`** = world→view, **row-major** (`M*v` convention, comment at
  `OptixDemoTOP.cpp:613`). Rows are the camera basis (`right`, `vup`, `fwd`) with
  translation `-dot(axis, eye)` in the 4th column (`:613-616`).
- **`proj16`** = view→clip, **left-handed, z in [0,1]** (`:617-619`):
  `proj[0]=tx`, `proj[5]=ty`, `proj[10]=zf/(zf-zn)`, `proj[11]=-zn*zf/(zf-zn)`,
  `proj[14]=1` with `zn=0.1`, `zf=1000`. `RRDenoiser.h:24` documents these as
  row-major 4×4.

Getting the handedness/row-major convention wrong here desynchronizes RR's
internal reprojection from the supplied motion vectors.

### Temporal reset (`OptixDemoTOP.cpp:621-622, 683`)

`InReset` (`RRDenoiser.cpp:134`) is driven by `rrReset`. It is forced to `1` the
first frame RR turns on (`if(rr && !myPrevRR) myRRReset=true`,
`OptixDemoTOP.cpp:621`) and on a scene `Reset` (`OptixDemoTOP.cpp:797`),
clearing RR's temporal history on a cut.

### Init & feature lifecycle

- **Init once** (`RRDenoiser.cpp:19-39`): `NVSDK_NGX_CUDA_Init_with_ProjectID`
  with a random R&D project UUID (`kProjId`, `RRDenoiser.cpp:10`) +
  `NVSDK_NGX_CUDA_GetCapabilityParameters`. Triggered lazily on the first cook
  via `myRR.init(...)` (`OptixDemoTOP.cpp:457`), guarded by `myRRInitTried`.
  > **Hardcoded DLL path.** `init` is called with a **hardcoded absolute path**
  > to the `Release` dir that must contain `nvngx_dlssd.dll`
  > (`OptixDemoTOP.cpp:457`). This is a ship-blocker for relocation — see
  > BUILD-AND-SHIP.md. [unverified: whether any relative/auto-discovery fallback
  > exists elsewhere.]
- **`ensure(W,H)`** (`RRDenoiser.cpp:60-107`) is safe to call every frame; it
  recreates the feature + arrays only on a resolution change (`:63`), releasing
  the old feature with `NVSDK_NGX_CUDA_ReleaseFeature` first (`:65`).
- **Status CHOPs.** `rrInit` (RR initialized?) and `rrResult` (last
  `NVSDK_NGX_Result`, `0x1` = Success) are exposed as info CHOP channels
  (`OptixDemoTOP.cpp:722-723`; `lastResult()` doc `RRDenoiser.h:18`).

### RELOAD-SAFETY: shutdown deliberately omits NGX CUDA shutdown

This is the most important operational constraint for RR. `RRDenoiser::shutdown`
(`RRDenoiser.cpp:148-158`) releases the per-feature handle, frees arrays, and
destroys the NGX parameter block — but **intentionally does NOT call
`NVSDK_NGX_CUDA_Shutdown()`** (comment `RRDenoiser.cpp:153-156`):

> `NVSDK_NGX_CUDA_Shutdown()` resets/destroys CUDA driver state on the **shared**
> context, which breaks the *reloaded* plugin's `optixDeviceContextCreate`
> (observed: `ready=0`/`optixOK=0` after an `unloadplugin` reload). We release
> per-feature state only and leave NGX resident for the process lifetime. NGX
> re-init on the next load is idempotent.

In short: because RR shares **TD's** CUDA context (`cuCtxGetCurrent`), a full NGX
shutdown would corrupt that context for everything else in the process —
including a freshly reloaded copy of this very plugin. NGX is left resident on
purpose; this is required for the in-place reload workflow (MCP/`unloadplugin`)
to keep working.

---

## Background interaction (all post-denoise modes)

Denoisers (OptiX and RR) reconstruct **RGB only**. When the background is
**Transparent** (`bgMode==2`) and a denoise result replaced the raw image, the
path-traced matte (`.w`) is restored from `myImage` after denoise
(`OptixDemoTOP.cpp:702-703`). The final chosen `result` buffer is then copied to
the output TOP's cudaArray (`OptixDemoTOP.cpp:704`).

---

## Quick reference — which slider does what

| Slider / toggle | Param      | Mode | Effect |
|-----------------|------------|------|--------|
| OptiX Denoise Strength | `Denoisestr` | Optix | `blendFactor = 1-str`; 1=full denoise, 0=raw (`:692`) |
| TAA Max History | `Maxhist`   | TAA  | Caps temporal history `n` (`demo_programs.cu:381`) |
| AA Jitter       | `Jitter`    | non-RR | Sub-pixel primary-ray jitter amount (`demo_programs.cu:284`) |
| RR Motion Scale | `Flowscale` | RR (& OptiX flow) | NGX `InMVScaleX/Y`; default `-1` flips+scales MVs (`:684`) |
| RR Flip MV Y    | `Flowinvy`  | RR   | NGX `InIndicatorInvertYAxis`; default on (`:684`) |

(Line numbers refer to `OptixDemoTOP.cpp` unless a file is named.)
