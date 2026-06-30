# TouchDesigner Parameter Reference

Complete custom-parameter reference for the `OptixDemoTOP` CPlusPlus TOP (`/project1/optixDemo/render`), grounded in the `setupParameters()` definitions and the `execute()` consumers in `OptixDemoTOP.cpp`.

---

## How to read this doc

Every parameter below is defined in `OptixDemoTOP::setupParameters()` (lines 730-794 of `phase2/OptixDemoTOP/OptixDemoTOP.cpp`) and — unless explicitly noted — is read back inside `OptixDemoTOP::execute()` (lines 376-420). The page order matches the order the pages are emitted in `setupParameters()`:

1. Render
2. Denoise
3. Camera
4. Input
5. Environment
6. Lights
7. Texture

There is no separate "Debug" page in the code; the two debug-style controls (`Showuv`, and `Reset`) live on the Input and Render pages respectively. The page order shown in TouchDesigner follows the order pages first appear in `setupParameters()`.

**Type/style** column uses the TD `OP_ParameterManager` append call:
- `appendFloat` = Float slider
- `appendInt` = Integer field
- `appendToggle` = Toggle (On/Off)
- `appendPulse` = Pulse button
- `appendMenu` = Menu (dropdown)
- `appendString` = String field
- `appendXYZ` = 3-component XYZ (vector)
- `appendRGB` = 3-component RGB (color)

**Default** values are the literals from `setupParameters()`. Where a snapshot live value differs from the code default, it is called out in the "Discrepancies vs live snapshot" section at the bottom (the live values in `_td_state_snapshot.txt` belong to the **child COMPs** `PTEnv`/`PTLights`, not to the render TOP itself — see that section).

---

## Page 1 — Render

| Parameter | Label | Type/style | Default | What it does |
|-----------|-------|-----------|---------|--------------|
| `Spp` | Samples / Pixel | Float (1-64, slider 1-16, min-clamped) | 4 | Path-tracer samples per pixel per cook. Read at line 376, rounded to int, floored to >=1, passed as `lp.spp`. |
| `Maxdepth` | Max Bounces | Float (1-32, slider 1-16, both clamped) | 8 | Maximum ray bounce depth. Read at line 377, floored to >=1, passed as `lp.maxDepth`. |
| `Firefly` | Firefly Clamp (0=off) | Float (0-2000, slider 0-50, min-clamped) | 10 | Clamps per-sample radiance to suppress fireflies; `0` disables. Read at line 390, negative values forced to 0, passed as `lp.fireflyMax`. |
| `Nee` | Direct Light Sampling (NEE) | Toggle | 0 (Off) | Enables Next-Event Estimation (direct light sampling). Read at line 392. **Only takes effect for the built-in sphere/emitter scene** — `lp.neeEnable` is gated `nee && myNumLights>0 && !triMode` (line 634), so NEE is force-disabled whenever live input geometry is active. |
| `Reset` | Reset Accumulation | Pulse | (button) | Restarts the progressive accumulation buffer. Handled in `pulsePressed()` (line 797): sets `myResetReq`, zeros `myFrameIndex`, and flags `myRRReset` so the DLSS-RR history also resets. |

Notes:
- Accumulation also resets automatically when the camera moves, the background mode changes, or the sky mode / HDRI rotation changes (lines 595-600).
- `Nee` defaults to Off, which matches the memory note that NEE "backfired (off)".

---

## Page 2 — Denoise

| Parameter | Label | Type/style | Default | What it does |
|-----------|-------|-----------|---------|--------------|
| `Denoiser` | Denoiser | Menu: `None` / `Optix` / `Taa` / `Rr` | `Optix` | Selects the post-pass. Labels: "None (raw)", "OptiX AI", "TAA (temporal)", "DLSS Ray Reconstruction". Read at line 382; sets the `denoise` / `taa` / `rr` booleans. |
| `Denoisestr` | OptiX Denoise Strength | Float (0-1, clamped) | 0.7 | OptiX AI denoiser blend. `1` = full AI denoise, `0` = raw path-trace (`dp.blendFactor = 1 - strength`, line 692). Greyed out unless `Denoiser=Optix` (`enablePar`, line 383). |
| `Maxhist` | TAA Max History | Float (1-512, slider 1-128, clamped) | 32 | Max temporal history length for the TAA mode (`lp.taaMaxHist`, line 665). Floored to >=1. Greyed out unless `Denoiser=Taa`. |
| `Jitter` | AA Jitter | Float (0-1, clamped) | 1 | Sub-pixel anti-alias jitter amount (`lp.jitter`, line 664). Greyed out **when** `Denoiser=Rr` (RR supplies its own jitter), enabled otherwise (`enablePar("Jitter", !rr)`, line 383). |
| `Flowscale` | RR Motion Scale | Float (-2 to 2, unclamped) | -1 | Motion-vector multiplier fed to DLSS Ray Reconstruction (`flowscale`, line 385; used at line 684). Default `-1` flips MV sign. Greyed out unless `Denoiser=Rr`. Also reused as the OptiX flow multiplier (`dp.flowMulX/Y`, line 694) when OptiX denoise is active. |
| `Flowinvy` | RR Flip MV Y | Toggle | 1 (On) | Flips the motion-vector Y axis for RR (`flowinvy`, line 391; passed to `myRR.evaluate`, line 684). Greyed out unless `Denoiser=Rr`. |

Notes:
- The menu is a single selector; only the slider(s) relevant to the chosen mode stay enabled (lines 383).
- RR path: `myRR.evaluate(...)` (lines 681-684). OptiX path: `optixDenoiserInvoke(...)` (line 698). TAA is handled inside the OptiX launch via the `taa*` launch-param fields.

---

## Page 3 — Camera

| Parameter | Label | Type/style | Default | What it does |
|-----------|-------|-----------|---------|--------------|
| `Usecamera` | Use Camera COMP | Toggle | 0 (Off) | When On, the renderer uses the externally-driven `Eye`/`Forward`/`Camfov` values instead of the built-in orbit camera (line 395, branch at 575). |
| `Camera` | Camera COMP | String | `pcam` | Path of the TD Camera COMP to follow. **Not read by the C++ plugin** — there is no `getParString("Camera")` in `execute()`. It is a binding hint for a TD-side script/Execute DAT that writes `Eye`/`Forward`/`Camfov`. [unverified: the exact TD-side mechanism that consumes it — no such DAT was inspected for this doc.] |
| `Orbit` | Orbit (fallback cam) | Toggle | 0 (Off) | Animates the built-in orbit camera. When On, advances `myOrbitAngle` each cook (line 572). |
| `Orbitspeed` | Orbit Speed | Float (-10 to 10, slider -5 to 5) | 1 | Orbit angular speed (`myOrbitAngle += orbitSpeed*0.01`, line 572). |
| `Distance` | Orbit Distance / Focus | Float (4-40, slider 6-25, clamped) | 13 | Orbit radius for the fallback camera; also the focus distance when `Usecamera` is On (line 579/581). Values <=0.5 fall back to a hardcoded distance. |
| `Aperture` | DOF Aperture | Float (0-0.6, slider 0-0.3, clamped) | 0 | Depth-of-field aperture (`lp.aperture`, line 632). `0` = pinhole (no DOF). |
| `Eye` | Cam Eye (driven) | XYZ (-1000 to 1000) | (0,0,0) | World-space eye position, used only when `Usecamera` is On (lines 396, 576). Intended to be driven by an expression/binding from a Camera COMP. |
| `Forward` | Cam Forward (driven) | XYZ (-1 to 1) | (0,0,-1) | World-space forward direction, used only when `Usecamera` is On (lines 397, 577). Normalized before use. |
| `Camfov` | Cam FOV (driven) | Float (1-170, slider 10-120, clamped) | 45 | Horizontal field of view in degrees; converted to vertical FOV internally (lines 398, 579/584). Applies in both `Usecamera` and orbit modes. |

Notes:
- "(driven)" parameters (`Eye`, `Forward`, `Camfov`) are designed to be set by TD bindings/expressions referencing the `Camera` COMP, not typed by hand.
- When `Usecamera` is Off, the renderer uses a hardcoded look-at-target orbit (target `(0,1,0)`, height `2.2`), still respecting `Camfov` and `Distance` (lines 581-584).

---

## Page 4 — Input

| Parameter | Label | Type/style | Default | What it does |
|-----------|-------|-----------|---------|--------------|
| `Useinput` | Use Input Geometry | Toggle | 0 (Off) | Master switch for live triangle-soup input. When On, the renderer acquires CUDA arrays from inputs 0-3 (and optional 6) and (re)builds the triangle GAS this cook (lines 393, 428-443, 485-507). |
| `Numverts` | Input Vert Count | Int (0-5,000,000, slider 0-300,000, min-clamped) | 0 | Number of valid vertices in the soup texture; `0` means "use the full padded texture". Bounds the GAS build count `N` (line 505). |
| `Showuv` | Show UVs (debug) | Toggle | 0 (Off) | Debug visualization that outputs UV coordinates as color (`lp.showUV`, line 628). |

Input wiring (from `execute()`, lines 428-447, gating in code):

| TD input index | Purpose | Acquisition gate |
|----------------|---------|------------------|
| 0 | Triangle-soup positions (poptoTOP) | `useInput && numInputs>0` |
| 1 | `Cd` color (poptoTOP) | `useInput && numInputs>1` |
| 2 | `Mat` (type, roughness, ior, emit) (poptoTOP) | `useInput && numInputs>2` |
| 3 | `N` smooth normals (poptoTOP) | `useInput && numInputs>3` |
| 4 | Equirectangular HDRI environment | `skyHdri && numInputs>4` (i.e. **only** when Sky Mode = HDRI) |
| 5 | Base-color map (Texture page) | `numInputs>5` — **NOT** gated on `useInput` or sky mode |
| 6 | UV (Tex), optional | `useInput && numInputs>6` |

> **Freeze-relevant finding (grounded in lines 438-447):** Input 5 (base color) is acquired with `getCUDAArray` whenever it is connected (`numInputs>5`), independent of sky mode. Input 4 (HDRI) is acquired **only** in HDRI sky mode (`skyHdri`, line 445). So in Gradient or Physical sky mode with both input 4 (HDRI/PTEnv) and input 5 (base color) connected, the plugin calls `getCUDAArray` on inputs {0,1,2,3} and {5}, **skipping the *connected* index 4** while still acquiring the higher index 5. This is an **acquisition gap, not a connection gap**: it is the leading freeze suspect because the skipped input is connected yet never acquired, leaving an asymmetric acquire set ({0,1,2,3,5}) on TD's main thread (where all CUDA ops must run).
>
> Note on counting: `getNumInputs()` returns the **count** of connected inputs, not (highest-connected-index + 1) — but TouchDesigner's UI/Python connection model enforces **contiguous** input connections (you cannot connect input N while input N-1 is empty). Under that contiguity, `getNumInputs() == highest-connected-index + 1`, so `numInputs>5` here is equivalent to `getInputTOP(5)!=nullptr` and is **not** a live bug; a per-index `getInputTOP(i)!=nullptr` check would still be clearer and gap-robust. The freeze is therefore not a non-contiguous *connection* (impossible in TD) but the acquisition asymmetry described above. This is a **hypothesis, not proven cause** — see `crash-investigation/` and `FREEZE-INVESTIGATION.md` for the incident history and the decisive control experiment.

---

## Page 5 — Environment

| Parameter | Label | Type/style | Default | What it does |
|-----------|-------|-----------|---------|--------------|
| `Skyzenith` | Sky Zenith | RGB (0-1) | (0.45, 0.62, 1.0) | Zenith color of the procedural gradient sky (`lp.skyZenith`, line 641). Used in Gradient sky mode. |
| `Skyhorizon` | Sky Horizon | RGB (0-1) | (1.0, 1.0, 1.0) | Horizon color of the procedural gradient sky (`lp.skyHorizon`, line 641). |
| `Skystrength` | Sky Strength | Float (0-10, slider 0-3, min-clamped) | 1 | Overall gradient-sky brightness multiplier (`lp.skyStrength`, line 641). |
| `Sundir` | Sun Direction | XYZ (-1 to 1) | (0.4, 0.85, 0.3) | Analytic sun direction; normalized (`lp.sunDir`, line 637). Also drives the Preetham physical-sky elevation (line 648). |
| `Suncolor` | Sun Color | RGB (0-1) | (1.0, 0.95, 0.85) | Sun color, multiplied by `Sunstrength` into `lp.sunColor` (line 638). |
| `Sunstrength` | Sun Strength | Float (0-200, slider 0-50, min-clamped) | 8 | Sun radiance multiplier (line 638). `0` disables the analytic sun. |
| `Sunangle` | Sun Angle (soft shadow) | Float (0-20, slider 0-5, min-clamped) | 0.5 | Soft-shadow cone half-angle in degrees; converted to radians (`lp.sunAngle = sunang*0.01745…`, line 639). |
| `Backgroundmode` | Background | Menu: `Environment` / `Solid` / `Transparent` | `Environment` | Background fill mode. Mapped to `bgMode` 0/1/2 (line 407) -> `lp.bgMode` (line 640). Transparent mode preserves the path-traced alpha matte after denoising (line 702). |
| `Bgcolor` | Background Color | RGB (0-1) | (0,0,0) | Solid background color, used when `Backgroundmode=Solid` (`lp.bgSolid`, line 640). |
| `Skymode` | Sky Mode | Menu: `Gradient` / `Hdri` / `Physical` | `Gradient` | Selects sky model. `Gradient` = procedural sky+sun; `Hdri` = equirect HDRI on **input 5** per the menu label (= index 4 in 0-based `getInputTOP` terms; the menu label is 1-based — the code wires HDRI to `getInputTOP(4)`, line 445 — see discrepancies, consistent with the Page-4 wiring table); `Physical` = Preetham analytic sky. Read at line 409 -> `lp.skyMode` (line 642). |
| `Hdrirot` | HDRI Rotation | Float (-360 to 360, slider -180 to 180) | 0 | HDRI yaw rotation in degrees, converted to turns (`lp.hdriRot = hdrirot/360`, line 642). Changing it resets accumulation (line 598). |
| `Envimportance` | HDRI Importance Sampling | Toggle | 1 (On) | Enables building/using the HDRI importance-sampling CDF (`wantImp = hdriOn && envImportanceOn`, line 537; `lp.envImportance`, line 643). The CDF rebuild is throttled to ~every 15 cooks (line 540). |
| `Rebuildenv` | Rebuild Env Map | Pulse | (button) | Forces an immediate HDRI importance-map rebuild. Handled in `pulsePressed()` (line 798): sets `myEnvRebuildReq`. |
| `Turbidity` | Turbidity (Physical) | Float (1.7-10, slider 1.7-8, clamped) | 2.5 | Atmospheric turbidity for the Preetham physical sky (used only when `Skymode=Physical`; lines 414, 647-662). |

Notes:
- HDRI features (`Hdrirot`, `Envimportance`, `Rebuildenv`) only have effect when `Skymode=Hdri` and a valid HDRI texture is bound (`hdriOn`, line 521).
- The full Preetham Perez-coefficient computation lives at lines 646-663 and is only consumed when `lp.skyMode==2`.

---

## Page 6 — Lights

| Parameter | Label | Type/style | Default | What it does |
|-----------|-------|-----------|---------|--------------|
| `Lightintensity` | Light Intensity (master) | Float (0-1000, slider 0-10, min-clamped) | 2 | Master multiplier over all parsed analytic Light COMPs (`lp.lightMaster`, line 636). |
| `Lightdata` | Light Data (auto-filled) | String | `""` | Serialized Light COMP data, **auto-filled by a TD-side Execute DAT** (comment line 415). Format: a leading count `N` followed by 13 floats per light: `px py pz dx dy dz cr cg cb type radius cosInner cosOuter` (parsed lines 549-569). Uploaded to `myPtLights` and exposed as `lp.ptLights` (line 636). |

Notes:
- This page is the resolution of the earlier GIL-deadlock: light data travels through a **string parameter** rather than a CUDA TOP input, per the project memory ("switching that data to a string-parameter Lightdata transport").
- Up to 4096 lights are parsed (guard at line 552); each light record must contain all 13 floats or parsing stops early (lines 555-557).

---

## Page 7 — Texture

Base-color map = the 6th input (TD input index **5**); optional UV = the 7th input (TD input index **6**). Comment at line 789.

| Parameter | Label | Type/style | Default | What it does |
|-----------|-------|-----------|---------|--------------|
| `Texmode` | Texture Coords | Menu: `Auto` / `Uv` / `Projection` | `Auto` | Texture-coordinate source. Labels: "Auto (UV if present, else projection)", "Force UV", "Force Projection (triplanar)". Mapped to `texmode` 0/1/2 (line 420) -> `lp.texMode` (line 629). |
| `Projscale` | Projection Scale (units/tile) | Float (0.01-100, slider 0.1-8, min-clamped) | 1.5 | World-units per texture tile for triplanar/projection mapping (`lp.projScale`, line 629). |

Notes:
- The base-color texture is only **applied** when live input geometry is active AND a base-color texture object exists: `lp.useBaseColor = (triMode && myBaseColorTex)?1:0` (line 629). The base-color CUDA array is still *acquired* whenever input 5 is connected (line 438) — acquisition and application are gated differently, which is central to the freeze investigation.

---

## Info CHOP channels (read-only outputs)

Not parameters, but useful for monitoring. From `getInfoCHOPChan()` (lines 715-727), the render TOP exposes 10 Info CHOP channels:

| Index | Channel | Meaning |
|-------|---------|---------|
| 0 | `executeCount` | Total cook count (`myExecuteCount`). |
| 1 | `optixOK` | 1 if `optixInit()` succeeded. |
| 2 | `ready` | 1 if the OptiX pipeline/GAS built successfully. |
| 3 | `frameIndex` | Current accumulation frame index. |
| 4 | `spheres` | Built-in sphere count. |
| 5 | `rrInit` | 1 if DLSS Ray Reconstruction initialized. |
| 6 | `rrResult` | Last RR result code. |
| 7 | `lights` | Built-in emitter light count (NEE). |
| 8 | `envReady` | 1 if the HDRI importance map is built. |
| 9 | `ptLights` | Number of parsed analytic Light COMPs (from `Lightdata`). |

---

## Discrepancies vs live snapshot

These are differences between the running assumptions / the snapshot file and what the code actually says. They are intentionally called out for accuracy.

1. **The snapshot does not contain the render TOP's own custom parameters.** `_td_state_snapshot.txt` section "RENDER CUSTOM PARAMETERS (by page)" lists parameters of the **child COMPs** `PTEnv` (a `base` COMP) and `PTLights` (a `base` COMP) — e.g. `Skyhdr`, `Mode='Physical'`, `Skyzenithr`, `Sundirx`, `Hdrifile`, `Root`, `Render`. **None of those are parameters of the render CPlusPlus TOP.** The render TOP's actual custom parameters are the ones in this doc, defined in `setupParameters()`. The snapshot's `PTEnv`/`PTLights` parameters are the *authoring UI* that a TD-side script presumably translates into the render TOP's `Skymode`, `Sundir`, `Lightdata`, etc. — but that translation layer was not in the files read for this doc. [unverified: the exact mapping from `PTEnv`/`PTLights` parameters to the render TOP parameters.]

2. **`PTEnv.Mode = 'Physical'` in the snapshot.** The render TOP's own `Skymode` defaults to `Gradient` in code. The live system was running the *PTEnv* COMP in Physical mode, which (per the freeze hypothesis) is exactly the configuration where input 4 (HDRI) is skipped while input 5 (base color) is still acquired. This is consistent with the leading deadlock hypothesis.

3. **Sky Mode menu label says HDRI is "input 5"; the code wires HDRI to input 4.** `setupParameters()` line 778 labels the HDRI option `"HDRI (equirect, input 5)"`, and the Texture-page comment (line 789) also calls the base color "the 6th input." Using TD's 0-based input indices in `execute()`: HDRI is acquired from `getInputTOP(4)` (line 445) and base color from `getInputTOP(5)` (line 438). So the menu label is using 1-based human counting ("input 5" = the 5th input = index 4), while the code uses 0-based indices. The labels are not wrong, just a different counting convention — worth flagging because it is easy to misread as a bug.

4. **`Nee` / NEE is effectively disabled for input geometry.** Although `Nee` exists and defaults Off, even turning it On has no effect while live geometry is active, because `lp.neeEnable` requires `!triMode` (line 634). Consistent with the memory note that NEE "backfired (off)".

5. **`Camera` string parameter is never read in C++.** It is defined (line 751, default `pcam`) but there is no consumer in `execute()`. It only matters to whatever TD-side logic binds `Eye`/`Forward`/`Camfov`. [unverified: that TD-side binding was not inspected.]

---

## Source files read for this reference

- `phase2/OptixDemoTOP/OptixDemoTOP.cpp` — `setupParameters()` (lines 730-794), `pulsePressed()` (lines 796-799), `execute()` parameter reads (lines 376-420) and consumers (lines 422-712), `getInfoCHOPChan()` (lines 714-727).
- `phase2/crash-investigation/_td_state_snapshot.txt` — node graph, input wiring, and `PTEnv`/`PTLights` child-COMP parameter dump.

Status: the renderer is **in progress** (feature build); this reference documents the parameters as they exist in the read sources. Anything not directly observable in those files is marked `[unverified]`.
