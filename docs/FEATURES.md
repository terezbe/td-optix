# Feature Inventory & Status

A code-grounded snapshot of every feature in the **OptixDemoTOP** real-time RTX path tracer — a standalone OptiX 9.1 + CUDA 12.8 `CPlusPlus TOP` plugin hosted at `/project1/optixDemo/render` (RTX 4080 Super, sm_89, Win11). Each entry carries an honest status and a description traced to the actual source. The project is **in progress**; this is a documentation checkpoint, not a ship claim.

---

## How to read this doc

**Status legend**

| Status | Meaning |
|---|---|
| **working** | Implemented, exercised, no known correctness blocker. |
| **works-but-fragile** | Implemented and functional, but has a known sharp edge (convention tuning, a deadlock path, scene-dependent behavior). |
| **in-progress** | Code present and wired, but blocked, unproven in the live scene, or known to misbehave. |
| **todo** | Referenced in plans/research only; **not** in this codebase. |

**Exposure** notes whether a feature is a **TD parameter** (user-facing custom par) or **internal** (host/device logic with no direct knob).

**Sources read for this doc:** `OptixDemoTOP.cpp` (host cook, full file), `demo_programs.cu` (device programs, full file), `LaunchParamsDemo.h` (shared structs), `RRDenoiser.h`, the live `_td_state_snapshot.txt`, and the `dlss-rr-build-state` memory. SVGF absence was verified by grep over the engine source dir (no hits).

---

## 1. Path tracer core

The renderer is a progressive Monte-Carlo path tracer. Raygen (`__raygen__rg`, `demo_programs.cu:263`) shoots `spp` primary rays/pixel, bounces up to `maxDepth`, and accumulates into an HDR buffer. The default scene is **procedural**: a field of a few hundred hardware spheres (up to ~490, data-dependent on the RNG skips in `genScene`) generated host-side (`OptixDemoTOP.cpp:51`).

| Feature | Status | Exposure | Notes (grounded in code) |
|---|---|---|---|
| Lambert (diffuse) material | working | internal/material | `type==0`: cosine-hemisphere scatter, throughput = albedo (cosine pdf cancels 1/π). Sphere CH `demo_programs.cu:550`, triangle CH `:487`. |
| Metal material | working | internal/material | `type==1`: mirror reflect with optional `fuzz` roughness lobe. Sphere `:532`, triangle `:473`. |
| Glass / dielectric | working | internal/material | `type==2`: Schlick-Fresnel reflect/refract, IOR ~1.5. Sphere `:540`, triangle `:479`. `refract3`/`schlick` at `:58`/`:64`. |
| Emitter material | working | internal/material | `type==3`: terminal hit, deposits `emission` (sphere) or `Cd*emitStrength` (triangle). Sphere `:528`, triangle `:469`. |
| Samples / pixel (spp) | working | **TD par** `Spp` | Default 4, clamp min 1, max 64. Per-sample loop `demo_programs.cu:280`. Read `OptixDemoTOP.cpp:376`. |
| Max bounces | working | **TD par** `Maxdepth` | Default 8, range 1–32. Bounce loop `demo_programs.cu:302`. Read `:377`. |
| Russian-roulette path termination | working | internal | After bounce ≥3, terminate by max-throughput probability (`demo_programs.cu:346`). Not a knob. |
| Firefly clamp | working | **TD par** `Firefly` | Default 10, 0=off. Clamps **indirect** luminance only (direct/emitter highlights preserved) `demo_programs.cu:350`. Direct snapshot at bounce 1 `:304`. |
| Progressive accumulation | working | internal (+`Reset` pulse) | Running-sum buffer `accum`, divided by `frameIndex` (`demo_programs.cu:389`). Auto-resets on camera move / bg or sky change (`OptixDemoTOP.cpp:600`); `Reset` pulse forces it (`:797`). |
| Thin-lens depth of field | working | **TD par** `Aperture` | Lens-disk jitter, focus plane fixed (`demo_programs.cu:289`). `cam_w` length = focus distance. |
| Pinhole/orbit & driven camera | working | **TD pars** | Orbit fallback or "Use Camera COMP" (Eye/Forward/Camfov driven by expr). `OptixDemoTOP.cpp:575`. TD horizontal FOV → vertical conversion `:579`. |
| Anti-alias subpixel jitter | working | **TD par** `Jitter` | 0=stable/aliased, 1=full AA. Random per-sample unless RR is on (then deterministic Halton) `demo_programs.cu:283`. |

**Tonemapping note:** the TOP outputs **linear HDR** RGBA32F (`image` buffer); tonemapping is done downstream in TD (a `tonemap` GLSL TOP in the snapshot), not in the plugin.

---

## 2. G-buffer / AOVs

First-hit auxiliary outputs, captured once per pixel at `sp==0, b==0` (`demo_programs.cu:310`). Albedo and normal feed the OptiX denoiser; the specular/roughness/depth/motion/hit-distance set feeds DLSS-RR.

| AOV | Status | Notes |
|---|---|---|
| Albedo | working | First-hit surface albedo, clamped [0,1] (`demo_programs.cu:398`). OptiX guide layer. |
| World normal | working | First-hit world normal (`:399`). OptiX guide layer. |
| Linear depth | working | View-space linear Z = `dot(P−eye, forward)` (`demo_programs.cu:407`), 10000 on miss. RR expects linear depth. |
| Motion vectors (flow) | working | Screen-space reprojection through the previous-frame camera (`demo_programs.cu:360`). Written to `flow` (float2) and `rrMotion`. See §8. |
| Specular albedo (F0) | working | Per-material: metal→albedo, glass/diffuse→0.04, emitter→0 (`demo_programs.cu:523`, triangle `:471`/`:477`/`:485`/`:489`). |
| Roughness | working | Per-material linear roughness (`demo_programs.cu:524`). RR guide. |
| Specular hit-distance | working | Primary-hit → first specular-bounce distance, 1000 on miss (`demo_programs.cu:315`). RR reflection reprojection. |

All AOVs are **internal** (no per-AOV TD export); they exist to drive the denoisers, not as user outputs.

---

## 3. Denoisers

Selected by the **TD par** `Denoiser` menu: `None`, `Optix`, `Taa`, `Rr` (`OptixDemoTOP.cpp:740`). Irrelevant sliders are grayed per selection (`:383`).

| Denoiser | Status | Exposure | Notes |
|---|---|---|---|
| OptiX AI denoiser | working | menu `Optix` + `Denoisestr` | Temporal HDR model, albedo+normal+flow guides (`OptixDemoTOP.cpp:327`,`:687`). `blendFactor = 1 − strength`. Keeps previous denoised frame for temporal stability. |
| TAA (temporal reprojection) | works-but-fragile | menu `Taa` + `Maxhist` | Hand-rolled running average that follows the surface via motion vectors (`demo_programs.cu:375`). `Maxhist` caps history (lower = faster ghost fade). Per the RR memory, this was a **stopgap** for motion smudging that RR superseded. |
| DLSS Ray Reconstruction (RR) | works-but-fragile | menu `Rr` + `Flowscale`/`Flowinvy` | NGX CUDA path via `RRDenoiser` (DLAA, no upscale) on TD's shared CUDA context (`OptixDemoTOP.cpp:677`, `RRDenoiser.h`). Memory marks RR **quality user-accepted** after the MV-scale fix + full convention pass. Fragile: the MV convention (`Flowscale` sign / `Flowinvy`) is a tuning 2×2, and a plugin **reload corrupts the shared CUDA context** — see §10. |
| SVGF | todo | — | **Not in this codebase.** Grep over the engine source dir finds no SVGF symbols, and `LaunchParamsDemo.h` has no SVGF fields. SVGF exists only in a separate research/plan note (`PHASE3-optix-x-svgf.md`), not in the shipped plugin. |

**RR convention details** (from `dlss-rr-build-state` memory, matching code): real row-major world→view / view→clip matrices built host-side (`OptixDemoTOP.cpp:609`), view-space linear Z depth, free-running Halton(2,3) jitter `myExecuteCount&15` fed as `-(jitter−0.5)` (`:680`), specular hit-distance AOV, and a reset flag on resolution change / `Reset` / RR-off→on (`:621`).

---

## 4. Environment

Three sky modes via the **TD par** `Skymode` menu: `Gradient`, `Hdri`, `Physical` (`OptixDemoTOP.cpp:777`). Host maps to device `skyMode` 0/1/2. Sky evaluation in `skyDome` (`demo_programs.cu:92`).

| Feature | Status | Exposure | Notes |
|---|---|---|---|
| Gradient sky | working | **TD pars** `Skyzenith`/`Skyhorizon`/`Skystrength` | Horizon→zenith lerp by `0.5*(dir.y+1)` (`demo_programs.cu:101`). |
| Analytic sun disk | working | **TD pars** `Sundir`/`Suncolor`/`Sunstrength`/`Sunangle` | Tight `pow(dot,1500)*40` disk in gradient/physical modes; suppressed in HDRI mode to avoid a double sun (`demo_programs.cu:105`). |
| Equirect HDRI environment | works-but-fragile | menu `Hdri`, **input 4** | HDRI arrives as a **CUDA TOP input** (input 4), wrapped as a `cudaTextureObject_t` (`OptixDemoTOP.cpp:509`). Equirect lookup `demo_programs.cu:94`. Fragile because it is a CUDA texture input acquired conditionally on sky mode — the same acquisition pattern implicated in the freeze (§10). In the live snapshot input 4 = `PTEnv`. |
| HDRI rotation | working | **TD par** `Hdrirot` | Degrees → turns, added to equirect U (`OptixDemoTOP.cpp:642`, device `:95`). Rotation change resets accumulation (`:598`). |
| Preetham physical sky | working | menu `Physical` + `Turbidity` | Per-channel Perez xyY model from sun elevation + turbidity, computed host-side (`OptixDemoTOP.cpp:646`) and evaluated on device (`preethamSky`, `demo_programs.cu:78`). `0.05` exposure normalization. |
| Background mode | working | **TD par** `Backgroundmode` | Primary-miss background: `Environment` / `Solid` (`Bgcolor`) / `Transparent` (alpha matte) (`demo_programs.cu:321`). Transparent restores the path-traced `.w` after denoise (`OptixDemoTOP.cpp:702`). Mode change resets accumulation. |
| HDRI importance sampling + MIS | works-but-fragile | **TD par** `Envimportance` (+`Rebuildenv` pulse) | Host builds a 256×128 2D CDF (luminance·sinθ) from the downloaded HDRI (`buildEnvCDF`, `OptixDemoTOP.cpp:184`), throttled to ~every 15 cooks. Device importance-samples it as diffuse-bounce NEE (`sampleEnv`, `demo_programs.cu:172`). On diffuse bounces the passive sky is suppressed so NEE owns the env (`:328`). Fragile: this is **NEE-style direct env sampling**, not a full multi-strategy MIS weight (no balance-heuristic combine of BSDF+light pdfs in code) — describe it as importance-sampled env NEE, not classic MIS. Rebuild only runs in HDRI mode. |

---

## 5. Lights

Two independent light systems beyond the sun.

### 5a. NEE over emissive spheres

| Feature | Status | Exposure | Notes |
|---|---|---|---|
| Emissive-sphere NEE | works-but-fragile | **TD par** `Nee` (default **off**) | Power-weighted CDF over the procedural emissive spheres (`OptixDemoTOP.cpp:290`), sampled in `sampleDirect` (`demo_programs.cu:114`) with a shadow ray. **Default off and intentionally so:** per the RR memory, naive NEE *backfired* on this scene (~50 tiny emitters on the ground → 1/dist² fireflies) and was crushed by the firefly clamp. Disabled when in triangle/input mode (`OptixDemoTOP.cpp:634`). |

### 5b. Analytic Light COMPs (point / cone / distant)

| Feature | Status | Exposure | Notes |
|---|---|---|---|
| Point / cone / distant lights | working | **TD par** `Lightdata` (string) + `Lightintensity` | `PTLight` array (`LaunchParamsDemo.h:20`). NEE looped every diffuse bounce, pdf=1 (`sampleLights`, `demo_programs.cu:205`). Point/cone use 1/d² with a built-in ×50 balance so TD dimmer units ~1 read; distant has no 1/d². Cone spot falloff via smooth `cosInner`/`cosOuter`. |
| Soft shadows | working | per-light `radius` | Angular cone (distant) or sphere-sampled penumbra (point/cone) inside `sampleLights`. Sun has its own soft shadow via `Sunangle` (`demo_programs.cu:147`). |
| **String transport (not a TOP input)** | working | **TD par** `Lightdata` | Lights arrive as a serialized **string** (13 floats/light), parsed host-side into the device buffer (`OptixDemoTOP.cpp:546`). This is a deliberate workaround: feeding a Python Script TOP into a CUDA input deadlocks TD's cook thread (documented at `LaunchParamsDemo.h:101` and §10). Snapshot shows `PTLights` with an `execute` DAT (`lightbake`) that writes this param. |

`Lightintensity` is the master multiplier (`OptixDemoTOP.cpp:636`). Info CHOP exposes live `lights` and `ptLights` counts (`:724`,`:726`).

---

## 6. Input geometry (triangle soup)

When **`Useinput`** is on, the renderer renders a triangle GAS built from TD attribute textures instead of the procedural spheres.

| Feature | Status | Exposure | Notes |
|---|---|---|---|
| Triangle-soup geometry | working | **TD par** `Useinput` + `Numverts`, **input 0** | Positions texture (input 0) copied to a vert buffer; no-index soup (3 verts = 1 tri). GAS built/refit each cook (`buildInputTriGAS`, `OptixDemoTOP.cpp:336`). Cheap REFIT for same-count deformation, full rebuild on count change or every 30 refits. Snapshot input 0 = `soupTex` (a `popto` TOP). |
| Per-vertex color (Cd) | working | **input 1** | `triCd` (input 1, `soupCd`). Default grey if absent (`demo_programs.cu:442`). |
| Per-vertex material (type/rough/ior/emit) | working | **input 2** | `triMat` (input 2, `soupMat`). Decoded `demo_programs.cu:432`; `type>=10` flags per-object smooth normals. |
| Per-vertex smooth normals (N) | working | **input 3** | `triN` (input 3, `soupN`), barycentric-interpolated when the material flags smooth (`demo_programs.cu:434`). |
| Per-vertex UV (Tex) | working | **input 6** | `triUV` (input 6). Size-checked against geometry to avoid OOB (`OptixDemoTOP.cpp:503`). |
| UV debug view | working | **TD par** `Showuv` | Renders UV gradient, magenta = projection fallback (`demo_programs.cu:454`). |

**Confirmed-safe inputs:** inputs 0–3 are `poptoTOP` geometry attribute textures and are the verified-stable CUDA inputs (snapshot + memory + §10).

---

## 7. Per-object materials via texture (base-color map)

| Feature | Status | Exposure | Notes |
|---|---|---|---|
| Base-color map | **in-progress (blocked by the freeze)** | **input 5** + `Texmode`/`Projscale` | Intended: input 5 GPU TOP wrapped as a `cudaTextureObject_t` (`OptixDemoTOP.cpp:523`), UV-sampled if geometry has UVs else triplanar-projected from world pos (`demo_programs.cu:443`,`triplanar` `:250`). `Texmode` = Auto/Uv/Projection, `Projscale` = world units/tile. **Status: blocked.** Wiring a base-color TOP into input 5 is the known freeze trigger (§10); in the live snapshot **input 5 = EMPTY**. The device + host code paths exist and compile; the feature is unproven end-to-end because the input cannot currently be fed without hanging TD's main thread. |

The triplanar projection and UV-sampling device code is complete and exercised via the `Showuv`/`Texmode` debug paths; only the **live texture input** is blocked.

---

## 8. Motion vectors + TAA

| Feature | Status | Exposure | Notes |
|---|---|---|---|
| Motion vectors | working | internal | Reproject the first-hit world position through the previous-frame camera basis (captured pre-update, `OptixDemoTOP.cpp:591`) to get the previous screen location; flow = current − previous pixel (`demo_programs.cu:360`). Shared by OptiX flow guide, TAA, and RR. |
| TAA (temporal AA / accumulation) | works-but-fragile | menu `Taa` + `Maxhist` | Surface-following running average using the motion vectors (`demo_programs.cu:375`), double-buffered by parity (`OptixDemoTOP.cpp:666`). Described in memory as a stopgap superseded by RR for motion. |

---

## 9. Parameter & exposure summary

**User-facing TD parameter pages** (from `setupParameters`, `OptixDemoTOP.cpp:730`):

- **Render:** `Spp`, `Maxdepth`, `Firefly`, `Nee`, `Reset` (pulse)
- **Denoise:** `Denoiser` (menu), `Denoisestr`, `Maxhist`, `Jitter`, `Flowscale`, `Flowinvy`
- **Camera:** `Usecamera`, `Camera`, `Orbit`, `Orbitspeed`, `Distance`, `Aperture`, `Eye`, `Forward`, `Camfov`
- **Input:** `Useinput`, `Numverts`, `Showuv`
- **Environment:** `Skyzenith`, `Skyhorizon`, `Skystrength`, `Sundir`, `Suncolor`, `Sunstrength`, `Sunangle`, `Backgroundmode`, `Bgcolor`, `Skymode`, `Hdrirot`, `Envimportance`, `Rebuildenv` (pulse), `Turbidity`
- **Lights:** `Lightintensity`, `Lightdata` (string transport)
- **Texture:** `Texmode`, `Projscale`

**Info CHOP channels** (`getInfoCHOPChan`, `:715`): `executeCount`, `optixOK`, `ready`, `frameIndex`, `spheres`, `rrInit`, `rrResult`, `lights`, `envReady`, `ptLights`.

**Inputs** (`maxInputs = 7`, `OptixDemoTOP.cpp:93`): 0=positions, 1=Cd, 2=Mat, 3=N, 4=HDRI, 5=base-color map, 6=UV.

---

## 10. Known headline issue — "the freeze"

Feeding certain TD TOPs into the renderer's CUDA inputs **hard-hangs TouchDesigner's main thread** (white window, "not responding") with **no GPU TDR** in the Windows event log — a CPU/main-thread **deadlock**, not a crash or GPU timeout.

**Confirmed-safe:** `poptoTOP` geometry attribute textures (inputs 0–3).
**Confirmed-deadlock:**
- A Python Script TOP into a CUDA input (GIL / cook-thread re-entrancy) — **solved** by moving that data to the `Lightdata` string transport (§5b, `LaunchParamsDemo.h:101`).
- A raw GLSL TOP at input 5 (base-color map); and that same GLSL TOP through a Null TOP (the "Null fixes GL→CUDA interop" theory was tried and **disproven**).

**What the code actually shows (relevant to the leading hypothesis):** `execute()` acquires CUDA arrays for the wired inputs **before** `beginCUDAOperations`. The HDRI input (4) is acquired **only in HDRI sky mode** (`OptixDemoTOP.cpp:445`), while the base-color input (5) is acquired **whenever `getNumInputs()>5`** (`:438`) — i.e. whenever ≥6 inputs are connected. Because TD's connection model is **contiguous** (you cannot connect input 5 while input 4 is empty), `getNumInputs()>5` means input 4 is connected too, so this gate is equivalent to `getInputTOP(5)!=nullptr`; it is **not** a connection gap and **not** a live bug. The real asymmetry is an **acquisition** gap: input 4 is **connected but SKIPPED** (its `getCUDAArray` is sky-mode gated and not called when sky mode ≠ HDRI), while input 5 **is** acquired. So in **Physical-sky mode** (the live snapshot state — `PTEnv.Mode = 'Physical'`) with a base-color TOP wired into input 5, `execute()` calls `getCUDAArray` on the **acquired set `{0,1,2,3,5}`**, skipping the **connected** input 4. The one demonstrably-working texture input (the HDRI) happened in HDRI mode, where the acquired set was **contiguous** `{0,1,2,3,4}`. This is consistent with the **acquisition-asymmetry** hypothesis (acquiring a higher input while skipping a connected lower-index input) over the GL-interop theory — but note the code grounds the *acquisition pattern*; it does not prove the *cause*. **[unverified]** whether the skip itself is the deadlock trigger vs. the input-5 source type. The per-index null check `getInputTOP(i)!=nullptr` would be clearer and gap-robust than the `getNumInputs()>5` count gate, even though the two are equivalent under TD contiguity.

**RR reload hazard (separate):** after a plugin reload, the previous instance's NGX shutdown can corrupt the shared CUDA context (`ready=0`/`optixOK=0`). Mitigated by **not** calling the global `NVSDK_NGX_CUDA_Shutdown` (per memory), but a corrupted process needs a TD restart.

---

## 11. Discrepancies between running assumptions and the code

These are differences worth flagging between the briefing/CONTEXT and what the source actually says:

1. **HDRI is input 4, not input 5.** The CONTEXT and even some in-code parameter labels disagree with the wiring. The `Skymode` menu label reads *"HDRI (equirect, input 5)"* and the `Texmode` page comment says *"base color map = the 6th input"* (`OptixDemoTOP.cpp:778`,`:789`). But the actual acquisition code uses **`getInputTOP(4)` for HDRI** and **`getInputTOP(5)` for base color** (`:445`,`:438`), and the live snapshot confirms input 4 = `PTEnv`, input 5 = EMPTY. The user-facing labels are off-by-one (1-based vs 0-based); the code indices are the truth.

2. **The freeze hypothesis is consistent with the code, but the asymmetry is in *acquisition*, not connection.** TD's contiguous-connection model means input 5 can never be connected with input 4 empty, so `getNumInputs()>5` (the input-5 gate at `:438`) is equivalent to `getInputTOP(5)!=nullptr` and is not a connection bug. The real asymmetry: input 4's `getCUDAArray` is sky-mode gated (skipped when sky ≠ HDRI) while input 5's is not, so a Physical-sky + base-color combo acquires `{0,1,2,3,5}` and **skips the connected input 4**. This matches the leading hypothesis. It does **not** by itself prove the deadlock cause — the input-5 source being a GL/GLSL TOP remains a confounder.

3. **"MIS" is overstated.** The env importance-sampling path (`sampleEnv`) is **NEE-style importance sampling of the HDRI**, not a full multiple-importance-sampling combine (no BSDF-vs-light balance-heuristic weighting in code). It suppresses the passive sky on diffuse bounces so NEE owns the env; that is single-strategy NEE, not MIS.

4. **SVGF is not implemented.** It appears in CONTEXT and a phase-3 research doc but has zero presence in the plugin (no symbols, no LaunchParams fields). Listed as **todo**.

5. **NEE over emissive spheres is intentionally OFF and known-bad on this scene** — not merely "available." The default and the memory both say it backfired (1/dist² fireflies on many small ground emitters).

6. **No in-plugin tonemapping.** The TOP emits linear HDR; tonemapping is a downstream TD GLSL TOP (snapshot `tonemap`). Worth stating so "the look" isn't attributed to the path tracer.
