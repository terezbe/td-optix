# Moving-Camera Background Noise in OptixDemoTOP — Diagnosis & Fix Design

**Status:** Research / design only — no implementation in this document.
**Date:** _(placeholder — fill on commit)_
**Component:** `phase2/OptixDemoTOP` (OptiX path tracer + DLSS Ray Reconstruction)
**Branch context:** `pt-render-comp`

---

## 0. The symptom

- **Static camera:** the rendered image is clean. Background (sky / HDRI / gradient) and foreground geometry both look stable and converged.
- **Camera animated (slow LFO orbit):** the **entire background turns into heavy, pixelated, boiling noise** that shimmers frame-to-frame, while the **foreground geometry stays comparatively clean**.
- Reproduces with the **default `Rr` (DLSS Ray Reconstruction) denoiser** path. The LFO both translates and rotates the view (an orbit), so the on-screen sky direction genuinely changes every frame.

This is the classic "skybox has no velocity, so the temporal denoiser breaks on the background under camera rotation" failure — confirmed below against this engine's code.

---

## 1. Root cause

**Hypothesis CONFIRMED by all three independent investigations**, with one mechanistic wording correction.

The engine writes a **zero motion vector for every primary-miss (sky/background) pixel**, and additionally hands DLSS-RR a **degenerate `(0,0,0)` normal guide** for those pixels. A zero MV is the *correct* answer only when the sky is static on screen. Once the camera rotates, the true sky velocity is non-zero, so DLSS-RR reprojects each sky pixel to the **same screen location** in its history — a location that showed a **different sky direction** last frame. RR's internal consistency/disocclusion test then rejects the mismatched history every frame across the whole background, so it can never temporally integrate the sky and instead outputs an unstable, poorly-reconstructed flat region → the boiling pixelated noise the user sees. Foreground pixels carry valid surface MVs and valid normals, so RR reprojects and integrates them correctly → they stay clean.

**Wording correction to the original hypothesis:** the sky MV is *not* "garbage / NaN / huge produced by reprojecting a zero hit position." The miss-reprojection block is **gated off** and `flow` is **explicitly hard-set to exactly `(0,0)`**. This is arguably worse than garbage: it positively asserts to DLSS-RR "this sky pixel did not move," which is false under any rotation. Net downstream effect is identical, but the fix targets the gating, not a numerical blow-up.

### Code evidence (verified in-tree)

**Device — `phase2/OptixDemoTOP/demo_programs.cu`:**

- **L423** — AOVs initialised: `float3 aoAlb=V(0,0,0), aoNrm=V(0,0,0), aoPos=V(0,0,0); ... int aoHit=0;`
- **L436 / L443** — primary world-space ray direction is available pre-bounce: `float3 dir=add(add(params.cam_w, scl(params.cam_u,sx)), scl(params.cam_v,sy));` then `float3 rd=norm(dir);`
- **L460–462** — primary-hit AOV capture. On a **miss**: `if(prd.miss){ aoAlb=prd.skyColor; aoNrm=V(0,0,0); aoHit=0; }` — `aoPos` is **never written** (stays `(0,0,0)`), normal is the degenerate zero vector.
- **L503–506** — the directly-visible background is a **deterministic** lookup, not a Monte-Carlo estimate: `else { envc=add(skyDome(rd), sunDisk(rd)); }` (or `bgSolid` / transparent). A 1-spp primary sky pixel is essentially clean.
- **L545–553** — the motion vector is computed **only** `if(aoHit && params.validPrev)`, by reprojecting `sub(aoPos, params.prev_eye)` through the previous camera basis (`prev_u/prev_v/prev_w`). The projection is scale-invariant (it divides by `dot(prev_*,prev_*)`).
- **L554** — the `else` branch: `float2 flow = haveProj ? ... : make_float2(0.0f,0.0f);` → **every sky pixel gets `flow=(0,0)`**.
- **L582** — `params.normal[pix]=make_float4(aoNrm.x,aoNrm.y,aoNrm.z,0.0f);` → degenerate sky normal forwarded to the guides.
- **L587** — `params.rrColor[pix] = make_float4(col.x, col.y, col.z, 1.0f);` → DLSS-RR consumes the **raw per-frame** color, NOT the accumulator.
- **L592–593** — `params.rrDepth[pix] = aoHit ? vz : 10000.0f;` (far on miss — sensible) and `params.rrMotion[pix] = make_float2(flow.x, flow.y);` → the **zero** sky MV is what DLSS-RR is told.

**Host — `phase2/OptixDemoTOP/OptixDemoTOP.cpp`:**

- **L812** — `bool moved = !myHavePrev || hlen(hsub(eye,myPrevEye))>1e-5f;` — keys on **eye translation only**.
- **L827–828** — `if(myResetReq || moved || bgChanged || skyChanged || texMapChanged || fogChanged){ myFrameIndex=0; ... } myFrameIndex++;` — the progressive running-sum accumulator restarts on any motion.
- **L848–849** — `if(rr && !myPrevRR) myRRReset=true; ... int rrReset = myRRReset?1:0; myRRReset=false;` — DLSS-RR history is reset **only** on RR-enable transition / resolution change / explicit Reset, and correctly **not** on `moved`.
- **L915–918** — DLSS evaluate call; passes **raw `(float)flowscale` for both `InMVScaleX` and `InMVScaleY`** plus `flowinvy`.

**DLSS wrapper — `phase2/OptixDemoTOP/RRDenoiser.cpp`:**

- **L94–95** — feature created with `NVSDK_NGX_PerfQuality_Value_DLAA` and `IsHDR | MVLowRes`. (DLAA → render res == output res, so any MV error is fully exposed, not masked by upscale resampling.)
- **L135** — `ep.InMVScaleX = mvScaleX;` with a **stale comment** claiming "host passes flowscale/W" — the host actually passes raw `flowscale` (L918). MVs here are render-pixel-space; the `flowscale` knob is a sign/scale multiplier (default **-1.0**, `Flowinvy` default **1.0**, see `OptixDemoTOP.cpp:994–995`).

### What this rules out

- **The accumulator-reset-on-motion is NOT the cause of the RR-mode bug.** In RR mode the accumulator (`params.accum`, L572–577) is **bypassed** — DLSS consumes `rrColor=col` (L587). So "`myFrameIndex` resets to 1 every frame under motion" is real (L827) but **irrelevant** to what RR sees. It only degrades the *non-RR / TAA* paths.
- **It is NOT 1-spp path noise resurfacing on the background.** The directly-visible sky is a deterministic `skyDome+sunDisk` / HDRI `tex2D` lookup (L503–506), so the per-frame sky input is essentially clean. The "noise" is **DLSS-RR temporal reconstruction instability driven by the zero MV + degenerate normal guide**, not sampling variance. Corollary: **more spp will not fix this**, and env importance sampling / ReSTIR will not fix this background.

---

## 2. Why the BACKGROUND specifically

Three compounding asymmetries between foreground (`aoHit==1`) and background (`aoHit==0`) pixels:

1. **Motion-vector asymmetry (the lynchpin).** Foreground pixels reproject a real world hit position through the previous camera → a correct non-zero MV (L545–553). Background pixels are forced to `flow=(0,0)` (L554). Under a static camera the sky's true screen velocity is zero, so `(0,0)` is correct → clean. Under rotation the sky genuinely moves on screen but is told it didn't → RR fetches the wrong co-located history → rejects it → unstable reconstruction over the whole background.

2. **Weak / degenerate AOV guides for the sky.** RR uses normal + albedo to guide *spatial* reconstruction when temporal history is unavailable. The sky's normal is `(0,0,0)` (L461/L582) — RR has nothing to spatially lock onto — and the albedo guide is the sky color **clamped to [0,1]** (L581), discarding HDR structure (bright sun/sky). So when the broken temporal channel drops history, the spatial fallback is also starved → blocky boiling rather than a graceful blur.

3. **No within-frame AA on the sub-pixel sky/sun.** In RR mode the jitter is **constant across all spp** (L431: `if(params.rrEnable){ jx=params.rrJitterX; jy=params.rrJitterY; }`), so the deterministic sky and the **extremely sharp sun disk** (`powf(s,1500.0f)*40.0f`, see `sunDisk`/L101–111) get no multi-sample AA within a frame — they rely entirely on cross-frame DLSS accumulation, which is exactly what the zero sky MV destroys. Result: aliased, flickering background, worst around the sun.

Foreground does not suffer any of these: valid MV, valid unit normal, real (often near-Lambertian, low-frequency) albedo.

> Note: if **fog is enabled**, `inscatterFog` (L484) makes the through-fog background genuinely stochastic — a *separate, real* MC noise source on top of the MV bug. Fog is optional and not required to reproduce the reported symptom; treat it as out of scope for the primary fix.

---

## 3. Fixes

Ranked by effect-per-effort. Fix (a) is the lynchpin and is expected to resolve the reported symptom on its own.

### (a) Correct sky / infinite-distance motion vector — **PRIMARY FIX**

**What it is.** For miss pixels, replace `flow=(0,0)` with a **rotation-only, infinite-distance reprojection of the primary view direction** through the previous camera basis. A point at infinity has **zero parallax under translation**, so we reproject the *direction* (drop the `prev_eye` term that surfaces use). This gives the sky a correct non-zero MV under rotation and ~zero under pure translation.

**Exact engine hooks.**
- Capture the primary ray direction in the `b==0` AOV block (`demo_programs.cu` ~L460), e.g. an `aoDir = rd;` (using the normalized primary `rd` from L443). It **must** be captured before the bounce loop reassigns `rd` (L525).
- In the reprojection block (L545–554), branch the reprojected vector:
  - `aoHit` → `dpv = sub(aoPos, params.prev_eye)` (unchanged surface path).
  - `!aoHit` → `dpv = aoDir` (direction only; **no** `prev_eye` subtraction).
  - then run the **existing** scale-invariant projection (`dot3(dpv,prev_w)`, `prev_u`, `prev_v`) to get `ppx,ppy` and set `haveProj=true` when `awp>1e-4f`.
- `flow` (L554) then becomes non-zero for the sky, and is written to both `params.flow` (L583) and `params.rrMotion` (L593) automatically.

Because the sky MV is emitted in the **identical render-pixel units** as the surface MV, it inherits the existing `InMVScaleX/Y = flowscale` convention with **no scale change**.

**Expected effect.** Background reprojects correctly under camera motion → the heavy background noise/boiling disappears. As a free bonus, the same `haveProj/ppx/ppy` also feeds the non-RR **TAA EMA** path (L558–569), giving that path a real sky history too (one change fixes both temporal paths).

**Effort.** Low (~10–15 lines + PTX recompile). **Risk.** Low — touches only the miss branch of the MV write; the surface MV path is untouched.

**Guards / caveats.**
- For rotations >~90°, a sky direction can fall *behind* the previous camera (`awp<=0`); the `awp>1e-4f` guard falls back to `(0,0)` + disocclusion — correct degenerate handling (a 1-frame noise band on very fast spins, same as foreground disocclusion).
- Must reproject by **rotation only** (point at infinity). Using a finite world point would over-predict sky motion under translation and introduce a *different* artifact.
- This assumes `prev_u/prev_v/prev_w` are the previous-frame camera **axes** scaled by halfW/halfH/focus and **eye-independent** — confirmed at `OptixDemoTOP.cpp:810/829`. A future refactor that bakes eye into the basis would break it.

### (b) Valid sky normal guide

**What it is.** Replace `aoNrm=V(0,0,0)` on miss (L461) with a unit vector — the natural choice is `-rd` (view direction toward camera) — so DLSS-RR has surface structure to spatially reconstruct the background instead of a degenerate zero normal.

**Hooks.** `demo_programs.cu:461` → `params.normal` write at L582.
**Expected effect.** Spatial fallback for the sky improves; reduces residual boiling even before temporal history settles. **Effort.** Trivial. **Risk.** Low — but **shared guide caveat**: `params.normal` (and `params.albedo`) feed **both** the DLSS-RR guides *and* the legacy OptiX-denoiser branch (`OptixDemoTOP.cpp:921+`). Verify both denoiser modes still look right.

### (c) DLSS-RR feeding / accumulator architecture — **confirm, don't change**

**Finding.** The current architecture is already correct for RR: feed DLSS the **raw per-frame** color (`rrColor=col`, L587) and let DLSS-RR be the sole temporal integrator; do **not** route the progressive accumulator into RR; and do **not** tie `myRRReset` to `moved` (L848–849). Resetting RR history every motion frame would cause *permanent* noise. **Keep all of this as-is.** This is documented here so a future contributor does not "fix" the accumulator and regress.

**Lower-priority option (Fix 6 in source reports):** feed DLSS a motion-reprojected/decayed history instead of raw per-frame color. Not needed once (a) lands; **medium effort, medium risk** (ghosting if MVs are wrong). Defer.

### (d) Lower-variance env sampling & sky AOV handling

**What it is.** (i) Stop hard-clamping the sky albedo guide to [0,1] (L581) — pass an unclamped/tonemapped sky guide so RR has real HDR background structure. (ii) Tighter env importance sampling / ReSTIR DI.

**Reality check.** The *directly-visible* background is deterministic (§1), so neither helps the reported symptom. (i) is a minor guide-quality improvement (low effort, low risk; same shared-guide caveat as (b)). (ii) only helps **surface shading and reflected/indirect sky** — pursue later for surface firefly/variance, **not** for this bug. **De-prioritized.**

### (e) spp / jitter during motion

**What it is.** Either vary the RR jitter per-spp (L431) for within-frame AA, or ramp spp (1→2–4) while `moved` is true to give RR/TAA a cleaner per-frame input on disocclusions and reflections. Optionally soften the sub-pixel sun disk (`powf(s,1500)*40`) so it flickers less without temporal accumulation.

**Reality check.** Does **nothing** for the directly-visible background (deterministic; jitter is constant across spp by design). Helps **surfaces and reflected sky** only, and costs frame budget exactly when the camera is moving. **Low priority**, cap any ramp to measured headroom. **Effort** low–medium. **Risk** low.

### (f) Verify the MV scale/sign convention — **verify, don't assume**

**What it is.** `flow` is pixel-space (L554) and `MVLowRes` is set (RRDenoiser.cpp:95), so `InMVScaleX/Y` should be a pixel-unit multiplier. The host passes raw `flowscale` (default **-1.0**, `OptixDemoTOP.cpp:918/994`), while the `RRDenoiser.cpp:135` comment misleadingly says "flowscale/W." Re-confirm the runtime value/sign matches the pixel-space convention. A wrong **global** scale degrades **all** reprojection (foreground too), so it is *not* the background-specific cause — but given this project's documented MV-scale history (see MEMORY: DLSS-RR build state), verify it before/while landing (a) so a pre-existing scale error is neither masked nor compounded. **Effort** low. **Risk** low.

---

## 4. Phased plan (most-impact-first)

> Verification method for every phase: animate the camera with the slow LFO orbit, then **numpy pixel A/B on a background ROI** across the orbit (sky-region noise variance should fall toward the static-camera baseline) and a **Gemini Vision** pass ("is the sky clean and stable during the camera move?"). Add a temporary **`rrMotion`→color debug viz**: under motion the sky should change from flat black (the bug) to a smooth rotation-rate gradient (the fix).

**Phase 1 — Sky motion vector (Fix a). [TOUCHES THE DLSS/AOV/INTEROP PATH — flag]**
The single change expected to resolve the symptom. Edit `demo_programs.cu` (capture `aoDir` at L460; branch `dpv` at L545–554), recompile `demo_programs.ptx`, clean TD reload (NGX/CUDA reload caveats per MEMORY — beware the stale `demo_programs.ptx.bak` in git status masking the build). **Sign check:** if the sky ghosts *opposite* to the camera motion, flip `Flowscale` (-1 ↔ +1) or toggle `Flowinvy`. Verify on the background ROI + debug viz.

**Phase 2 — Sky normal guide (Fix b) + unclamped sky albedo guide (Fix d-i). [TOUCHES THE AOV/GUIDE PATH — flag, shared with OptiX denoiser]**
Small guide-quality improvements that harden the background's spatial fallback. Verify **both** the `Rr` and legacy OptiX-denoiser modes still look correct (shared `params.normal`/`params.albedo`).

**Phase 3 — Convention verification (Fix f). [TOUCHES THE DLSS MV PATH — flag]**
Confirm `flowscale`/`flowinvy` against the pixel-space `MVLowRes` convention; fix the stale `RRDenoiser.cpp:135` comment. Do as a guarded check alongside Phase 1, not a blind change.

**Phase 4 — Non-RR path hygiene (optional, no DLSS risk).**
Broaden host `moved` (L812) to also detect camera-**basis** (u/v/w) change so the non-RR progressive accumulator and motion-gated logic respond to pure pans/tilts (today a pure rotation never resets and smears orientations). Gate so a truly static camera still gets full progressive accumulation (preserve hero-shot quality). Optional spp-during-motion ramp / per-spp jitter (Fix e) for surfaces & reflected sky. None of this affects the RR background symptom.

**Phase 5 — Deferred.** Env IS / ReSTIR DI for surface/indirect variance; motion-reprojected history into DLSS (Fix c-option). Only after Phases 1–3 confirm the background is clean.

---

## Sources

- `phase2/OptixDemoTOP/demo_programs.cu` — raygen `__raygen__rg`: AOV init L423; primary-miss AOV capture L460–462; deterministic background L503–506; surface MV reproject L543–553; **miss→`flow=(0,0)` L554**; TAA EMA vs plain accumulator L556–578; AOV/flow writes L580–583; RR G-buffer writes (rrColor raw L587, rrDepth L592, **rrMotion zero on sky L593**); RR-constant jitter L431; primary ray dir L436/L443; sun disk `powf(...,1500)` (L101–111).
- `phase2/OptixDemoTOP/OptixDemoTOP.cpp` — prev-camera capture L808–810; `moved` = eye-translation-only L812; accumulator reset L827–828; prev-basis/`validPrev` wiring L829/L858; view16/proj16 + Halton jitter L831–847; **RR reset only on enable transition L848–849**; DLSS evaluate w/ raw `flowscale` L915–918; `Flowscale` default -1.0 / `Flowinvy` default 1.0 L994–995.
- `phase2/OptixDemoTOP/RRDenoiser.cpp` — DLAA + IsHDR + MVLowRes L94–95; DLSSD eval params, `InMVScaleX/Y`, `InReset`, view/proj matrices L128–138; **stale "flowscale/W" comment L135**.
- `phase2/OptixDemoTOP/LaunchParamsDemo.h` — prev-camera + RR G-buffer param layout.
- Background technique: Schied et al. 2017 SVGF (reprojection + history accumulation for 1-spp); Schied 2018 A-SVGF; NVIDIA DLSS 4 Ray Reconstruction docs (MVs + prior-frame feedback, disocclusion); NVIDIA Streamline DLSS/DLSS-RR Programming Guides (MV space, `InMVScale`, `InReset`, jitter); skybox/infinite-distance reprojection (UE `SkyVelocity`, Godot #67332, three.js infinite-distance background); TAA writeups (Tardif, El Lopez, Brash & Plucky); NVIDIA forum "DLSS-RR swimming/flowing under camera motion."