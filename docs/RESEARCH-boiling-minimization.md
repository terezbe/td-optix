# RESEARCH — Minimizing "boiling" (temporal noise), esp. with fog

**Date:** 2026-06-29 · Method: 7-agent UltraCode workflow (4 readers → 2 designers → adversarial critic), then code-verified.

## TL;DR
Boiling has two regimes. **(1) No-fog (sky + specular)** is **already fixed** in the working tree (uncommitted): the sky infinite-distance MV (`aoDir`) and the specular MV (`rrSpecMV`) both shipped. **(2) With-fog** is the dominant remaining problem and is **not** fixable by DLSS-RR (it's a *surface* denoiser; fog in-scatter has no surface to reproject) nor by a downstream temporal post-pass (would double-stack on RR). **The real fix is engine-side, at the fog noise source — and the single highest-leverage change is decorrelating the RNG seed under motion.**

## ✅ Implementation status (2026-06-29)
- **STEP 1 (decorrelate RNG seed) — SHIPPED & VALIDATED.** `sampleSeed` field (LaunchParamsDemo.h), seed at `demo_programs.cu:425`, host `lp.sampleSeed = Boildecorrelate ? myExecuteCount : 0`. Wrapper toggle **`PT_Render.par.Boildecorrelate`** (Denoise page, bound EXPRESSION), **default ON** (validated net-positive). A/B under a fixed fog motion path (RR): per-frame **spatial noise −8.3%** (Immerkär 0.0145→0.0133); glass/chrome surfaces visibly cleaner (`boil_off.png` vs `boil_on.png`). NB: *temporal-std* rose (0.0042→0.0057) — a **confounded metric** here (frozen-seed noise is a static screen-space pattern = low temporal-std but grainy); spatial noise + visual are the correct measures.
- **STEP 2 (fog in-scatter firefly clamp) — SHIPPED.** `fogFireflyMax` field, clamp at `demo_programs.cu:~505` (clamps `fc` into BOTH `L` and `fogS`), added to the `fogSig` hash. Wrapper **`PT_Render.par.Fogfireflyclamp`** (Fog page, EXPRESSION), **default 0 (off)** — situational safety valve; effect is in HDR (the Reinhard tonemap masks it in the output-domain metric).
- **STEP 3 (motion-reproject the Fogstability EMA) — NOT YET DONE** (next, the bigger fog win). STEPs 4–5 pending.
- Built via `scripts/compile_ptx.ps1` + `scripts/build_dll.ps1`, hot-reloaded; engine healthy (optixOK/rrInit/rrResult=1), 0 errors. Both params byte-reversible at off.

## Verified ground truth (code, not docs)
- ✅ `aoDir` sky MV: `demo_programs.cu:432` (init), `:475` (capture `aoDir=rd`), `:589` (`dpv = aoHit ? aoPos-prev_eye : aoDir`).
- ✅ `rrSpecMV` specular MV gate: `LaunchParamsDemo.h:75`, applied `demo_programs.cu:650` (`specW = rrSpecMV*(1-clamp(aoRough*4))`), host `OptixDemoTOP.cpp:941`. Default 1.0; 0 = exact prior.
- ✅ `fogStability` EMA: `LaunchParamsDemo.h:159/161`, `demo_programs.cu:571-576`, host `:934`. Default 0 = off. **Reads the same pixel `fogAccum[pix]` — no motion compensation → ghosts under motion** (why its default is 0).
- ❌ `sampleSeed`, `fogFireflyMax`, `fogReproject`, `fogSpp` — do **not** exist yet (the new work below).

## Why fog boils worse (the mechanism)
Turning fog on injects three genuine Monte-Carlo sources RR cannot remove:
1. **Free-flight Bernoulli** scatter-vs-reach-surface coin flip per pixel (`demo_programs.cu:500-501`), variance scales with `fogDensity` — the #1 cause.
2. **Single-sample in-scatter NEE** at a random scatter depth (`:504`, sun-cone + HDRI 2D-CDF + 1/d² emitter pick).
3. The **b==0 in-scatter lands in the DIRECT bucket → bypasses the existing firefly clamp** (which only clamps indirect, `:556-558`).

All three are then **frozen in screen space by the RNG seed** (`:425`), because under motion the host pins `frameIndex=1` (`OptixDemoTOP.cpp:866`) → identical RNG stream every frame → static grain that even the Fogstability EMA averages to ~nothing.

## The GLSL-post-pass verdict (what the user asked about)
- A **temporal/feedback** post-pass on the render output is **FORBIDDEN** as the default-mode fix: DLSS-RR is already the sole temporal integrator (raw `rrColor` is load-bearing) → stacking a 2nd temporal filter = ghosting. **And** the C++ TOP exposes a **single RGBA32F output** — no depth/normal/MV downstream — so a post-pass *cannot* motion-reproject; under motion it smears or degrades to a no-op exactly where boiling is worst.
- A **spatial-only** post-pass (single-frame firefly/variance clamp) is the **only** safe downstream option — useful as a last-resort safety net, not the primary fix. A bilateral/à-trous *blur* is discouraged (it over-softens RR's already-denoised detail and temporal-std A/B is blind to that over-blur).
- The **only** legal temporal stabilization touching the RR pipeline is the **existing fog-channel EMA** (fog has no surface for RR, so it's the documented exception); improving it (STEP 3) stays within that exception.

## Recommended plan (engine-side, RR-safe, byte-reversible; in priority order)
| # | Fix | Where | Effort | New wrapper param (default = off/no-op) |
|---|---|---|---|---|
| **1** | **Decorrelate RNG seed** under motion (`sampleSeed` = free-running counter; seed falls back to `frameIndex` when 0) — *prerequisite that lets RR + the fog EMA average fog noise instead of a frozen pattern* | `LaunchParamsDemo.h` new field; `demo_programs.cu:425`; host `OptixDemoTOP.cpp:~871` (`lp.sampleSeed = Decorrelate ? myExecuteCount : 0`) | LOW (~6 lines + PTX) | `Boildecorrelate` (Toggle, 0) |
| **2** | **Fog in-scatter firefly clamp** (clamp `fc` before adding to BOTH `L` and `fogS`) — kills the unclamped b==0 fog fireflies | `demo_programs.cu:~505`; `LaunchParamsDemo.h`; host `:~933` + add to `fogSig` hash | LOW | `Fogfireflyclamp` (Float 0..20, 0) |
| **3** | **Motion-reproject the Fogstability EMA** through the already-computed `flow` MV (bilinear gather + disocclusion fallback) — fog history follows the camera instead of ghosting; makes the existing knob usable at strength | `demo_programs.cu:571-576` (move after `flow`); host + `fogSig` | MEDIUM | `Fogstabilityreproject` (Toggle, 0→1 after A/B) |
| **4** | *(optional)* **Fog-only sub-sampling** (`fogSpp`, or stratified single-scatter quadrature) — cut free-flight variance cheaply without raising whole-path Spp | `demo_programs.cu:498-521`; host + `fogSig` | MEDIUM | `Fogsamples` (Int 1..8, 1) |
| **5** | *(last resort)* **Downstream spatial firefly-clamp GLSL** `boilsuppress` between `render`→`tonemap` (linear HDR; 3×3 luma mean+std, rescale outliers chroma-preserving; **no history**) — denoiser-agnostic safety net | new GLSL TOP in PT_Render | LOW | `Boilreduce` (Toggle 0) + `Boilstrength` (0..1, 0); bypass via `1-Boilreduce` |

## Pitfalls (carry into implementation)
- **Stale-finding trap:** sky-MV + spec-MV are SHIPPED — do **not** re-apply `RESEARCH-camera-motion-noise` / `IMPL-specular-motion-fix`; touching that load-bearing MV/jitter code risks regressing working fixes.
- **Binding mode:** set `par.mode = EXPRESSION` explicitly on every inner bind (the recurring "wrapper does nothing" = correct expr stuck CONSTANT). See [[pt-render-binding-mode-pitfall]].
- **fogSig hash:** every new fog param must be added to the fog-reset hash (`OptixDemoTOP.cpp:~860`) or the `fogAccum` EMA won't refresh → stale ghosting.
- **Reversibility byte-exact:** `sampleSeed==0`, `fogFireflyMax==0`, `fogReproject==0`, `fogSpp==1`, `Boilreduce` off must each reduce to today's path — verify with the `image_delta` no-op detector, not by eye.
- **Never** tie `myRRReset` to `moved` (permanent noise — documented dead end). The seed change must not touch `myRRReset`.
- **Measure fog correctly:** fog at `Fogskystr=0` grays the bg — baseline with fog OFF or `Fogskystr>0.1`; freeze `pcam.tx`, manifest every capture. Validate with temporal-std **and** FLIP vs a converged reference (temporal-std is blind to over-blur).
- **PTX rebuild** required for STEPs 1–4 (hot-reload: `unloadplugin=True` → nvcc → `False` + `reinitpulse`); verify the `.ptx` timestamp (a stale `.ptx.bak` can mask it).

## Recommended starting point
**STEP 1 (seed decorrelation)** — highest leverage, lowest risk, and the prerequisite that makes the fog EMA + STEP 3 actually work. Implement + A/B with `tools/param_validate.py temporal_boil` on a fixed fog motion path before moving to STEP 2/3.
