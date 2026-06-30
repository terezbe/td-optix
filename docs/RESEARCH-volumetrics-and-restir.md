# Research: Volumetric Fog / God Rays & ReSTIR DI+GI

**Status:** research / design — nothing implemented yet.
**Date:** 2026-06-25.
**Target engine:** `OptixDemoTOP` — OptiX 9 **megakernel** path tracer (`__raygen__rg` in `demo_programs.cu`, loop-in-raygen, PCG32 RNG seeded per pixel by `(x,y,frameIndex)`), driven from a TouchDesigner C++ TOP.

This note distills two deep research passes (each cross-checked against 2012–2024 primary literature) plus a line-level audit of the engine, into one actionable plan. It answers: *what would it take to add (A) volumetric fog / god rays and (B) ReSTIR DI+GI to this specific engine, and in what order.*

---

## 0. Executive summary

**Engine readiness (audited, with line refs in §1):**
- **Volumetric fog / god rays — ~60% ready.** The path loop, NEE light samplers (`sampleSun`/`sampleEnv`/`sampleLights`/`sampleDirect`), HG-friendly RNG, and accumulation are all there. Missing: the medium itself (distance sampling, transmittance, phase-function in-scatter). No density grid needed for homogeneous fog — it's analytic.
- **ReSTIR DI — ~40% ready.** Critically, the **motion vectors, world normal, linear depth, albedo G-buffers already exist** (built for DLSS Ray Reconstruction), plus the previous-frame camera basis and a TAA reprojection buffer. That *is* the temporal-reuse infrastructure. Missing: reservoir buffers + temporal/spatial reuse passes, and one refactor (decouple NEE candidate-generation from the shadow ray).
- **ReSTIR GI — harder.** The megakernel resolves each path in one launch with no seam to persist secondary-vertex reservoirs; needs either a wavefront refactor or a GI-only post-pass (recommended).

**Recommended order (most value per unit risk):**
1. **Homogeneous fog (free-flight)** — small, unbiased, composes with accumulation for free. Immediate visual payoff (aerial perspective, light glow).
2. **God-ray mode (HG + equiangular)** — the dramatic "light shafts" look; moderate cost (shadow rays per march step).
3. **ReSTIR DI** (temporal → spatial) — the biggest *quality/noise* win for direct lighting; reuses existing MV/G-buffer infra.
4. **ReSTIR GI** (post-pass) — biggest win for indirect noise, but most invasive; do last.

**Two hard constraints to respect from day one:**
- **Don't stack temporal filters.** ReSTIR-temporal × `accum`/TAA averaging × DLSS-RR = triple smear/ghost. Feed *per-frame* (un-accumulated) output to DLSS-RR; let ReSTIR carry sampling history and RR carry image reconstruction.
- **Hand-roll, don't import RTXDI.** It's HLSL/D3D12-Vulkan and would fight TD's CUDA/Vulkan interop (the documented `beginCUDAOperations` deadlock class). Our light count is small, so RTXDI's many-light tiling is unnecessary. Use RTXDI/papers as the *spec*, implement as CUDA/OptiX.

---

## 1. Engine audit — the hooks (file:line)

Megakernel path loop: `demo_programs.cu:263–413` (`__raygen__rg`); bounce loop `302–356`; per-sample accumulate `354`.
Accumulation: `accum` running-sum + `taaPrevCol/taaCurCol` ping-pong (`LaunchParamsDemo.h:34,60–63`); device `demo_programs.cu:373–395`; reset triggers `OptixDemoTOP.cpp:771` (camera/bg/sky/texmap change → `frameIndex=0`).
NEE samplers (the reusable light machinery): `sampleDirect` (emissive spheres, power CDF) `114–142`; `sampleSun` (analytic sun, soft shadow) `147–169`; `sampleEnv` (HDRI 2D-CDF importance) `172–199`; `sampleLights` (point/cone/distant) `205–247`. All fire only on `prd.isDiffuse` bounces (`334–341`).
**G-buffer / AOVs (the ReSTIR enabler):** color `image` `397`; albedo `398`; world normal `399`; **motion vectors `flow` `400` (computed `361–371`)**; spec albedo `405`; roughness `406`; linear view-Z `rrDepth` `407–409`; raw per-frame HDR `rrColor` `404`. Prev-frame camera + `validPrev`: `LaunchParamsDemo.h:51–56`, host `OptixDemoTOP.cpp:757–773`.
RNG: PCG32 `demo_programs.cu:20–22`, seed `270`. Materials/BSDF: triangle CH `415–494`, sphere CH `503–559`, miss/sky `496–501`.

**Where each feature splices in:**
- *Volumetrics:* between the `optixTrace` result and the surface BSDF, inside the bounce loop (`~302–347`). Reuse the NEE samplers from a medium point (phase replaces BRDF·cosine, no normal).
- *ReSTIR DI:* refactor `sampleDirect` into `evalPHat()` (everything except the occlusion ray) + `shadowVis()` (just the ray). Do reservoir DI at the **primary diffuse hit** first; leave deeper-bounce NEE inline initially.

---

## PART A — Volumetric fog & god rays

### A.1 Theory in one screen
Medium = absorption `σ_a`, scattering `σ_s`, extinction `σ_t=σ_a+σ_s`, single-scatter albedo `α=σ_s/σ_t`, phase function `p(ω,ω′)` (normalized). **Homogeneous transmittance is closed-form:** `T(s)=exp(−σ_t·s)` — *no ray march*. Heterogeneous (grid/noise) needs ray-march or delta/ratio tracking. The transfer equation along a camera ray to a surface at distance `s`:
```
L = T(0→s)·L_surface  +  ∫_0^s T(0→t)·σ_s·L_scat(p_t,ω) dt,   L_scat = ∫ p(ω,ω′)·L_i dω′
```

### A.2 Henyey–Greenstein phase (PBRT-v4 convention)
```
p_HG(cosθ) = (1/4π)(1−g²)/(1+g²+2g·cosθ)^{3/2},  cosθ = ω·ω′  (g>0 = forward scatter)
sample: s=(1−g²)/(1+g−2g·u0); cosθ = −(1/2g)(1+g²−s²);  φ=2π·u1   (isotropic if |g|<1e-3: cosθ=1−2u0)
```
Exact sampler ⇒ returned pdf == phase value. **Sign-convention trap:** other refs use `−2g cosθ` with θ vs the *incident* dir; pick one so forward scattering points at the sun. For god rays use `g≈0.6–0.9`.

### A.3 Two estimators
- **Option A — free-flight distance sampling (recommended primary).** Sample `t_s = −ln(1−u)/σ_t`. If `t_s<s`: real scatter at `p_{t_s}` → medium NEE + HG-sampled continuation; throughput `*= α` (the `T·σ_t` cancels). Else: proceed to surface unchanged. **Unbiased, minimal code, composes with `frameIndex` accumulation for free.**
- **Option B — equiangular ray-march god-ray mode (directed shafts).** March the primary ray; per step, shadow-ray toward the sun/spot (occluded segments = the dark gaps between shafts). Biased but cheap and low-noise *if* you use equiangular sampling.

### A.4 God rays = equiangular sampling (Kulla & Fajardo 2012)
The in-scatter toward a point/spot light has a `1/d²` spike → **unbounded variance** under uniform/exponential distance sampling. Equiangular sampling cancels it. With light `L`, ray `o+t·ω`, `δ=(L−o)·ω`, `D=‖(L−o)−δω‖`:
```
θ_a=atan2(a−δ,D), θ_b=atan2(b−δ,D)
t = δ + D·tan(lerp(θ_a,θ_b,u));  pdf(t)= D / ((θ_b−θ_a)(D²+(t−δ)²))   // ∝ 1/d²
```
MIS-combine equiangular with the free-flight pdf for robustness across near/far-light regimes. Guard `D≈0` (ray pointing at the light).

### A.5 Integration sketch (megakernel)
```c
optixTrace(...);  float s = hit ? hitT : LARGE;
if (fogEnabled && insideFogBounds(ray)) {
    float tS = -logf(1.0f - pcg(rng)) / sigma_t;
    if (tS < s) {                                   // scatter event in the fog
        float3 pT = ray.origin + tS*ray.dir;
        radiance += throughput * mediumNEE(pT, ray.dir, rng);  // phase×light×vis×Tr, NO normal/BRDF
        ray.dir = sampleHG(ray.dir, g, pcg(rng), pcg(rng));
        ray.origin = pT;  throughput *= albedo;  continue;
    } // else fall through to surface (transmittance implicit in the sampling prob)
}
// existing surface BSDF + NEE; surface shadow rays also *= exp(-sigma_t * d_light)
```
`mediumNEE` calls the **existing** `sampleSun/sampleEnv/sampleLights`, replacing BRDF·cosine with `p_HG(ω·ω_L)` and attenuating the light by `exp(−σ_t·d)`. **Don't double-count transmittance** (free-flight branch needs no explicit `exp`; analytic branch does — never both).

### A.6 Phased plan (volumetrics)
| Phase | Adds | New params | Cost | Look |
|---|---|---|---|---|
| 1 | Homogeneous fog, free-flight, isotropic, analytic surface attenuation | `sigma_a,sigma_s,fogEnabled` | ~free (thin fog) | haze, aerial perspective, light glow |
| 2 | HG anisotropy + equiangular god-ray march | `g,mode,maxMarchSteps,godrayLightIndex` | +M shadow rays/primary ray | crepuscular shafts, sun halo, dusty spot cones |
| 3 | Height fog / AABB bounds (height optical depth stays **closed-form**) | `aabbMin/Max,heightFalloff,heightRef` | negligible | ground fog banks, bounded fog rooms |
| 4 (note) | Heterogeneous density via **delta/ratio tracking** | density field | substantial | clouds, noise-driven fog |

### A.7 Perf
Cost = extra shadow rays per march step. Levers: equiangular+MIS (biggest), jittered march start, blue-noise on scatter-distance/light-pick, **N=1 sample/frame + accumulation**, and the **froxel half-res + upsample** alternative (Frostbite/Hillaire 2015) if per-pixel marching is too costly at 4K. DLSS-RR denoises the per-frame fog noise — render volumetrics at RR *input* res. Write sane volumetric G-buffer values at scatter points (depth/position, null normal, scatter color) so RR guides stay valid.

---

## PART B — ReSTIR DI + GI

### B.1 DI in one screen
**RIS:** draw `M` cheap candidates from a source pdf `p` (your power-CDF light pick), re-select one ∝ a cheap **target `p_hat`** = the **unshadowed** contribution `‖f_r·L_e·G‖` (everything in `sampleDirect` *except* the occlusion ray). **WRS** streams candidates in O(1):
```c
struct Reservoir { LightSample y; float w_sum, M, W; };   // y = {lightIndex,type,samplePos}
bool update(r, x_i, w_i, rng){ r.w_sum+=w_i; r.M+=1; if(rnd(rng) < w_i/r.w_sum){ r.y=x_i; return true;} return false; }
// per candidate w_i = p_hat(x_i)/p(x_i);  after streaming:  W = w_sum / (M · p_hat(y))
// final shade: L = f_r·L_e·G · V(x,y) · W    ← ONE deferred shadow ray
```
`E[W | y] = 1/p(y)` → unbiased, and `W` can stand in for `1/p` when this reservoir is fed into another (that's how reuse chains).

### B.2 Temporal reuse (reuses existing infra)
Reproject with the engine's existing `ppx,ppy` (`demo_programs.cu:360–371`) + `validPrev`; fetch prev reservoir like `taaPrevCol`. **Reject** on out-of-bounds / `dot(N,N_prev)<0.906` (~25°) / `|z−z_prev|>0.1·z` (use `rrDepth`). **M-cap** history to ~20× current M (lower to ~10× for fast-animating lights). Merge with the **full balance heuristic** (only 2 domains → trivial, lowest variance). This is the biggest variance win.

### B.3 Spatial reuse + the bias problem
k≈4 neighbors, radius ~25px, same normal/depth rejection. **Naive `1/M` merge is biased** (each neighbor sampled under its own `p_hat_j` → boundary **darkening**). Fixes, in order:
1. **`1/M` biased** — ship first for a correct-looking image / to validate plumbing.
2. **`1/Z` unbiased** — count neighbors that *could* have produced the chosen sample; removes darkening (costlier, needs visibility rechecks).
3. **Pairwise MIS** (Wyman & Panteleev 2021) — O(N) approximation of the balance heuristic, unbiased; the **production target** (up to ~7× lower lighting cost). Use full balance for temporal (N=2), pairwise for spatial.

### B.4 ReSTIR GI
Reused sample = a **secondary path vertex** + cached directionless outgoing radiance `L_o(x_s→x_v)`. Reconnecting neighbor `q`'s sample at pixel `r` requires the **reconnection Jacobian**:
```
|J| = (cosφ_s^r / cosφ_s^q) · (‖x_v^q − x_s‖² / ‖x_v^r − x_s‖²)   // multiply into the weight; +visibility recheck
```
Megakernel makes this awkward (no seam to persist secondary-vertex reservoirs). **Two options:**
- **A — wavefront refactor** (correct, invasive): split into G-buffer/initial-sample, temporal, spatial, final-shade launches.
- **B — GI-only post-pass (recommended):** megakernel emits the initial GI sample (write `x_s,n_s,L_o` to a G-buffer at the first indirect bounce); separate CUDA kernels do temporal+spatial GI resampling (one reconnection visibility `optixTrace` each) and add the result to `image`. Least disruptive; composes with the DI work. Clamp secondary-vertex roughness against fireflies.

### B.5 RTXDI — spec, not dependency
RTXDI is HLSL include-files for D3D12/Vulkan + NVIDIA `donut`. **Don't adopt** — it would fight TD's CUDA/Vulkan interop (our documented deadlock class), and its headline many-light tiling/ReGIR is unneeded for our handful of lights + 1 HDRI. Mirror its **pass decomposition, M-cap/rejection thresholds, and the `1/M`→pairwise-MIS progression** in CUDA/OptiX. Reference: the SDK docs + *A Gentle Introduction to ReSTIR* (SIGGRAPH 2023 course).

### B.6 Phased plan (ReSTIR)
| Phase | Adds | Benefit | Risk |
|---|---|---|---|
| 0 | Reservoir buffers (ping-pong) + prev N/Z G-buffer; refactor `sampleDirect`→`evalPHat`+`shadowVis` | none (validate identical at M=1) | pure refactor |
| 1 | DI initial RIS at primary hit (M≈16–32, one deferred shadow ray) | de-risk reservoir math | wrong `W` normalization → brightness off |
| 2 | **Temporal reuse** (reproject, M-cap 20×, balance heuristic) | **large** noise drop | ghosting on moving lights (tune M-cap) |
| 3 | Spatial reuse (biased `1/M`) | more interior noise reduction | boundary darkening (expected) |
| 4 | Spatial **pairwise MIS** (+ optional fused spatiotemporal) | removes darkening; production target | MIS bookkeeping — validate energy vs Phase-1 ground truth |
| 5 | **ReSTIR GI** post-pass (Option B) | large indirect noise drop | Jacobian errors, glossy fireflies |

### B.7 Reservoir VRAM (pack to ~32 B/reservoir, ×2 buffers)
720p ≈ **59 MB**; 1080p ≈ **133 MB** (×3 ≈ 200 MB); 4K ≈ **531 MB** (×3 ≈ 800 MB). GI reservoirs ~2× larger → DI+GI at 4K can approach **~2 GB**. **Render ReSTIR at the DLSS-RR input resolution** (not output 4K) — the single biggest VRAM saver, and it aligns with the existing RR upscale pipeline.

### B.8 Pipeline pitfalls (megakernel + accumulation + DLSS-RR)
1. **Triple temporal filter** — ReSTIR-temporal *is* a temporal accumulator; don't also run `accum`/TAA averaging on the ReSTIR term and then RR. Feed RR the **per-frame** ReSTIR output (`rrColor`-style raw). 2. **Consistent MVs** — the `flow` driving reservoir reprojection must equal the MVs fed to RR. 3. **Jitter** — snap reservoir reprojection to the unjittered pixel center. 4. **Fireflies** — ReSTIR DI fireflies are a *direct* phenomenon; the existing firefly clamp only covers indirect — add a clamp on the ReSTIR direct term. 5. **Divergence** — keep `M_init≤32` and confine DI to the primary hit to limit register pressure (the deeper argument for a wavefront refactor long-term).

---

## Sources
**Volumetrics:** PBRT-v4 (Equation of Transfer; Transmittance; Volume Scattering Integrators / delta+ratio tracking; Phase Functions); Kulla & Fajardo 2012 (equiangular sampling); Hillaire 2015 *Physically Based and Unified Volumetric Rendering in Frostbite* (froxels).
**ReSTIR:** Bitterli et al. 2020 (ReSTIR DI); Wyman & Panteleev 2021 (pairwise MIS, light tiles, RTXDI); Ouyang et al. 2021 (ReSTIR GI); Lin et al. 2022 (GRIS); Wyman et al. 2023 (*A Gentle Introduction to ReSTIR* course); NVIDIA-RTX/RTXDI SDK docs (Integration.md, RestirGI.md).
```
Full per-topic notes with derivations live in the session research agents' output (2026-06-25).
```
