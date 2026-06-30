# Research: PBR Material Maps (normal / roughness / metallic / AO / emission)

**Status:** research / design — nothing implemented. **Date:** 2026-06-25.
**Target:** `OptixDemoTOP` — OptiX 9 megakernel path tracer (`demo_programs.cu` / `LaunchParamsDemo.h` / `OptixDemoTOP.cpp`). Triangle path only (`__closesthit__tri`).

Today the engine supports **only a base-color map**. This note designs adding **normal, roughness, metallic, AO, and emission** maps (incl. the requested *textured-emissive* material), grounded in the exact current code, with a phased, risk-ordered plan.

---

## 0. How the current material/texture system works (the foundation everything reuses)

**Per-vertex geometry (no-index triangle soup; vertex `3*pid+{0,1,2}` = triangle `pid`; all `float4`):**

| Attr | Array | Packing |
|---|---|---|
| Position | `triVerts` | xyz |
| Color | `triCd` | **xyz = base/emission color, `.w` = matID** |
| Material | `triMat` | **x=type(+10 ⇒ smooth-normals flag), y=roughness, z=ior, w=emitStrength** |
| Normal | `triN` | xyz smooth normal (else flat geometric) |
| UV | `triUV` | xy ("Tex"); z (W) ignored |

**No tangents exist** — the single biggest gap for normal mapping. `type = rawtype % 10` (0 lambert, 1 metal, 2 glass, 3 emitter); `rawtype ≥ 10` enables smooth normals.

**Base-color map, end to end:**
- **Host:** `matbake` Execute DAT → `texStack` layout TOP → `texToPop` → render `Texturepop` POP → **CPU-bounce** readback (`POP_BufferLocation::CPU`, the Vulkan→CUDA deadlock fix) → HtoD into `myTexStage` → band-copied into a **layered** `cudaArray` `myBaseColorLayered` (`cudaMalloc3DArray(..., cudaArrayLayered)`), **one square layer per matID**, re-copied **every cook** so textures animate live (`OptixDemoTOP.cpp:704-741`). Layer dims inferred assuming **square layers**: `per = totalPts/nMat`, `s = round(sqrt(per))`, verified `s*s*nMat == totalPts`.
- **Device:** `tex2DLayered<float4>(baseColorTex, uu, vv, matID)` (UV path) or `triplanar()` (3 axis lookups blended by `|N|²`, tiled at `projScale`) — chosen by `texMode` (0 auto / 1 force-UV / 2 force-projection) and a `haveUV` degeneracy test. The map **MODULATES** the vertex color: `color = mul(color, tx)` — for a pure texture, vertex Cd must be white.

**BSDF** (`demo_programs.cu:654-675`): a discrete switch. emitter (`emitted=color*estr`, terminal); metal (mirror reflect fuzzed by roughness); glass (Fresnel/refract by ior); lambert (cosine hemisphere, `isDiffuse=1`). **NEE hardcodes the lambert lobe `f=albedo/π` and gates on `isDiffuse`** — this is the key constraint on any BSDF change (§C). AOVs `hitAlbedo/hitNormal/hitSpecAlb/hitRough` feed the OptiX denoiser + DLSS-RR and **must stay sane** through any change.

**Emissive geometry already has an area-light list** (`emitTriIdx`, host-scanned from `triMat`, consumed by `mediumEmitNEE` for fog). Surfaces get emission via the throughput `addEmit` term (no surface area-light NEE yet).

---

## A. Architecture — multiple per-map-type layered stacks

**Generalize the single `myBaseColorLayered` into a small set of parallel layered arrays, all keyed by the same matID, all sharing the existing host pipeline** (matbake → texStack → texToPop → POP → CPU-bounce → band-copy):

```
baseColorTex   RGBA, sRGB→linear on bounce            [existing]
normalTex      RGB tangent-space, LINEAR, 0.5 = flat   new
ormTex         R=AO, G=roughness, B=metallic (glTF ORM) new  ← one array covers 3 maps
emissionTex    RGB linear * emitStrength               new
```

Packing **AO+Roughness+Metallic into one ORM array** is the key VRAM/bandwidth win — glTF-native, one POP + one band-copy + one sample instead of three. Normal and emission stay separate (different host processing / often different res / optional).

**LaunchParams additions** (mirror the base-color trio): `normalTex/useNormalTex/normalStrength/normalFlipY`, `ormTex/useORM/aoStrength`, `emissionTex/useEmissionTex/emissionScale`. `numMaterials/projScale/texMode` are **shared** (same UVs, same triplanar fallback).

**Host:** factor the band-copy block (`OptixDemoTOP.cpp:712-741`) into a `buildLayeredStack(popDev, totalPts, nMat, arr, tex, curW, curH, curMat, addressMode)` helper (cudaMalloc3DArray on dims-change + per-layer cudaMemcpy3D every cook). Each stack reads its own POP (`getParPOP("NormalPop")`, `"OrmPop"`, `"EmissionPop"`), CPU-bounces it, calls the helper. Recommend **shared layer dims** (one matbake driving all stacks) for v1.

**Device:** one shared `sampleMap(tex, use, useUV, uu, vv, P, Nf, projScale, matID, dflt)` helper that respects `texMode`/triplanar identically for every map.

**VRAM / perf:** per stack = `numMaterials × layerH² × 16 B`. 16 mats @ 1024² = 256 MB/stack → 4 stacks ≈ **1 GB**. Mitigations, in order: **(1) fp16 (`__half4`) → halves VRAM** (normal/ORM/emission tolerate it; biggest win); (2) **per-map resolution caps** (ORM/normal rarely need base-color res — independent POP layouts make this free; extend the adaptive license-cap logic across all stacks); (3) **lazy stacks** (only allocate if the POP is wired — a base-color-only scene pays zero); (4) mips later. Sampling cost is +1 `tex2DLayered` per active map (negligible on Ada vs traversal); triplanar fallback is 3× per map (the expensive corner).

**⚠ Color-space discipline (most common PBR bug):** baseColor + emission are **color data** → must be **linear** when sampled (existing bounce already is). normal + ORM are **non-color data** → must **NOT** be sRGB-decoded. Set those source TOPs / Movie File In to *Input Color Space = Linear*. A normal map decoded as sRGB gives too-strong bumps; roughness decoded as sRGB gives wrong glossiness.

---

## B. Normal map — tangent-space → world

**The tangent problem:** the closest-hit has **no screen-space derivatives** (raygen-only) and the soup has **no tangents**. Three options:

| Option | Cost | Quality | Verdict |
|---|---|---|---|
| **(1) Per-triangle tangent from UV gradients** (in CH) | cheap (2×2 solve) | correct per-triangle; faceted across edges but fine for normal *detail* | **v1** |
| (2) Per-vertex tangents in the soup (new `triTan` array + barycentric) | +1 vertex stream + matbake work | smooth, matches DCC/glTF | Phase-2 upgrade |
| (3) Screen-derivative tangents | — | impossible in CH | rejected |

**Option (1) needs zero new vertex data** (all 3 verts + UVs are already fetched in CH).

**Math** (verts `v0,v1,v2`, UVs `uv0,uv1,uv2`):
```
e1=v1-v0  e2=v2-v0 ;  du1=uv1-uv0  du2=uv2-uv0
r = 1/(du1.x*du2.y - du1.y*du2.x)            // guard |det|>eps
T = (e1*du2.y - e2*du1.y)*r                  // dP/du
B = (e2*du1.x - e1*du2.x)*r                  // dP/dv
T = normalize(T - Nf*dot(Nf,T))              // Gram-Schmidt against shading normal
sign = dot(cross(Nf,T), B) < 0 ? -1 : 1      // handedness (mirrored UV islands)
B = cross(Nf,T) * sign                       // TBN columns (T,B,Nf) = tangent→world
```
**Decode + perturb:** `ns = sample.rgb*2-1; ns.xy *= normalStrength; ns.y *= flipY?-1:1; ns = normalize(ns); Nshade = normalize(T*ns.x + B*ns.y + Nf*ns.z)`. Expose a **`flipY` toggle** (OpenGL +Y vs DirectX −Y green channel — TD's texToPop `vv` origin sets the default). Guard `dot(rd, Nshade) > 0 → revert to Nf` (avoids black back-face artifacts).

**AOV:** write the **perturbed** `Nshade` to `prd->hitNormal` (so the denoiser/DLSS-RR see the detail and don't smooth it away) — but **keep geometric `Nf` for ray offsets + glass refraction sidedness** (avoids self-shadow acne at grazing angles). Triplanar normal blending (RNM/whiteout of 3 projected normals) is deferred — under projection mode, skip normal mapping in v1.

---

## C. Roughness + metallic — bolt-on first, unified GGX second

**Two-phase, because NEE is the constraint.** Every NEE function (`sampleSun/sampleEnv/sampleLights/sampleDirect`) hardcodes `f=albedo/π` and gates on `isDiffuse`. A correct metallic-roughness BSDF needs NEE to evaluate the **GGX lobe + MIS** for glossy hits, or glossy materials get *no* direct lighting. So:

**Phase C-a — bolt-on (~½ day, near-zero risk).** Keep the discrete switch; use ORM to drive existing params:
- `roughness (ORM.g)` → multiplies `mm.y` (metal fuzz).
- `metallic (ORM.b)` → **picks the lobe per-texel**, stochastically to avoid a hard seam: `if(rnd(seed) < metallic) {metal lobe, albedo=baseColor → F0≈baseColor} else {lambert}` — unbiased, converges to the glTF lerp `(1-metallic)·dielectric + metallic·metal`. Metal stays `isDiffuse=0`, lambert stays `isDiffuse=1` → **existing NEE intact**. Most visual value per hour.
- Set `hitSpecAlb=F0`, `hitRough=roughness` → **better DLSS-RR guides for free**.

**Phase C-b — unified GGX metallic-roughness BSDF (~2–3 days, medium-high risk).** glTF: `F0 = lerp(0.04, baseColor, metallic)`, `diffuse = baseColor*(1-metallic)`. GGX/Trowbridge-Reitz `D·G·F/(4·n·l·n·v)`; **sample with VNDF (Heitz 2018)** in a tangent frame (the weight `f·cos/pdf = F·G2/G1` is numerically clean). Stochastic lobe pick (diffuse vs specular by luminance weights) keeps the megakernel single-bounce. **The real work:** generalize the four NEE functions to a `bsdfEval(wo,wi,mat)` and add **MIS (power heuristic)** between light- and BSDF-sampling for glossy lobes; relax `isDiffuse` to "non-delta lobe." Gate behind a toggle; validate with a white-furnace test and that fog/lights/env match Phase-1 at rough≈1. Leave glass as-is (rough-glass transmission is Phase 5).

---

## D. AO map + ORM packing

- **Packing:** glTF stores **Occlusion(R), Roughness(G), Metallic(B)** in one texture. One layered `ormTex` covers all three → one POP, one band-copy, one sample.
- **AO nuance (important):** a path tracer computes occlusion *for real* via GI — applying a baked AO map at full strength **double-darkens**. Correct (glTF): AO modulates **only the indirect/ambient + environment-NEE diffuse term**, NOT analytic direct lights and NOT specular. v1: multiply AO into the **env IBL contribution only** (`sampleEnv` return + passive `skyDome` ambient-on-miss); leave `sampleSun`/`sampleLights` un-AO'd. Expose `aoStrength` (default ~0.5 since GI already does most of the work).

---

## E. Emission map / textured-emissive (the requested feature)

A **base-color-mapped material that ALSO emits** per-texel. Today emission is per-triangle flat (`color*mm.w`) and only the *medium* NEEs it.

**E.1 Surface emission (easy, do first):** in `__closesthit__tri`, sample the emission map and add it whenever `addEmit`:
```c
float3 emis = useEmissionTex ? sampleMap(emissionTex,...).rgb * color/*tint*/ * emissionScale * estr : 0;
prd->emitted = add(prd->emitted, emis);   // a lambert/metal material now ALSO emits per-texel
```
Works immediately (raygen already adds `prd->emitted` along the throughput). A textured emitter that **still scatters** must NOT set `absorbed=1` (distinguishing it from a pure `type=3` light).

**E.2 Textured area-light NEE (higher quality), three tiers:**
1. **Average-emission (RECOMMENDED v1):** precompute a per-material (per-layer) **mean emission color** by reduction over the emission layer (one mean per matID, on texture change). NEE uses `Le = matEmitAvg[matID]*estr`; per-texel detail still appears via the throughput term (E.1) on direct view + specular. Low effort, reuses `emitTriIdx`.
2. **Per-triangle average** (integrate over the triangle's UV footprint) — medium effort, better for high-contrast emission.
3. **Per-texel importance sampling (full):** build a 2D-CDF per emissive layer — **you already have this exact machinery in `sampleEnv`/`buildEnvCDF`!** Lowest noise for sparse bright emitters (LED/neon). High effort; defer.

**Also:** switch `mediumEmitNEE`'s **uniform** triangle pick to a **power-weighted CDF** (`power = luminance(matEmitAvg)·area`, mirroring the existing `lightCDF` pick) to kill fireflies when one small material is very bright. Extend the host emissive scan so a triangle is "emissive for NEE" if `type==3` **OR** `useEmissionTex && matEmitAvg[matID]` is non-black.

---

## F. Phased plan (most-visual-value-first, risk-ordered)

| Phase | What | Effort | Risk | Touches NEE? |
|---|---|---|---|---|
| **1 — ORM bolt-on** (roughness+metallic+AO) | host ORM stack + `buildLayeredStack` refactor; device stochastic lobe-pick + roughness/AO; `hitSpecAlb=F0`, `hitRough` | ~1 day | low (color-space) | no |
| **2 — Normal maps** (per-triangle tangent) | host normal stack; device TBN-from-UV + perturb `Nshade`; write perturbed normal AOV | ~1 day | low-med (Y-flip/handedness/acne) | no |
| **3 — Emission maps** (surface glow + avg-emission NEE) | host emission stack + `matEmitAvg` reduction; device `prd->emitted` add; `mediumEmitNEE` reads avg; power-weighted CDF | ~1 day | low (double-count handled by `addEmit`) | minor |
| **4 — Unified GGX BSDF + MIS** | device GGX/VNDF; generalize NEE to `bsdfEval`; MIS; relax `isDiffuse` | ~2–3 days | **med-high** (NEE/MIS core that DLSS-RR + fog + lights depend on) — gate behind a toggle | yes |
| **5 — deferred** | per-texel emission CDF, rough-glass GGX transmission, normal mips, triplanar normal (RNM) | high | — | — |

**Rationale:** ORM (1) + normal (2) deliver the biggest perceptual jump (painted metal/roughness + surface detail) for ~2 days at low risk **without touching NEE**. Emission (3) is self-contained and reuses existing emitter machinery. The risky unified-BSDF rewrite (4) is last and gated, so the renderer stays shippable throughout.

---

## Sources
- glTF metallic-roughness BRDF (F0=lerp(0.04,baseColor,metallic), diffuse=baseColor·(1−metallic)): [glTF 2.0 Spec (Khronos)](https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html), [KHR_materials_specular](https://github.com/KhronosGroup/glTF/blob/main/extensions/2.0/Khronos/KHR_materials_specular/README.md), [PBR notes — Tarun Ramaswamy](https://rtarun9.github.io/blogs/physically_based_rendering/).
- Tangent from UV gradients + Gram-Schmidt: [The Geometry Behind Normal Maps](https://www.shlom.dev/articles/geometry-behind-normal-maps/), [Mikkelsen tangent critique (IrrlichtBAW wiki)](https://github.com/buildaworldnet/IrrlichtBAW/wiki/How-to-Normal-Detail-Bump-Derivative-Map,-why-Mikkelsen-is-slightly-wrong-and-why-you-should-give-up-on-calculating-per-vertex-tangents).
- GGX VNDF sampling: [Heitz, JCGT 2018](https://jcgt.org/published/0007/04/01/paper.pdf), [Dupuy & Benyoub, spherical caps, arXiv:2306.05044](https://arxiv.org/pdf/2306.05044), [Bounded VNDF Sampling (ACM)](https://dl.acm.org/doi/10.1145/3651291).
- Textured/energy-weighted area-light IS: [Iray Light Transport (arXiv:1705.01263)](https://arxiv.org/pdf/1705.01263), [Path Tracer in Unreal Engine](https://dev.epicgames.com/documentation/unreal-engine/path-tracer-in-unreal-engine).

Engine targets: `OptixDemoTOP.cpp` (base-color stack 704-741; emissive scan 664-686; LaunchParams fill 841-854), `demo_programs.cu` (`__closesthit__tri` 606-676), `LaunchParamsDemo.h` (78-93). Related: [docs/RESEARCH-volumetrics-and-restir.md], the DLSS-RR G-buffer (don't break the AOVs).
