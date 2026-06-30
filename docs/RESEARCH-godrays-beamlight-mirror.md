# God-Ray Crispiness, Cone/Beam Light, and Mirror Reflection — Integrated Design Document

**Engine:** `OptixDemoTOP` (real-time RTX path tracer, TouchDesigner C++ TOP)
**Status:** Research / Design — **NO implementation yet**
**Date:** _2026-06-26 (placeholder)_
**Branch context:** `pt-render-comp`

---

## Goal

Three features that reinforce one another into a single visual: a **tight cone/beam light** is the natural source of a god-ray shaft; a **crispiness slider** sharpens that shaft from milky haze to a clean blade of light; a **perfect mirror** redirects the beam and its shaft. Together: *cone light + crisp fog = clean shaft; mirror = redirected shaft.*

> **Headline finding from the engine-grounding pass: two of the three features already exist in shipped code.** The cone/spot light (`PTLight.type==1`, `cosInner`/`cosOuter`) is fully wired end-to-end — struct, parse, surface NEE, fog NEE, and TD Light COMP serialization. The perfect mirror also already exists (`type==1` metal with `roughness==0` → exact `reflect3`). The **only genuinely new engine work is the fog crispiness control** (Section A). Sections B and C are therefore framed as *verify + tune + polish*, not *build from scratch* — this is a deliberate de-risking of the plan.

---

## 0. Current engine grounding

Files (verified against source this session):
- **Device:** `phase2/OptixDemoTOP/demo_programs.cu`
- **Params struct:** `phase2/OptixDemoTOP/LaunchParamsDemo.h`
- **Host:** `phase2/OptixDemoTOP/OptixDemoTOP.cpp`
- **TD serializer (Execute DAT):** `pathtttracer.toe.dir/project1/optixDemo/PT_Render/lightSerialize.text` (lives inside the `.toe`)

### 0.1 The path loop (`__raygen__rg`, `demo_programs.cu:452-530`)
- Per bounce, traces the camera/continuation ray, then runs **fog free-flight scattering** (`:478-499`) *ungated by `isDiffuse`*, then surface NEE *gated by `prd.isDiffuse`* (`:516-523`).
- Flags that matter for these features: `lastSpec` (sees `sunDisk`+`skyDome` on miss), `addEmit` (accumulate emissive geometry on the continuation), `fromFogScatter` (double-count guard for env-on-miss).

### 0.2 Fog system (`inscatterFog`, `demo_programs.cu:336-395`; free-flight `:478-499`)
- Free-flight distance `ts = -log(1-u)/fogDensity` (`:480`); if `ts < hitDist`, scatter at `pT` (`:482`), add `L += thr*fogColor*inscatterFog(pT,rd,seed)` (`:484-485`).
- `inscatterFog` does phase-weighted medium NEE with **no surface normal / no BRDF**:
  - **Sun shaft** — `:351`: `sum += E * phaseHG(dot(wi,rd), g)`. *This single line is the sun god-ray.* Soft-shadow cone jitter driven by `sunAngle` (`:343-346`).
  - **PTLights** — `:355-371`: distant vs point/cone; **cone mask** `:365` `smooth01(cosOuter,cosInner,cd)`; finite-light fog attenuation `segTr=exp(-fogDensity*d)` (`:364`); `atten=master*50/(d²+r²)` (`:364`). The cone-mask zero-region *is the shaft edge*.
  - **Env in-scatter** — `:376-391`, scaled by `fogSkyStr` (importance-sampled `mediumEnvNEE` or a 2-sample uniform-sphere fallback).
  - **Emitter glow** — `mediumEmitNEE` `:393`.
- All delta-light in-scatter terms use the *same* `g = params.fogG` (`:338`). **There is no decoupled shaft-sharpness control today** — sharpness is an emergent byproduct of `fogG` + the spot's `conedelta`.
- **Constraint — firefly clamp:** fog in-scatter is added to `L` *before* the `Ldir` snapshot and is treated as **direct** light, so it is **not** covered by the indirect-only `fireflyMax` clamp (`:533-535`). Sharper shafts raise variance with no firefly safety net → lean on accumulation/DLSS-RR, not clamping.

### 0.3 Light system (`PTLight`, `LaunchParamsDemo.h:20-29`; `sampleLights`, `demo_programs.cu:205-247`)
- `PTLight = { pos, dir, radiance, int type (0 point/1 cone/2 distant), radius, cosInner, cosOuter }` — **13 floats** when serialized.
- `sampleLights` (surface NEE, `pdf=1`): cone block `:231-236` mirrors the fog cone block; contributes `albedo*radiance*(1/π)*ndl*atten`.
- **Key constraint — NEE hardcodes Lambert:** every NEE function evaluates `f = albedo/π` (`:141, :168, :198, :244`). NEE has *no concept of a specular lobe*, which is exactly why specular surfaces must skip it (Section C).
- **Serialization contract** (host parse `OptixDemoTOP.cpp:763-787`): `"N  px py pz dx dy dz cr cg cb type radius cosInner cosOuter ..."` — **13 floats per light**, `strtof` loop, filled into `PTLight` (`type=f[9]`, `radius=f[10]`, `cosInner=f[11]`, `cosOuter=f[12]`). The TD serializer (`lightSerialize.text`) builds this per Light COMP from native `lighttype`/`coneangle`/`conedelta` (lines 11/14/17). **This is a hand-synced positional contract** — any new field must be appended as float #14+ in *both* the DAT and the parser.

### 0.4 Material / BSDF (`__closesthit__tri`, `demo_programs.cu:599-677`)
- Discrete switch on `triMat = float4(type, roughness, ior, emitStrength)`; `rawtype>=10` ⇒ smooth normals, `type = rawtype%10` (`:616`); base-color layer keyed by `matID = triCd.w`.
- **`type==1` metal/mirror** (`:658-662`): `r=reflect3(rd,Nf)`; `if(rough>0) fuzz` else **exact mirror**; `attenuation=color` (flat reflectivity, **no Fresnel**); `hitSpecAlb=color`; `hitRough=fmaxf(rough,0.02f)` (**floored at 0.02**); `isDiffuse=0`.
- `type==3` emitter terminal; `type==2` glass Fresnel; else Lambert (`isDiffuse=1`).
- **`isDiffuse` gating** (`:516-523`): specular surfaces carry light *only* through the bounce ray — exactly what a mirror needs; NEE is correctly off for them.

### 0.5 AOVs → denoisers
- PRD guides set per material: `hitAlbedo, hitNormal, hitPos, hitSpecAlb, hitRough`.
- Captured **once at primary hit** `sp==0 && b==0` (`:460-463`); miss → `aoAlb=skyColor, aoNrm=0, aoHit=0`. Specular reflection hit-distance captured at `sp==0 && b==1` (`:465-468`, only when the *primary* hit was specular).
- **OptiX AI denoiser** guides: `albedo`, `normal` (+ `flow`). **DLSS-RR G-buffer** (`:586-595`): `rrColor` (raw per-frame), `rrSpecAlbedo`, `rrRoughness`, `rrDepth`, `rrMotion`, `rrHitDist`. **RR is the project default denoiser.**
- **Critical constraint:** fog in-scatter is added *after* AOV capture, so a god-ray pixel carries the AOVs of the *surface/sky behind the fog* (`aoNrm=0`, `aoAlb=skyColor`, `roughness=1`) — a maximally "blur-me" guide. This is the root cause addressed in Section D.

---

## A. Fog "Crispiness" slider

### A.1 The physics — what makes a shaft crisp vs soft
A crepuscular ray is **in-scattered radiance shaped by a shadow.** It reads as *crisp* when three things hold: (1) in-scatter is **directional** (concentrated toward the light, so the shaft is brighter than its surroundings); (2) the **shadow boundary cutting the shaft is hard** (thin penumbra); (3) nothing **re-blurs** the structure afterward (no multi-scatter glow, no ambient fill, no low-contrast tone curve). Crispiness is a **high-spatial-frequency, high-contrast** property; softness is a low-frequency milky glow. The four physical knobs that move along that axis, and where they live in this engine:

| Knob | Engine field | Effect on crispiness | Notes |
|---|---|---|---|
| HG anisotropy `g` | `fogG` (`:338`) | **Primary lever.** g=0 isotropic → flat milky glow; g→1 forward lobe → tight bright streak. | Practical ceiling ~0.90 (above it the env fallback at `:380-388` — only 2 samples × `phaseHG*0.5` — gets noisy/biased). |
| Source angular size / shadow hardness | `sunAngle` (`:343-346`); point-light `Lt.radius` (`:362`) | Smaller source = harder shadow edge = razor shaft + narrower halo. | `sunAngle→0` = hardest; but floor at ~0.25–0.30° (true solar disk) so edges anti-alias under accumulation instead of crawling. |
| Single- vs multi-scatter | `fogMaxScatter` (`:492-493`), `fogSingleScatter` (`:491`) | Multi-scatter is the great softener: it fills shadowed gaps and lowers contrast. ~1 ≈ crisp single-scatter, ~8/uncapped = soft. | `fogMaxScatter` is the *continuous* lever; keep `fogSingleScatter` as a hard perf toggle. |
| Tonal contrast | `fogSkyStr` (`:376-391`); **NEW** `fogContrast` | `fogSkyStr` adds an ambient floor that lifts shadowed air and washes out contrast; lower it to deepen the blacks between shafts. A contrast curve on `ins` steepens the lit/shadowed transition. | Free (no added noise). Don't drive `fogSkyStr` to 0 — HDRI scenes lose fog color. |

**Density (`fogDensity`) is deliberately excluded.** Higher density brightens the screen but pushes the medium into the soft multi-scatter regime, so coupling it to crispiness would make high-c look *worse* in thick fog. It stays the artist's "how thick is the air" dial.

### A.2 What ONE 0..1 'Crispiness' slider should drive
Drive **four internal values from `c`** + add **one new contrast term**. Leave density and color to the artist.

```
fogG          = lerp(0.40, 0.90, c)        // milky → tight forward halo (hard-cap 0.90)
sunAngle(deg) = lerp(3.0,  0.30, c)        // 3° penumbra → ~solar disk (floor 0.25–0.30°)
fogMaxScatter = round(lerp(8, 1, c))       // full glow → single-scatter; optionally fogSingleScatter=1 when c>0.85
fogSkyStr     = lerp(0.80, 0.15, c)        // strong ambient fill → minimal (floor 0.15, never 0)
fogContrast   = lerp(1.0,  1.7, c)         // NEW: contrast curve on the in-scatter term
```
Each knob moves one *independent* factor (directional in-scatter, hard shadow, no re-blur, high contrast), so the macro feels monotonic and physical across its whole range instead of saturating early.

**Optional 5th:** scale authored point-light `Lt.radius` by `(1 − 0.8*c)` so finite-light shafts sharpen on the same dial.

### A.3 Exact engine hooks + math

**Decoupled shaft anisotropy (recommended primary mechanism).** Rather than only animating the global `fogG`, add a *separate* shaft-only `g` so the bulk haze stays soft while shafts tighten:
- **New field** in `LaunchParamsDemo.h` after `fogRRStart` (`:152`): `float fogShaftG;` (optionally `float fogShaftSharp;`).
- In `inscatterFog`, use `params.fogShaftG` **only** at the **sun line `:351`** and the **PTLight line `:371`**; leave the **continuation `sampleHGdir` `:494`**, `mediumEnvNEE` `:378`, and the uniform-sky fallback `:388` on `params.fogG`. Higher `fogShaftG` (~0.7–0.95) tightens sun/spot shafts without making the bulk fog forward-peaked.
- The single-`Crispiness`-slider mapping (A.2) and the decoupled `fogShaftG` are *two compatible options*; see Open Questions for which to ship. The cleanest hybrid is: the macro drives `fogShaftG` (shaft tightness) + `sunAngle` + `fogMaxScatter` + `fogSkyStr` + `fogContrast`, and leaves the global `fogG` to the artist as "bulk haze softness."

**Contrast curve (new, free, no added noise).** Add LaunchParams float `fogContrast`; apply about a pivot `p ≈ 0.18` to the in-scatter term at `:485` (or inside `inscatterFog` before returning):
```
ins = p * powf(ins/p, k);   // k = fogContrast; luminance-preserving preferred over per-channel
```
Per-channel exponentiation can shift hue/clip — prefer a luminance-preserving contrast or keep `k ≤ 1.7` and clamp.

**Optional extra "snap":** `powf(phaseHG(dot(wi,rd), gShaft), fogShaftSharp)` at `:351`/`:371` — art-directed, breaks energy normalization (acceptable for shafts).

### A.4 Host wiring (`OptixDemoTOP.cpp`)
- Read new params near the fog reads (`:502-510`); set `lp.fogShaftG` / `lp.fogContrast` near `:892-894`.
- **Append params** on page "Fog" after the last fog param (`Fogrrstart` at `:1056`): `Crispiness` (Float 0..1), a `Crispdrives` toggle, plus the advanced fields (`Fogshaftg`, `Fogcontrast`, and exposed `Foganisotropy`/`Sunangle`/`Fogmaxscatter`/`Fogskystr`).
- **Critical:** add every new term to the `fogSig` accumulator-reset signature (`:823-827`) — otherwise live edits won't restart progressive accumulation and will appear to do nothing until the camera moves.

### A.5 UI: macro + advanced (matches the project's wrapper ethos)
Add `Crispiness` + a `Crispdrives` toggle on the PT_Render Fog tab. When `Crispdrives` is **on**, the host derives the four/five internal values from `c` and the advanced knobs are grayed via `enableExpr` (the existing 89-rule smart-graying pattern — gray by `menuName`, not label). When **off**, the advanced knobs are live for full manual control. This gives a beginner one slider and an expert full reach, with **no hidden state fights** (the host only overrides fields the macro owns).

---

## B. Cone / Beam / Spot light

### B.1 Status: already built end-to-end — this section is *finish + verify*
The model: cone/spot/beam are the **same emitter** — a point-or-disk source + an **angular mask** (`cosInner`/`cosOuter`) + **distance attenuation**. Spot = point apex + wide spread; tight spot = point + narrow; **beam** = disk apex + narrow; collimated cylinder = disk + ~0 spread. The engine already implements the point/disk-sphere source, the angular mask, and the `1/d²` attenuation:
- `PTLight` (`LaunchParamsDemo.h:20-29`) already carries `type`, `radius`, `cosInner`, `cosOuter`.
- Surface NEE cone block `sampleLights` `:231-236`; fog NEE cone block `inscatterFog` `:365`. *Both* apply the mask — so the shaft already appears.
- Serialized from native Light COMP `lighttype`/`coneangle`/`conedelta` via `lightSerialize.text` lines 11/14/17 → host parse `:777`.

**Why the cone carves the shaft:** a distant/point light has no angular mask, so every visible scatter point in-scatters → uniform haze, **no shaft**. A cone multiplies in-scatter by `cone = smooth01(cosOuter, cosInner, dot(-wi, dir))`; outside the outer cone `cone==0` → dark; the transition surface (the cone boundary) **is the visible shaft edge.** Already correct and working at `:365`.

### B.2 The four gaps vs a full TD-native + PBR spotlight (the actual work)
1. **`coneroll` ignored** — falloff is a hardcoded `smooth01`; TD's `coneroll` (1–10, gentle↔sudden edge) is not honored.
2. **No finite range** — the `1/d²` in-scatter tail never ends, so the shaft has no visible tip and far/empty scatter points waste shadow rays.
3. **No `startRadius`** — can't author a true beam (finite-disk emitter).
4. **Half-angle convention unverified** — the cos conversion lives in the Execute DAT (inside the `.toe`, not in the repo). Recommended convention (treat `coneangle` as a **full apex angle**, matching Ragan's `/2`):
   ```
   cosInner = cos(radians(coneangle * 0.5))
   cosOuter = cos(radians((coneangle + conedelta) * 0.5))
   ```
   **Must be A/B'd against a native TD Render TOP cone light** — if the beam comes out 2× wide/narrow, drop the `/2`. Keep inner and outer in the *same* convention.

### B.3 PTLight struct extension (only if pursuing the full beam model)
Extend `PTLight` by three floats (13→16-float record):
```cpp
struct PTLight {
    float3 pos; float3 dir; float3 radiance;
    int    type;        // 0 point, 1 cone, 2 distant
    float  radius;      // soft-shadow / source radius
    float  cosInner;    // cos(inner half-angle)
    float  cosOuter;    // cos(outer half-angle)
    float  coneRoll;    // NEW: TD coneroll [1..10] → falloff exponent (≈1 = current smoothstep)
    float  range;       // NEW: attenuation end (world units); 0 = infinite
    float  startRadius; // NEW: beam emitter-disk radius; 0 = point apex (classic spot)
};
```

**Cone falloff honoring `coneRoll`** — replace the two `smooth01` calls (`:233`, `:365`) with a shared `coneMask`:
```cpp
__device__ float coneMask(const PTLight& L, float cd){
    float t = __saturatef((cd - L.cosOuter) / (L.cosInner - L.cosOuter + 1e-8f));
    t = t*t*(3.0f - 2.0f*t);                         // base smoothstep (keeps current look at coneRoll≈1)
    float e = (L.coneRoll > 1e-3f) ? L.coneRoll : 1.0f;
    return powf(t, 1.0f/e);                           // higher coneRoll = gentler edge (verify sign vs TD)
}
```
**`coneRoll` direction (`pow(t,1/e)` vs `pow(t,e)`) must be locked by one Gemini-Vision A/B** — TD's "higher = gentler" is not formally specified.

**Range window** — at the point/cone attenuation (`:364`):
```cpp
if(L.range > 0.0f){ float x = d/L.range; float w = __saturatef(1.0f - x*x*x*x); atten *= w*w; }
```
This gives the fog shaft a **defined tip** and bounds ray cost — the single highest-value cone upgrade for volumetrics.

**Beam (`startRadius>0`)** — in NEE, sample a point on a disk (radius `startRadius`, plane ⊥ `dir`) instead of the point/sphere, form `wi` toward it, then apply the cone mask. **Must be added in BOTH `sampleLights` and `inscatterFog`** or the beam looks different on surfaces vs in fog. Soft beam edges fall out of the disk sampling for free.

### B.4 Serializing TD Light COMP cone params into `Lightdata`
- Native Light COMP: `lighttype` ∈ {point→0, cone→1, distant→2}; `coneangle` (full-intensity inner cone, degrees); `conedelta` (penumbra band where intensity falls to zero); `coneroll` (1–10 edge curve); `c`×`dimmer` = radiance. TD points down **local −Z** — derive world `dir` from the world-transform's −Z, not +Z.
- **If extending to 16 floats:** new record = `N  px py pz  dx dy dz  cr cg cb  type radius cosInner cosOuter  coneRoll range startRadius`. Update **both**: host parser (`OptixDemoTOP.cpp:772-778`, `f[13]→f[16]`, add `L.coneRoll/range/startRadius`, update the `:764` comment) **and** the Execute DAT — in lockstep, or every field after `type` shifts and lights render garbage.
- `startRadius` and `Softradius` are **not native TD params** — add them as **custom parameters** on the Light COMP (or PT_Render wrapper). For a distant light store `cosI/cosO` unused and put the angular soft-shadow radius in `radius` as today.
- Match TD's "ignore lights < 0.001 intensity" on the host to avoid "missing light" confusion (`PTLight` radiance check already exists at `:357`).

### B.5 Phase-2 quality win — equiangular sampling
Free-flight distance sampling is noisy for bright local spotlights (most scatter samples land where the light barely reaches). **Equiangular sampling (Kulla & Conty 2012)** for point/cone lights in `inscatterFog` — importance-sample distance along the ray ∝ the light's `1/d²` kernel, then apply the cone mask on top — collapses shaft noise dramatically. Keep free-flight for sun/env, equiangular for finite lights. This is the single best follow-up after the fields land, but it **touches the medium NEE inner loop** (higher risk).

---

## C. Mirror material that reflects the beam

### C.1 Status: a perfect mirror already exists
`type==1` metal with `roughness==0` skips the fuzz branch (`:660`) → exact `reflect3` (`:659`). `isDiffuse=0` so all four surface-NEE calls are correctly skipped (`:516-523`). The reflected continuation ray re-enters the loop, runs fog `inscatterFog`, and (with `lastSpec` true) sees `sunDisk`/env on miss. **No engine change is strictly required for a basic mirror** — author `triMat=(1,0,*,*)` (or `11` for smooth normals) on the geometry's Mat input.

### C.2 Why a delta mirror "sees" lights only via the continuation ray
A perfect specular reflection is a Dirac-delta lobe `f = δ(wi − reflect(wo))`. Surface NEE picks a light direction and evaluates `f` there, which is zero almost surely → NEE contributes nothing at a mirror, and (per MIS theory for delta surfaces) you *must* skip it. The mirror therefore only sees radiance arriving along its single reflected direction (the continuation ray). Consequences:
- A **delta light** (analytic sun/point/cone) has zero solid angle, so the continuation ray hits it with probability 0 → **a pure delta spot is invisible in the mirror.** (The *sun* is the exception: `sunDisk(rd)` is drawn on miss for specular rays, `:507-508` — it has a finite drawn disk.)
- To make the spot's *source* (the bright lamp) appear in the mirror, the light needs **area/angular extent**: either author a small emissive disk at the fixture, or add a `lightRadiusVisible` analytic glowing disk drawn on specular/primary miss like `sunDisk`. Without this, only the spot's *effect* is mirrored, never the bulb.

### C.3 How the reflected beam and god ray actually appear

**Reflected beam highlight (lit pool on a surface) — FREE, already works.** The continuation ray hits the lit diffuse surface, which runs `sampleLights` NEE toward the spot; the radiance returns through `thr`, tinted by the mirror's `attenuation=color`. Robust and low-noise because the *diffuse* hit (not the mirror) does the light sampling. `addEmit` stays true after the specular bounce (`:528`) so the reflected ray also picks up the sun disk + emissive lamp geometry.

**Reflected god-ray shaft (Effect A) — FREE, already works.** The shaft is the in-scatter of the spot in the fog volume in front of the mirror. The camera ray reflects, travels in the reflected direction, and `inscatterFog` runs (ungated by `isDiffuse`) at free-flight scatter points on that reflected segment, doing medium NEE straight to the spot (`PTLight type==1`, `:365-371`). Those are **real world-space points**, so their shadow ray to the spot is a normal straight line — no "seeing through" the mirror needed. You see the mirror-image of the shaft, correctly tinted via `thr = mul(thr, fogColor)` after `thr` already carries the mirror `color`.
- **Requirement:** `fogEnable=1` and prefer **multi-scatter** (`fogSingleScatter=0`). In single-scatter mode the *primary* segment terminates at its first scatter (`:491`) and never reaches the mirror, so the reflected shaft only appears on whichever segment scatters first. Either document "use multi-scatter for reflected shafts," or special-case: don't terminate single-scatter until *after* the first specular bounce.

**Mirror-bounced NEW beam (Effect B: spot → mirror → fog) — does NOT work with forward NEE.** A fog point in the *reflected* beam is lit by spot→mirror→P. Medium NEE from P samples the spot with a straight shadow ray that doesn't pass through the mirror → finds nothing. The only forward path is eye→fog@P→continuation→mirror→spot, and the HG-sampled continuation hitting the mirror *and then the delta spot* has probability 0. **Same root cause as specular caustics (LSDE paths).** A forward megakernel PT cannot produce mirror-focused caustics or mirror-bounced volumetric beams without extra machinery.
- **Optional fix (planar mirrors only): virtual-light trick.** Reflect each delta light across the mirror plane → virtual light `L'`. In `sampleLights`/`inscatterFog`/`mediumEmitNEE`, additionally NEE `L'` with a two-segment occlusion test (P→mirror-hit M, then M→L'), attenuated by reflectivity. Exact for a planar perfect mirror; makes both the mirror-bounced surface highlight AND volumetric shaft appear with low noise. Needs a `mirrorPlanes[]` device array (plane eq + reflectivity) + `numMirrorPlanes` + a wrapper toggle. **Ship Effect A first; gate Effect B behind a "Bounce lights through planar mirrors" toggle.**

### C.4 Recommended mirror material polish (optional — improves "reads as a mirror")
Keep the discrete switch; two options (recommend **option 1**, minimal disruption):

**(1) `type==1` `rough==0` → true mirror with metal Fresnel.** Replace flat `attenuation=color` (`:661`) with Schlick metal Fresnel so it brightens to white at grazing (physically correct for conductors, and what makes a mirror *read* as a mirror):
```cpp
float3 F0 = color;                              // reflectivity tint (neutral mirror ≈ 0.97..1.0)
float  cth= fmaxf(dot3(Nf, scl(rd,-1.0f)), 0.0f);
float3 F  = add(F0, scl(sub(V(1,1,1),F0), powf(1.0f-cth, 5.0f)));
prd->attenuation = F;
prd->hitAlbedo   = color;
prd->hitSpecAlb  = F0;                           // F0 is the correct RR specular-albedo AOV
prd->hitRough    = (rough<=0.0f) ? 0.0f : fmaxf(rough,0.02f);  // TRUE 0 for a mirror; keep 0.02 floor only for glossy
prd->isDiffuse   = 0;
```
**Caveat:** metal Fresnel changes the look of *all* existing `type==1` metals (brighter at grazing). If existing scenes were tuned against flat `attenuation=color`, gate the Fresnel behind the `rough==0` path or a toggle.

**(2) Dedicated `type==4` "Mirror" (and `14` smooth).** Identical math, but keeps "metal" as the fuzzy conductor and "Mirror" as always-perfect — cleaner UX for a distinct PT_Render material menu entry. Host needs no change (the emissive scan only special-cases `type%10==3`); only the device `else if` + TD material-attribute authoring.

**Params:** reuse per-vertex `Cd` (`triCd`) as reflectivity/F0 (neutral = white; gold = `(1.0,0.78,0.34)`; dim = grey) and `triMat.y` as roughness (0 = perfect). **No new buffers** for the core mirror. **For a neutral mirror use white `color`** so it doesn't tint the RR specular-albedo guide.

### C.5 isDiffuse handling — keep it 0, never add a mirror to any NEE gate
A delta lobe has no valid NEE. The existing `if(prd.isDiffuse)` gates (`:516-523`) already do the right thing. The fog/medium code must stay *ungated* by `isDiffuse` (it is) so the reflected ray keeps in-scattering — that is the load-bearing property for the reflected shaft. The only `isDiffuse`-adjacent invariant is that `addEmit` stays true after the mirror bounce (`:528`).

---

## D. Denoiser / DLSS-RR caution

**Short answer:** crispy god rays and sharp beam reflections do **not** reliably survive AI denoising in the current setup, and the engine's AOV plumbing is the reason. For a fog pixel the guides describe the *surface/sky behind the fog*; for a mirror pixel they describe the *flat mirror plane* — never the high-frequency thing actually in the pixel. The denoiser sees a smooth, low-variance neighborhood and blurs the shaft / reflected beam across it. NRD itself states it is "not natively designed for volumetrics or transparency" — production keeps volumetrics **out** of the surface denoiser and composites them *after*.

### D.1 God rays / fog — keep them OUT of the guided denoiser
**Best (recommended): separate medium-radiance buffer, composite after denoise.** The engine already isolates every medium contribution: the in-scatter add (`:484-485`), emitter glow (`mediumEmitNEE :393`), env in-scatter (`:376-391`). Route those into a parallel accumulator `Lfog` and output a `fogImage` AOV; denoise only the surface `image`; then `final = denoise(surface) + Lfog` downstream. This is the NRD/production pattern and makes god rays **perfectly crisp** because the AI denoiser never touches them. Fog single-scatter NEE is already low-noise, so leaving `Lfog` lightly filtered (or relying on accumulation) is acceptable — and the shaft *is* the detail you must preserve, so it must not be blurred.

**If keeping one combined buffer (cheap path):** do **not** fabricate albedo/normal for shaft pixels — it cannot encode a thin high-frequency feature and only changes the smear pattern. Instead:
- Lean on **progressive accumulation** (`params.accum`, `:572-577`): on a static hero shot, ~100–500 spp gives clean shafts with zero spatial blur. Gate the AI denoiser to early frames; fall back to raw accumulation once converged.
- For the live path, set the OptiX denoiser **`blendFactor` ~0.15–0.3** so some un-smoothed image survives; cap `taaMaxHist` lower so moving shafts don't ghost.
- **Zero the motion vector for fog-dominated primaries.** The MV is computed from `aoPos` = the surface behind the fog (`:543-554`), but a shaft is anchored to *light+camera*, not that surface — so temporal reprojection drags the shaft along the wrong vector → ghost trails (the documented UE5.7 DLSS volumetric-fog bug + RR "swimming" reports). Detect "a medium scatter happened at b==0 before any surface hit" and set `flow/rrMotion = 0` and reset history weight there. **No AOV value fixes this — only MV zeroing or post-composite does.**

### D.2 Mirror / sharp specular beams — keep DLSS-RR, feed it correctly
DLSS-RR demodulates (divides lighting out of the BRDF, denoises, re-applies) using `rrSpecAlbedo` + `rrRoughness` + `rrHitDist` + specular MV — so it *can* keep a mirror sharp. The OptiX basic/HDR model has **no specular guide at all**, so mirror beams over-smooth there regardless.
- **Keep `hitRough` LOW.** The `fmaxf(rough,0.02)` floor (`:662`) is correct — low roughness = "tight lobe, don't blur." **Never raise it to "fill" fog-pixel guides** — that directly widens the kernel and blurs *all* speculars. For a true mirror, push toward 0 (C.4) for razor reflections, accepting a small RR-stability risk.
- **Fix the spec-hit-distance gap.** `aoSpecHitT` is captured *only* when the **primary** hit is specular (`:463,465-468`). A mirror seen *through fog* or first reached at b≥1 gets `rrHitDist=0` → RR can't reproject the reflection → it smears. Track the *first specular interaction across bounces* (a `specPrevPos`/`gotSpecHit` carry) so any mirror gets a valid `rrHitDist`.
- **Provide specular motion (optional, high value).** Reproject the *reflected* hit point through the previous camera for a proper specular MV (you already have `spec0Pos`). Without it, reflections "swim" under camera motion.
- **For the OptiX (non-RR) denoiser only:** make near-mirror pixels (`roughness < ~0.1`) "look through" — write the **reflected hit's** `hitNormal/hitAlbedo` into the `albedo`/`normal` AOVs instead of the mirror plane's (reuse the existing b==1 reflected hit) so edge detection follows the reflected image.

### D.3 Net guidance on the specific AOVs
- `hitNormal` for fog pixels: leave **0** (honest "no surface"); don't invent one.
- `hitAlbedo` for fog pixels: can't rescue crispness in a combined buffer — *move* in-scatter to a separate buffer (D.1).
- `hitSpecAlb`: keep correct F0 (metal = tint/F0, dielectric = 0.04, emitter = 0); needed for RR demodulation; don't zero for fog.
- `hitRough`: keep the **low floor**; this is the single most important value for surviving specular blur.
- Add a valid spec hit-distance for **every** first specular interaction + ideally a specular MV.

### D.4 Optional upgrade
Switch to `OPTIX_DENOISER_MODEL_KIND_AOV` with demodulated diffuse+specular layers (divide by `hitSpecAlb`/Fresnel before denoise, re-modulate after) so even the OptiX-side denoiser preserves sharp speculars. Pairs naturally with the fog-separation in D.1. Watch for F0-near-0 / emitter NaNs — clamp.

---

## E. Phased, risk-ordered implementation plan

Ordered most-visual-value-first; the renderer stays shippable after every phase. "Touches NEE" flags the high-risk core (the `sampleLights`/`inscatterFog` inner loops + the hardcoded-Lambert NEE functions).

| Phase | What | Effort | Risk | Touches NEE? |
|---|---|---|---|---|
| **0. Verify what exists** | In TD: set a Light COMP to `cone`, confirm the volumetric shaft renders; author `triMat=(1,0,*,*)` mirror, confirm it reflects the beam + shaft (Effect A). Lock the half-angle convention (B.2) with one Gemini-Vision A/B vs a TD Render TOP. | XS | Low | No (read-only) |
| **1. Fog Crispiness slider** | Add `fogShaftG` (+ optional `fogContrast`) to LaunchParams; use `fogShaftG` at `:351`/`:371` only; add the `Crispiness`+`Crispdrives` macro mapping (A.2) + advanced knobs on the Fog tab; add to `fogSig`. **The headline feature and the only genuinely new engine work.** | M | Med (variance/fireflies at the crisp end — lean on accumulation, no firefly net) | **Light touch** — adds a param read inside `inscatterFog`; no NEE restructure |
| **2. Mirror polish** | Metal Fresnel + true-0 `hitRough` for `rough==0` (C.4 option 1) or a dedicated `type==4` (option 2); use white Cd for neutral mirrors. | S | Med (changes all existing metals; gate behind `rough==0` or a toggle) | No (closest-hit only) |
| **3. Cone falloff + range window** | `coneMask` honoring `coneRoll` (`:233`/`:365`); range window at `:364`; extend `PTLight`→16 floats + parser + Execute DAT *in lockstep*. Gives the shaft a defined tip + soft authorable edges. | M | Med (the 13→16-float contract is hand-synced and unvalidated; `coneRoll` sign needs A/B) | **Yes** — edits the cone mask in both NEE paths |
| **4. Denoiser: separate fog buffer + spec-hit-distance fix** | Route medium in-scatter to `Lfog`, composite after denoise (D.1); track first-specular-interaction hit-T across bounces (D.2); zero MV for fog-dominated primaries. The phase that actually *keeps* the crispiness/sharpness alive through RR. | L | Med-High (G-buffer semantics change; temporal validation needed) | No, but touches AOV/G-buffer writes + raygen accumulation |
| **5. Beam (`startRadius`) disk sampling** | Disk emitter sample in **both** `sampleLights` and `inscatterFog`. Unlocks searchlight beams/collimated cylinders. | M | Med (must update both paths or surface≠fog) | **Yes** — both NEE paths |
| **6. (Optional) Equiangular sampling for finite lights in fog** | Kulla & Conty distance importance sampling + cone mask on top; biggest shaft-noise reduction. | L | High | **Yes** — rewrites the medium NEE distance sampling |
| **7. (Optional) Effect B: planar virtual-light mirror bounce** | `mirrorPlanes[]` device array + two-segment NEE in `sampleLights`/`inscatterFog`/`mediumEmitNEE`; toggle-gated. Makes spot→mirror→fog beams + specular caustics appear. | L | High (planar-only; doubles shadow-ray cost; new device buffer) | **Yes** — all three NEE paths |

**Shippability:** Phases 0–2 deliver the headline trio (crisp shafts + verified cone + a mirror that *reads* as a mirror) with zero/low NEE risk. Phases 3–5 are incremental polish. Phases 6–7 are high-value but high-risk and explicitly optional. Do the verification in Phase 0 *before* writing any cone/mirror code — both already exist, and re-implementing them would duplicate/conflict with working systems.

---

## Sources

**Engine files (verified this session):**
- `phase2/OptixDemoTOP/LaunchParamsDemo.h:20-29` (PTLight struct), `:137-153` (fog params)
- `phase2/OptixDemoTOP/demo_programs.cu:250-265` (phaseHG + sampleHGdir), `:201` (smooth01)
- `phase2/OptixDemoTOP/demo_programs.cu:205-247` (sampleLights; cone `:231-236`; Lambert `f=albedo/π :244`)
- `phase2/OptixDemoTOP/demo_programs.cu:336-395` (inscatterFog; sun shaft `:351`, cone shaft `:365`, env `:376-391`, emitter `:393`)
- `phase2/OptixDemoTOP/demo_programs.cu:452-530` (raygen path loop: free-flight `:478-499`, in-scatter `:484-485`, AOV capture `:460-468`, isDiffuse NEE gates `:516-523`, lastSpec/addEmit `:527-528`)
- `phase2/OptixDemoTOP/demo_programs.cu:599-677` (`__closesthit__tri` BSDF switch; metal/mirror `:658-662`)
- `phase2/OptixDemoTOP/demo_programs.cu:543-554` (motion vector from aoPos), `:580-595` (AOV + DLSS-RR G-buffer)
- `phase2/OptixDemoTOP/OptixDemoTOP.cpp:763-787` (Lightdata 13-float parse), `:502-510,892-894` (fog read/upload), `:823-827` (fogSig reset), `:863-866` (light/sun upload), `:1038` (Lightdata param), `:1046-1056` (Fog param UI)
- `pathtttracer.toe.dir/project1/optixDemo/PT_Render/lightSerialize.text` (lines 11/14/17: native Light COMP → cosInner/cosOuter)
- `docs/RESEARCH-volumetrics-and-restir.md` (HG/free-flight/equiangular theory; perf/AOV constraints)

**External:**
- PBRT — Phase Functions: https://pbr-book.org/3ed-2018/Volume_Scattering/Phase_Functions ; Volumetric Light Transport: https://www.pbr-book.org/3ed-2018/Light_Transport_II_Volume_Rendering/Volumetric_Light_Transport ; Path Tracing (delta-BSDF MIS): https://pbr-book.org/4ed/Light_Transport_I_Surface_Reflection/Path_Tracing
- Henyey–Greenstein: https://en.wikipedia.org/wiki/Henyey%E2%80%93Greenstein_phase_function ; Ansys HG: https://www.ansys.com/blog/henyey-greenstein-distribution-model-for-bulk-scattering
- GPU Gems 3 — Volumetric Light Scattering: https://developer.nvidia.com/gpugems/gpugems3/part-ii-light-and-shadows/chapter-13-volumetric-light-scattering-post-process
- Kulla & Conty equiangular sampling (volumetric): https://cseweb.ucsd.edu/classes/sp17/cse168-a/CSE168_14_Volumetric.pdf
- V-Ray soft area shadows: https://novedge.com/blogs/design-news/v-ray-tip-soft-physically-accurate-area-shadows-in-v-ray
- TD Light COMP: https://docs.derivative.ca/Light_COMP ; https://derivative.ca/UserGuide/Light_COMP
- Ragan deferred cone lights (half-angle hint): https://matthewragan.com/2018/03/23/touchdesigner-deferred-lighting-cone-lights/ ; https://github.com/raganmd/touchdesigner-deferred-lighting/tree/master/example-lights-cone
- NRD README (volumetrics not denoised; demodulation): https://github.com/NVIDIA-RTX/NRD/blob/master/README.md
- OptiX denoiser guide layer: https://raytracing-docs.nvidia.com/optix7/api/html/struct_optix_denoiser_guide_layer.html
- DLSS-RR swimming artifact: https://forums.developer.nvidia.com/t/dlss-ray-reconstruction-persistent-swimming-flowing-artifact-during-camera-motion-even-at-16-spp/361209
- DLSS + UE5.7 volumetric fog bug: https://forums.developer.nvidia.com/t/dlss-and-ue5-7-volumetric-fog-bug/354828
- Denoising overview: https://alain.xyz/blog/ray-tracing-denoising
- Specular caustics / volumetric PT context: https://en.wikipedia.org/wiki/Volumetric_path_tracing ; https://noahpitts.github.io/godRay/