# PT_Render — Parameter Map & Scene Architecture

**Status:** Reference (mapped live from `pathtttracer_branch.toe`, 2026-06-26). Use this to assign each test scene the correct dialed-in settings.
**COMP:** `/project1/optixDemo/PT_Render` (wraps the `render` C++ OptiX TOP). Wrapper params drive the engine via expression/bind; the engine also has params not all mirrored on the wrapper (see Gaps).

---

## 1. Wrapper parameter pages (38 params)

### Render Scene (the scene bindings)
| Param | Type/Mode | Value | Meaning |
|---|---|---|---|
| `Camera` | OP | `pcam` | Camera COMP (transform lives on `pcam` / its parent `null5`) |
| `Lights` | Object | **None** | Light-COMP wildcard — UNUSED; lights actually arrive via the `Lightdata` string (see §2) |
| `Environment` | OP | `PTEnv` | Environment COMP (sky/sun/HDRI/background) |
| `Active` | Toggle | True | render on |
| `Materials` | Object | **None** | per-material texture registry — UNUSED (matbake paused) |
| `Heroskymode` | Menu(BIND) | Physical | Gradient / Hdri / Physical — bound two-way to PTEnv |
| `Heroskystr` | Float(BIND) | 0.681 | sky strength (0..3), bound to PTEnv |
| `Herohdrirot` | Float(BIND) | 0.0 | HDRI rotation (−180..180), bound to PTEnv |

### Render
| Param | Value | Meaning | ⚠ |
|---|---|---|---|
| `Samplesperpixel` | 9 | samples/pixel/frame | slider range mis-set to 0..1 |
| `Maxbounces` | 6 | path depth | slider range mis-set to 0..1 |
| `Fireflyclamp` | 1.0 | indirect firefly clamp (0=off) | |
| `Directlightnee` | False | sphere-mode NEE (off in tri mode) | |
| `Lightmaster` | 1.0 | global light intensity ×  | |
| `Reset` | Pulse | restart accumulation | |
| `Renderw` / `Renderh` | 1280 / 720 | desired output res (16..3840) | |
| `Updateres` | Pulse | **safe resolution-change button** (pauses COMP, applies, resumes) | |

### Denoise
| Param | Value | Meaning | ⚠ |
|---|---|---|---|
| `Denoiser` | Rr | None / Optix / Taa / **Rr (DLSS-RR)** | |
| `Optixstrength` | 0.428 | OptiX denoise blend (Optix only) | |
| `Taamaxhistory` | 0.789 | TAA history cap (Taa only) | |
| `Aajitter` | 1.0 | AA jitter amount (non-RR) | |
| `Rrmotionscale` | 1.0 | → engine `Flowscale` (MV scale/sign) | engine default is **−1.0** — verify mapping (MV-scale bug zone) |
| `Rrflipy` | True | → engine `Flowinvy` (MV Y flip) | |

### Texture
| Param | Value | Meaning |
|---|---|---|
| `Texcoords` | Auto | Auto / Uv / Projection |
| `Projectionscale` | 0.3 | triplanar tile size |

### Fog
| Param | Value | Meaning |
|---|---|---|
| `Fogenable` | True | participating medium on |
| `Fogdensity` | 0.014 | extinction σ_t (0..0.4) |
| `Fogcolor` r/g/b | 0.9 | single-scatter albedo |
| `Foganisotropy` | 0.74 | HG g (−0.9..0.9; >0 forward/sun-halo) |
| `Fogskystr` | 0.0 | sky/env in-scatter strength (0..2) — **at 0 the env background is hidden by fog** (known bug) |
| `Fogemitnee` | True | medium NEE toward emissive geometry |
| `Fogsinglescatter` | True | single-scatter only (fast, low-noise) |
| `Fogmaxscatter` | 1 | multi-scatter cap (0..16) |
| `Fogrrstart` | 0.147 | fog Russian-roulette start |

### Camera (DOF only — transform is on `pcam`)
| Param | Value | Meaning |
|---|---|---|
| `Dofaperture` | 0.0 | lens radius (0..0.3; 0 = pinhole) |
| `Focusdistance` | 10.3 | focus plane (0.5..40) |

---

## 2. Scene-feeding architecture (what a "scene" actually is)

A renderable scene is a **bundle** of five independent feeds — switching scenes means switching all of them together:

| Feed | Source | How it reaches the render |
|---|---|---|
| **Geometry** | `/project1/optixDemo/uvt_merge1` (POP merge) | → PT_Render input 0 → `soupFacet` (de-index) → `soupTex/soupCd/soupMat/soupN` poptoTOP CUDA carriers (inputs 0–3). `Cd.w`=matID, `Mat`=(type,rough,ior,emit), `Tex`=UV |
| **Lights** | `Lightdata` STRING param | serialized from TD Light COMPs by the `lightSerialize` Execute DAT (count + 13 floats/light: pos, dir, rgb, type, radius, cosInner, cosOuter). NOT a TOP input (Vulkan→CUDA deadlock). Currently 3 lights |
| **Environment** | `PTEnv` COMP | Mode (Gradient/Hdri/Physical), sun (dir/color/strength/angle), HDRI file + rot + importance, background mode + color. Currently Physical, sun strength 0, Venice HDRI loaded-but-inactive |
| **Camera** | `pcam` (Camera COMP) ← parent `null5` | `null5` carries the transform / LFO animation; `pcam` referenced by `Camera` param |
| **Render settings** | the PT_Render params above | spp, bounces, fog, denoiser, DOF, etc. — the per-scene "look/quality" dial-in |

**Material model:** per-vertex on the POP soup — `Mat.x`=type (0 lambert / 1 metal-or-mirror / 2 glass / 3 emitter; +10 = smooth normals), `Mat.y`=roughness (0 metal = mirror), `Mat.z`=ior, `Mat.w`=emitStrength; base color via `Cd` (+ optional layered base-color map keyed by `Cd.w` matID).

---

## 3. Gaps / cleanup (the "param architecture mess")

- **`Fog Stability` (engine `Fogstability`) is NOT mirrored on the wrapper Fog page** — engine-only; add a wrapper param when adopted.
- **Slider ranges (normMin/normMax) mis-set** on `Samplesperpixel` (0..1 vs value 9) and `Maxbounces` (0..1 vs value 6) — handles peg; widen to the engine clamp (Spp 1..64, bounces 1..32).
- **`Rrmotionscale` wrapper=1.0 vs engine `Flowscale` default −1.0** — confirm the wrapper→engine mapping doesn't drop the sign (MV-scale is the documented swim/ghost bug zone).
- `Lights` and `Materials` Object params are wired but **None** (lights via `Lightdata`, materials path paused) — either use or hide to avoid confusion.

---

## 4. Per-scene settings concept (for the test fixtures)

Each test scene = a **preset** that dials: geometry feed, the `Lightdata` (its lights), the `PTEnv` settings, the `pcam`/`null5` camera, and the PT_Render look/quality params, **plus** a baked converged reference (see `RESEARCH-render-quality-measurement.md`). A scene **switch** applies the active preset to all of the above. (Design in progress — see the scene-switch proposal.)
