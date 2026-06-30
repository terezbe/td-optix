# Research: Rendering an FBX file in the OptiX path tracer

**Status:** research / design — nothing implemented. **Date:** 2026-06-25.
**Target:** `OptixDemoTOP` — the renderer consumes a **POP triangle-soup**, not SOPs/meshes. **TD 2025.32820.**

The goal: drag an FBX in (a downloaded model) and render it. The catch is that TD's FBX import produces a SOP/Geometry-COMP hierarchy, while the renderer needs a flat CUDA-origin triangle soup with a per-vertex matID. This note maps the bridge, grounded in the renderer's source + the TD docs.

---

## 0. What the renderer actually consumes (verified from source)

The closest-hit reads vertices `3·pid … 3·pid+2` — **3 consecutive un-indexed verts = 1 triangle** (flat soup, no index buffer). Attributes arrive through **separate carriers**, all in soup (vertex==point) order:

| Soup attr | Device array | Carrier today | Packing |
|---|---|---|---|
| Position P | `triVerts` | **TOP input 0** (CUDA-origin `poptoTOP`) | xyz |
| Color/matID | `triCd` | **TOP input 1** | rgb=color, **`.w`=matID** |
| Material | `triMat` | **TOP input 2** | (type, roughness, ior, emitStrength) |
| Normal N | `triN` | **TOP input 3** | smooth normal; null→flat |
| UV "Tex" | `triUV` | **`Geopop` POP param** (`getAttribute(Point/Vertex,"Tex")`) | float3 UVW (NOT a TOP input — Vulkan deadlock) |
| Base-color map | `baseColorTex` (layered, 1 layer/matID) | **`Texturepop` POP param**, CPU-bounced | square layers keyed by `triCd.w` |

**Three hard constraints the bridge must satisfy:** (a) geometry **triangulated + de-indexed** into a soup; (b) every vertex carries a **matID in `Cd.w`**; (c) the carriers into CUDA inputs must be **CUDA-origin** (`poptoTOP`) — any Vulkan-origin POP/TOP must be **CPU-bounced** (§5). No tangents today; one map type (base color).

---

## 1. The TD FBX COMP (TD 2025)

The **[FBX COMP](https://docs.derivative.ca/FBX_COMP)** imports geometry, animations, and scenes from Maya/Max/C4D/Houdini/Blender FBX.
- **Import cache:** assets bake into a **`.tdc`** file in a **`TDImportCache`** folder next to the `.toe`, reconstructed via **Import Select** OPs. Missing `.tdc` ⇒ re-parse.
- **Structure:** a container building a **hierarchy of [Geometry COMPs](https://docs.derivative.ca/Geometry_COMP)**, each holding its mesh as a **SOP** + a **Material (MAT)** parameter ("every Geometry component needs a Material assigned"). Optional imported Lights/Cameras.
- **Textures:** embedded or external, located via the **Texture Directory** par.
- **2025 — `Import POPs` parameter, DEFAULT = On (confirmed):** the FBX COMP (and USD COMP) "Import POPs" parameter "create[s] POPs in place of SOPs when importing geometry, **with the default set to 'On' to import using POPs**" (TD **2025.30000+** [Release Notes](https://docs.derivative.ca/Release_Notes)). So **a freshly imported FBX is POP-native by default** — geometry lands as POPs inside the Geo COMPs, not SOPs. (The FBX COMP doc page itself doesn't state the default; the release notes do.) POP import currently supports **mesh + point** primitives; unsupported types are converted to triangles/quads/line-strips/points.
- **Drag-drop vs File ▸ Import** are documented as equivalent (no behavioral difference).
- **To reach the geometry as a POP from outside:** the **[Import Select POP](https://docs.derivative.ca/Import_Select_POP)** — `Import Parent` = the FBX (or USD) COMP, `Geo Path` = which object, **`Compute Tangents`** toggle (relevant once normal maps land), `Reload`. Output is a **GPU-resident POP**, wireable straight into a POP Merge. Derivative staff explicitly recommend `Import Select POP` (+ **Skin Deform POP** for skinned/deformed meshes — [forum](https://forum.derivative.ca/t/importing-fbx-wishlist/526349)). This is the clean, POP-native extraction path (no SOP→SOP-to-POP detour needed in 2025).

---

## 2. FBX geometry → POP soup (the SOP→POP path)

Two entry points, both landing on a POP:
1. **`Import POPs` On** → geometry loads as POPs directly.
2. **SOP route** (more control over triangulation/tangents): **[Import Select SOP](https://docs.derivative.ca/Import_Select_SOP)** (`Geo Path` selects the object; has a `Compute Tangents` toggle) → **[SOP to POP](https://docs.derivative.ca/SOP_to_POP)**.

**SOP to POP** is the converter and defines the soup attribute names — and they **already match the renderer**: `P`→`P`, **`uv`→`Tex`**, **`Cd`→`Color`**, `N`→`N`. Set **`5+ Points Polygons = Triangulated`** to guarantee tris (and pre-triangulate quads with a **Convert/Facet** since the renderer assumes strictly 3 verts/prim). Its GPU params (`Free Extra GPU Memory`, `Delete Input Attributes`) confirm the output is **GPU-resident**.

POP **[attribute classes](https://docs.derivative.ca/Attribute)** are Point / Vertex / Primitive. FBX UVs are typically **per-face-vertex** → they land as **Vertex-class `Tex`** — and the renderer's `Geopop` read already falls back to Vertex-class `Tex` ("soup → vertex order == point order"), so this is handled **provided the soup is de-indexed vertex-faithfully**.

---

## 3. FBX materials + textures → the PT Material system (the matID problem)

TD's **[PBR MAT](https://docs.derivative.ca/PBR_MAT)** is what FBX/Substance materials map onto:

| PBR MAT | Param | Renderer target |
|---|---|---|
| Base Color | `basecolormap`/`basecolor` | `baseColorTex` layer + `triCd.rgb` |
| Metallic | `metallicmap`/`metallic` | `triMat` type-1 selector (or ORM, per [material-maps research]) |
| Roughness | `roughnessmap`/`roughness` | `triMat.y` |
| Normal | `normalmap` | **needs tangents (T[4]) — out of scope for v1** |
| Emission | `emitmap`/`emit` | `triMat` type-3 + `triCd` |
| AO/Height/Alpha | — | none today |

The PBR MAT normal map "**requires tangent attributes (T[4]) on geometry**" — the renderer has none, so **normal maps from FBX are v1-out-of-scope** (until the engine grows tangents + a normal slot, per the material-maps research).

**The matID-per-object problem (central integration cost):** FBX assigns materials **per object / per face-group**, but the renderer keys everything off a single **per-vertex `Cd.w` matID** into a **layered** atlas (1 layer per matID, `numMaterials` layers). The bridge must:
1. **Enumerate** the FBX's N objects/materials (one Geo COMP per object; Import Select SOP's Info DAT exposes original primitive paths).
2. **Assign a stable integer matID** (0..N-1) and **stamp it into `Cd.w`** for every vertex of that object (a per-object constant — an Attribute POP writing `Color.a = matID`).
3. **Bake each material's base color** (map or flat) into a **square layer** of the `texStack` at index matID — the existing matbake→texStack→texToPop→Texturepop pipeline, driven by FBX materials.
4. **Pack `triMat`** (type/rough/ior/emit) per object from the PBR MAT values.

Respect the engine's **`numMaterials` license cap** (already adaptive). This per-object→per-vertex flattening + layered atlas is the main material-side engineering.

---

## 4. UVs / normals / tangents from FBX

- **UVs:** `uv`→`Tex` (Vertex-class for per-face-vertex). Handled by the existing Geopop Vertex-fallback if the soup is de-indexed.
- **Normals:** imported as `N`; if missing, a **Facet SOP** (compute normals) or letting TD recompute fixes it; renderer falls back to flat geometric.
- **Tangents:** "most 3D software does not prep models on export with tangents" — FBX rarely ships them. Generate via Import Select SOP's **`Compute Tangents`** when needed. **But the renderer ignores tangents today**, so skip until normal mapping lands.

---

## 5. Recommended bridge architecture (phased)

**⚠ Backend reality check (highest risk).** TD's GPU backend is **Vulkan 1.1** and POPs run almost entirely on the GPU. The project's root-cause ([td-cuda-input-freeze] memory): a **Vulkan-backed TOP** into a CUDA input **hard-deadlocks inside `beginCUDAOperations`**, and `getBuffer(CUDA)` on a **Vulkan-origin POP** deadlocks at the interop handoff. **Only CUDA-origin sources are safe on CUDA inputs** (`poptoTOP`, soupFacet-style POPs with no Vulkan upstream); **anything Vulkan-origin must be CPU-bounced.** This dictates the whole bridge.

**A custom "FBX→soup" COMP** (the recommended shape) wraps an FBX COMP, processes every object, and outputs one POP to merge upstream of `soupFacet`. The FBX is POP-native by default (Import POPs On), so use **Import Select POP** — no SOP detour:
```
FBX COMP  (Import POPs = On → POPs by default)
  └─ per object i (enumerate via the FBX/Import-Select Info):
       Import Select POP (Import Parent=FBX COMP, Geo Path=obj_i)   ← GPU POP, optional Compute Tangents
       → triangulate (Facet/Convert POP)  → ensure N (compute if missing)
       → Attribute POP  (Color.a = matID_i ; pack a Mat POP: type/rough/ior/emit from the object's PBR MAT)
  Merge all objects → ONE soup POP → soupFacet (de-index to 3 verts/tri, vertex-faithful)
       → carriers:  P→poptoTOP→in0 ✅ | Cd→poptoTOP→in1 ✅(.w=matID) | Mat→poptoTOP→in2 ✅ | N→poptoTOP→in3 ✅
                    Tex→kept ON the soup POP → render "Geopop" param ✅
  Materials (per matID):
       PBR MAT base color → matbake → texStack Layout TOP (square layers, N=numMaterials)
       → toptoPOP "texToPop" → render "Texturepop"  ⚠ Vulkan-origin → render reads it CPU-BOUNCED (already shipped)
```
**CUDA safety:** the FBX POP is Vulkan-origin, but it flows into the **existing `soupFacet → poptoTOP` bridge** (the proven CUDA-origin hand-off for inputs 0-3) — so feeding it in upstream of `soupFacet` adds **no new deadlock surface**. (Skinned/animated FBX: insert a **Skin Deform POP** after Import Select POP, per Derivative staff.)

**Phasing:**
- **A — static single-object FBX, base color only.** One object → Import Select → triangulate → SOP to POP → soupFacet → 4 poptoTOPs + Geopop. matID=0 everywhere. Proves the geometry path end-to-end.
- **B — multi-object / multi-material.** Per-object matID stamping into `Cd.w`; merge; bake each base color into its texStack layer; drive `numMaterials`. The matID-flattening + atlas + license-cap work.
- **C — material richness.** Map PBR `metallic`/`roughness`/`emit` → `triMat`.
- **D — animation / instancing / scale** (hard parts below).

**Hard parts / risks:**
- **Vulkan→CUDA deadlock** (re-verify with the breadcrumb watchdog after wiring; never feed an FBX-derived Vulkan TOP/POP straight into a CUDA input; keep the base-color atlas CPU-bounced).
- **De-indexing + vertex order:** the renderer needs un-indexed 3-verts/tri with per-face-vertex UV/N/matID, in an order where soup index == vertex index. Indexed FBX meshes must be expanded.
- **Large meshes / per-cook rebuild:** FBX can be 10⁵–10⁶ tris; the renderer rebuilds the GAS + re-copies carriers each cook. Static geometry should cook once / cache the GAS, not rebuild per frame.
- **Instancing:** FBX instances must be **flattened** (transforms baked into world-space verts) — the soup is one flat world-space buffer with no instance transforms.
- **Quads/n-gons:** `5+ Points Polygons = Triangulated` + pre-triangulate quads.
- **Tangents / normal maps:** out of scope until the engine grows tangents (see [material-maps research]).
- **Per-object→per-vertex material flattening + layered-atlas license cap** — the main material-side engineering.

---

## Sources
[FBX COMP](https://docs.derivative.ca/FBX_COMP) · [Import Select POP](https://docs.derivative.ca/Import_Select_POP) (POP-native FBX extraction; default since 2025.30000 per [Release Notes](https://docs.derivative.ca/Release_Notes)) · [Geometry COMP](https://docs.derivative.ca/Geometry_COMP) · [Import Select SOP](https://docs.derivative.ca/Import_Select_SOP) · [SOP to POP](https://docs.derivative.ca/SOP_to_POP) · [Attribute (POP classes / Tex / Color / N)](https://docs.derivative.ca/Attribute) · [PBR MAT](https://docs.derivative.ca/PBR_MAT) · [POPs — A new Operator Family](https://derivative.ca/community-post/pops-new-operator-family-touchdesigner/69468) · [2025 Official Update](https://derivative.ca/community-post/2025-official-update/73153) · [System Requirements (Vulkan 1.1)](https://docs.derivative.ca/System_Requirements) · [Import FBX as SOP (forum)](https://forum.derivative.ca/t/import-fbx-file-as-a-sop/457818) · [USD→GPU materials/tangents/normals (forum)](https://forum.derivative.ca/t/solved-usd-import-to-gpu-with-materials-tangents-and-normals/300976).

Related: `memory/td-cuda-input-freeze.md` (Vulkan→CUDA deadlock + CPU-bounce rule), `docs/RESEARCH-material-maps.md` (the maps FBX materials would drive). Renderer soup contract: `LaunchParamsDemo.h`, `OptixDemoTOP.cpp`.
