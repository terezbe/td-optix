# FIX — Quad-topology render bug (triangulate before the soup)

**Date:** 2026-06-29 · **Status:** ✅ shipped in the live `optixDemo/PT_Render` network.

## Symptom
Shapes built with **quad** connectivity render **broken/garbled**; the same shapes built with **triangle** connectivity render fine. Reported across the whole engine (not one scene).

## Root cause (engine reads a non-indexed triangle soup)
`OptixDemoTOP.cpp`:
- `:437` — `N -= (N%3u); // whole triangles only`
- `:445` — `bi.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_NONE; // triangle soup: 3 consecutive verts = 1 triangle`

The engine has **no connectivity/index buffer** — it reads the baked vertex stream as a flat list where **every 3 consecutive verts = 1 triangle**. The soup chain is:

```
in1 (POP In) → transform1 → soupFacet (Facet, operation='unique') → soupCd/soupMat/soupTex/soupN (popto, extract='point')
```

`Facet operation='unique'` un-shares points so point-order == vertex-order. This is a valid triangle soup **only if every primitive is a triangle**. A **quad** contributes 4 verts → the engine reads `[0,1,2]` as a triangle and starts the next triangle at vert 3 → **every following triangle is mis-aligned**, and the trailing `N%3` verts are dropped. The grids (`cf_floor`, `mir_plane`, all `*_grid`) were `surftype='quads'`.

**Objective proof (live chrome scene, before fix):** `soupFacet` = **84580 verts, 84580 mod 3 = 1** → not a whole number of triangles.

## Fix — one triangulation chokepoint
Inserted **`soupTri` (Triangulate POP)** between `in1` and `transform1`:
```
in1 → soupTri → transform1 → soupFacet → …
```
- `triangulatequads = True`  ← **critical** (defaults to OFF; without it the POP is a no-op on quads)
- `mode = convex`, `remzerotri = True` (drop degenerate tris defensively)

Facet POP **cannot** triangulate (its `operation` menu is only `none/unique/cusp/conspoints`). The Triangulate POP is the purpose-built operator.

## Verification (all four geoSwitch scenes)
| Scene (geoSwitch idx) | tris | verts | verts mod 3 | clean (tris×3==verts) |
|---|---|---|---|---|
| 0 UV-test | 27800 | 83400 | 0 | ✅ |
| 1 Mirror | 27800 | 83400 | 0 | ✅ |
| 2 Ember/Oracle | 36480 | 109440 | 0 | ✅ |
| 3 Chrome | 28210 | 84630 | 0 | ✅ |

Visual: the floor that previously looked broken now renders as a clean reflective plane. Because the chokepoint sits before the bake, **any** future geometry (boxes, tubes, imported meshes, n-gons) is auto-triangulated — engine-wide robustness, as requested.

## Why this is the right layer
The engine can't triangulate (no connectivity at that point), and per-grid `surftype='triangles'` would only fix *current* grids. A single in-chain Triangulate POP is the robust, future-proof fix.
