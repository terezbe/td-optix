# Examples

Two POP-built showcase scenes for **td-optix**. Each `.toe` loads the renderer from a
**`bin/`** folder beside it (the `OptixDemoTOP.dll` + `demo_programs.ptx` + CUDA/DLSS
runtime), so keep each `.toe` together with its `bin/`. In a distributed package the
layout is:

```
td-optix/  ├── EMBER_PILE.toe  ├── CHROME_FIELD.toe  └── bin/ (DLL + PTX + cudart + nvngx)
```

> `bin/` holds build outputs and is **gitignored** — in the repo you populate it by
> building (`scripts/`) + copying the runtime there, or grab a Release. Open either
> `.toe` in TouchDesigner 2025.32820+.

Each file opens with its scene pre-selected (the internal `geoSwitch` inside
`/project1/optixDemo`).

| File | Scene switch | Look |
|---|---|---|
| `EMBER_PILE.toe` | `geoSwitch = 2` | Warm pile of glowing **emissive** orbs in a dark void; **glass** spheres refract the embers; central god-ray glow. Showcases emissive + glass + dark-environment glow. |
| `CHROME_FIELD.toe` | `geoSwitch = 3` | Cooler field of mirror-**chrome** and clear-**glass** spheres over a reflective floor, lit by teal / white / amber **emitters**. Showcases reflection + refraction + colored emissive lighting. |

## How they're built (POP-native)

Each sphere group is a `grid` POP of scattered template points → a `copy` POP that
instances a geodesic `sphere` onto those points → a material `base` COMP that tags the
points with the per-point material (`Cd` color + `Mat` = type/roughness/ior/emit). The
groups merge into the scene's `*_merge`, which feeds the `geoSwitch` → `PT_Render`.

Materials are encoded per point:
- `Cd` = base color
- `Mat` = (type, roughness, ior, emitStrength), where **type** is `0` diffuse · `1` metal · `2` glass (dielectric) · `3` emitter.

## Tuning

Select the **`PT_Render`** COMP and use its parameter pages:
- **Render** — samples/pixel, bounces, firefly clamp.
- **Denoise** — denoiser (default **Rr** / DLSS-RR), and **Decorrelate Noise** (anti-boil).
- **Fog** — volumetric fog density / color / sky in-scatter, stability, **Fog Firefly Clamp**.
- **Camera** — DOF aperture, focus.

> These scenes share the consolidated dev project; the geometry generators for the other
> (UV-test / mirror) scenes are present but unused at their scene index.
