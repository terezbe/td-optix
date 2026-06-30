# td-optix — real-time RTX path tracing inside TouchDesigner

> **I built my own real-time render engine inside TouchDesigner — custom NVIDIA OptiX + CUDA C++ with AI ray reconstruction.**

<!-- TODO: add a looping screen-capture of the live moving viewport here:  ![hero](assets/hero.gif) -->
**[▶ Download & run it (Releases)](https://github.com/terezbe/td-optix/releases/latest)** · MIT · Windows + NVIDIA RTX + TouchDesigner

**PT_Render** is a real-time, physically-based **path tracer** that runs as a native **C++ TOP** inside [TouchDesigner](https://derivative.ca). It traces real light — soft shadows, glass refraction, mirror reflections, emissive geometry and volumetric fog — and reconstructs a clean image every frame with **NVIDIA DLSS Ray Reconstruction**, so it stays interactive instead of waiting on an offline render.

It's wrapped in a single drop-in COMP (**`PT_Render`**) with a clean parameter UI, and it feeds on **POPs** — so you build scenes with TouchDesigner-native geometry.

---

## ✨ Features

- **Real-time RTX path tracing** — NVIDIA **OptiX 9.1** + **CUDA 12.8**, hardware-accelerated on RTX GPUs.
- **AI denoising with graceful fallback** — DLSS **Ray Reconstruction** → OptiX AI denoiser → raw accumulation, chosen automatically so it works even without DLSS.
- **POP-native geometry** — feed live POP/SOP geometry straight into the renderer (triangulated automatically; any topology is safe).
- **Materials** — diffuse, metal (roughness), glass (dielectric / IOR), and emissive — encoded per-point so a whole scene is one geometry stream.
- **Volumetric fog** — free-flight single-scatter, emitter glow, HDRI/sky in-scatter, with motion-stability and god-ray "crispiness" controls.
- **Lighting & environment** — HDRI (importance-sampled), analytic Preetham physical sky, and TouchDesigner Light COMPs (point / spot / distant).
- **Camera-motion robustness** — infinite-distance sky motion vectors + specular reflection motion vectors keep the background and mirrors stable while the camera moves.
- **Anti-boil controls** — decorrelated sampling + fog firefly clamp to tame temporal shimmer.

---

## 🖥️ Requirements

| | Minimum |
|---|---|
| GPU | NVIDIA **RTX** (Turing / Ampere / Ada / Blackwell) — built for `sm_89` |
| Driver | recent driver for the OptiX path; **≥ 590** for DLSS Ray Reconstruction (auto-falls back if older) |
| OS | Windows x64 |
| Host | TouchDesigner **2025.32820+** (C++ TOP, CUDA execute mode) |
| Bundled runtime | `cudart64_12.dll` (CUDA EULA redistributable) |

> **DLSS Ray Reconstruction is included** in the [Release](https://github.com/terezbe/td-optix/releases/latest) build (`nvngx_dlssd.dll`) and used by default; if the driver is too old it auto-falls back to the OptiX AI denoiser. The bundled `nvngx_dlssd.dll` is **NVIDIA's unmodified redistributable, © NVIDIA Corporation**, provided under the NVIDIA DLSS SDK license — **not** covered by this project's MIT license. *This software contains NVIDIA DLSS technology.* See [`THIRD-PARTY-NOTICES.md`](THIRD-PARTY-NOTICES.md).

---

## 🚀 Quick start

1. Install the requirements above.
2. Open one of the example projects in [`examples/`](examples/) (see [examples/README.md](examples/README.md)).
3. Drive the camera, tweak the `PT_Render` parameters (Render / Denoise / Fog / Camera pages), and watch it converge live.

To use the renderer in your own project, drop the **`PT_Render`** COMP into your network and feed it geometry (a POP), a camera, and lights.

---

## 🧪 Examples

| Project | Scene |
|---|---|
| [`examples/EMBER_PILE.toe`](examples/) | A warm pile of glowing emissive orbs in a dark void, with glass spheres refracting the embers — emissive + glass + god-ray glow. |
| [`examples/CHROME_FIELD.toe`](examples/) | A cooler field of mirror-chrome and clear-glass spheres over a reflective floor, lit by teal / white / amber emitters — reflections + refraction + dark-environment lighting. |

Both are built entirely from **POPs**. See [examples/README.md](examples/README.md) for the scene breakdown and how to switch between them.

---

## 🔨 Build from source

Device code (PTX) and the host DLL are built with the scripts in [`scripts/`](scripts/):

```powershell
# Device → PTX (when demo_programs.cu / LaunchParamsDemo.h change)
pwsh scripts/compile_ptx.ps1
# Host → DLL (when the .cpp/.h change)
pwsh scripts/build_dll.ps1
# Stage a runnable bundle
pwsh scripts/stage_runtime.ps1
```

Build toolchain: **CUDA Toolkit 12.8**, **MSVC v142** (VS2019 Build Tools), **OptiX 9.1** headers, and (optional) the **DLSS/NGX** SDK. Full requirements and paths are in [`docs/DEPENDENCIES.md`](docs/DEPENDENCIES.md).

> The NVIDIA SDKs (OptiX headers, DLSS SDK) are **not** included — install them locally and point the build at them.

---

## 📚 Docs

In-depth notes live in [`docs/`](docs/): the denoiser conventions, camera-motion noise research, render-quality measurement methodology, the fog/volumetrics design, and the parameter reference.

---

## 📜 License

**MIT** for the original source in this repo — see [`LICENSE`](LICENSE).
Third-party (NVIDIA OptiX / DLSS / CUDA) components are **not** redistributed and are governed by their own licenses — see [`THIRD-PARTY-NOTICES.md`](THIRD-PARTY-NOTICES.md).

## 🙌 Credits

Built by **Erez** ([erezc.media](mailto:erezc.media@gmail.com)). Powered by NVIDIA OptiX, CUDA and DLSS Ray Reconstruction, inside Derivative TouchDesigner.
