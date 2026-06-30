# Third-Party Notices

This project builds against, and at runtime uses, software from NVIDIA. **None of the
NVIDIA SDKs or their headers/libraries are redistributed in this repository.** They are
governed by their own licenses. You must obtain them from NVIDIA and agree to those
licenses to build or run this software.

The MIT license in [`LICENSE`](LICENSE) applies **only** to the original source in this
repository (the OptixDemoTOP plugin source, the PT_Render COMP, scripts, docs and
example projects).

---

## NVIDIA OptiX (9.1)
- **Use:** ray-tracing pipeline + AI denoiser. The OptiX runtime ships **inside the NVIDIA
  display driver**; only the OptiX **headers** are used at build time (not redistributed).
- **License:** NVIDIA OptiX SDK License Agreement / NVIDIA Software License Agreement.
- Get it: <https://developer.nvidia.com/rtx/ray-tracing/optix>

## NVIDIA DLSS — Ray Reconstruction (NGX)
- **Use:** AI denoiser/upscaler. The runtime DLL **`nvngx_dlssd.dll` is bundled in the Release
  build** (the renderer's default denoiser); the graceful fallback uses the OptiX denoiser when
  DLSS is unavailable. It is shipped **unmodified and NVIDIA-signed**.
- **Ownership / license:** `nvngx_dlssd.dll` is **© NVIDIA Corporation**, provided under the
  **NVIDIA DLSS SDK License / NVIDIA RTX SDKs License** — it is **NOT** covered by this project's
  MIT license and may not be modified, reverse-engineered, or resold on its own.
- **Required attribution:** *"This software contains NVIDIA DLSS technology"* / *"This software
  contains source code provided by NVIDIA Corporation."* (also stated in the README and the
  release `HOW-TO-RUN.txt`).
- **Commercial use:** if this were shipped commercially, NVIDIA requires pre-release notification
  (<https://developer.nvidia.com/sw-notification>) and a registered NGX App ID. This is a
  **non-commercial** project, so those obligations do not apply; the in-code NGX id is an R&D UUID.
- Source / updates of the DLL: <https://github.com/NVIDIA/DLSS>

## NVIDIA CUDA Toolkit (12.8)
- **Use:** device kernels + the CUDA runtime. `cudart64_12.dll` **is** redistributable under
  the CUDA Toolkit EULA (Supplement / Attachment A) and may be shipped in releases.
- **License:** NVIDIA CUDA Toolkit EULA.
- Get it: <https://developer.nvidia.com/cuda-toolkit>

---

## Derivative TouchDesigner SDK headers
`phase2/OptixDemoTOP/CPlusPlus_Common.h` and `TOP_CPlusPlusBase.h` are the TouchDesigner
C++ TOP API headers, **owned by Derivative Inc.** and provided under Derivative's
**"Shared Use License"** (the notice is retained at the top of each file). They may be
shared/redistributed for use with TouchDesigner provided that notice is kept and
Derivative's name/trademarks are not used to endorse derived works. They are NOT covered
by this project's MIT license.

## Host application
**Derivative TouchDesigner** is required to run this plugin and is licensed separately by
Derivative. This project is not affiliated with or endorsed by Derivative or NVIDIA.

> This file is informational, not legal advice. Verify against the **current** OptiX, DLSS
> and CUDA EULAs before distributing anything that touches these components.
