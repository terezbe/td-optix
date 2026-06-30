# Fix: Output-Resolution Change Freezes TouchDesigner

**Status:** SOLVED (2026-06-26). Shipped in `pathtttracer.92.toe`.
**Component:** `OptixDemoTOP` at `/project1/optixDemo/PT_Render/render` (OptiX 9.1 + CUDA 12.8 C++ TOP, TD 2025.32820, RTX 4080 Super).

---

## Symptom

Changing the renderer's **output resolution** (e.g. width 720 → 1080 on the render TOP's Common page) **hard-freezes TouchDesigner** — black/unresponsive window, `process.Responding == False`, recoverable only by force-kill. DLSS-RR was on; ~10 GB VRAM free, so **not** an out-of-memory issue. It reproduced on the very first resolution change, in either direction.

## Root cause (confirmed by breadcrumb)

The breadcrumb watchdog (`phase2/crash-investigation/oxd_bread.bin`, last `OXD_STEP`) froze at **step 5** — the line immediately before `output->createCUDAArray(info, nullptr)` in `OptixDemoTOP::execute()`. The hang is **inside TD's `createCUDAArray` when the output dimensions change**: TD must destroy + reallocate the output Vulkan image and re-export it to CUDA, and that **Vulkan↔CUDA reallocation deadlocks the main thread mid-cook**. Same deadlock *family* as the input-side [`td-cuda-input-freeze`](FREEZE-INVESTIGATION.md), but on the **output** side. In steady state the dims don't change, so `createCUDAArray` returns the cached array instantly (reaches step 91) — only a dimension change triggers the reallocation that hangs.

## Why it cannot be fixed in the C++ TOP

Every in-plugin approach was built and empirically tested this session — **all failed**:

| Attempt | Result |
|---|---|
| `createCUDAArray` **before** `beginCUDAOperations` (the required order) | **step-5 deadlock** on resize |
| `createCUDAArray` **after** `beginCUDAOperations` (move begin earlier) | no freeze, but render goes **BLACK** on resize — the new-size array is never bound as the output, and stays black for all later resolutions. So createCUDAArray **must** be before begin for output binding. |
| Raw `cudaDeviceSynchronize` / `cudaStreamSynchronize` **before** begin | deadlocks the **project load** at step 3 (raw CUDA ops before begin are illegal — SDK: *"all CUDA operations must occur between begin/endCUDAOperations"*). Both `.toe` files hung at the 100 % splash; a vanilla TD project loaded fine, confirming the DLL. |
| `cudaStreamSynchronize(myStream)` at the **end** of every cook (legal, inside begin/end) | still **step-5 deadlock** on a direct resize → proves the deadlock is **internal to TD's output realloc**, not our pending work. |

The bundled reference `phase2/CudaTOP/CudaTOP.cpp` uses the **identical** call pattern with **no** resize handling — it would deadlock the same way; it was simply never tested with runtime resolution changes. No TD doc or forum thread documents a fix.

## The fix (TD-side: pause the COMP around the resize)

Because the deadlock is in TD's output reallocation *during a live cook*, the cure is to **change the resolution while the render is not cooking**. The render TOP can't be paused directly (`allowCooking` is COMP-only), so we pause its **parent COMP** (`PT_Render`).

This is exposed as an **"Update Resolution" button** on the PT_Render wrapper:

1. **Custom params** on PT_Render's **Render** page:
   - `Renderw` (Render Width, Int), `Renderh` (Render Height, Int) — the desired size (no live effect on the output).
   - `Updateres` (Update Resolution, Pulse) — the button.
2. **Parameter Execute DAT** `/project1/optixDemo/oxd_resupdate` (placed **outside** PT_Render so the pause can't stop it), watching `op = PT_Render`, `pars = Updateres`, `onpulse = On`:

```python
def onPulse(par):
    comp = par.owner            # PT_Render COMP
    r = comp.op('render')
    w = max(16, int(comp.par.Renderw.eval()))
    h = max(16, int(comp.par.Renderh.eval()))
    comp.allowCooking = False   # stop the render cooking before changing its output size
    r.par.resolutionw = w
    r.par.resolutionh = h
    run('op(' + repr(comp.path) + ').allowCooking = True', delayFrames=10)  # resume after TD settles
    return
```

While the COMP is paused the render does **not** cook, so TD reallocates the output texture **outside a live cook** (no `createCUDAArray` deadlock) **and** the new-size output binds correctly (no black). The deferred `run(..., delayFrames=10)` re-enables cooking a few frames later.

**Usage:** set Render Width / Render Height, click **Update Resolution**. Do **not** change the render TOP's Common-page resolution directly — that still freezes.

**Verified:** width up (720→1080), height down (1280→720), large jump (1920×1080), restore (720×1280) — every case: **no freeze, no black**, COMP auto-resumes, denoiser (Rr) intact.

## C++ change kept (complementary, not the fix)

`OptixDemoTOP.cpp` now drains the stream at the **end** of every cook (`cudaStreamSynchronize(myStream)` before `endCUDAOperations`, breadcrumb steps 97/98) so each cook is self-contained — when the COMP resumes after a resize, the resume-cook's `createCUDAArray` sees no in-flight work referencing the old output. It does **not** prevent the resize deadlock on its own (the button does) and may be removed if perf-tuning is needed (TD's `endCUDAOperations` likely syncs anyway). `createCUDAArray` stays **before** `begin`.

## Debugging method that cracked it

- **Breadcrumb step** (`oxd_bread.bin`, slot 2) localized each distinct freeze: 5 = `createCUDAArray`, 3 = illegal pre-begin sync, 91 = healthy.
- **Background `.Responding` monitor** caught the exact instant of freeze (a hung MCP call = freeze).
- **`numpyArray` on the C++ CUDA TOP reads garbage/black** — read **downstream regular TOPs** (`tonemap`, `null2`, `level1`) for reliable pixel truth.

## If the button is ever lost (re-create from scratch)

Re-add the three custom params on PT_Render's Render page and the `oxd_resupdate` Parameter Execute DAT (config + callback above). Everything lives in `pathtttracer.92.toe`; if reverting to an earlier `.toe`, re-apply from this doc.
