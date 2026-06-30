# OptixDemoTOP — Performance Optimization Roadmap

**Status:** Research / design only. No implementation. Code-grounded against the current `pt-render-comp` branch.
**Date:** _TBD_
**Target HW:** RTX 4080 Super (Ada, sm_89), OptiX 9.1, TouchDesigner C++ TOP.
**Scope:** the real-time OptiX path tracer in `phase2/OptixDemoTOP/` (megakernel `__raygen__rg` + DLSS-RR).

---

## 0. Where the time goes

A measured frame at the current config:

| Phase | Time | Notes |
|---|---|---|
| **`PT_Render/render` TOP cook** | **14.08 ms** | the work we attack |
| TD / POP / window overhead | ~3.87 ms | fixed, not ours |
| **Total frame** | **~17.95 ms (~55.7 fps)** | |

The TOP cook is honest GPU time because the cook ends with `cudaStreamSynchronize(myStream)` (`OptixDemoTOP.cpp:952`), which serializes CPU/GPU. Inside that 14 ms, three GPU phases run in order on the single `myStream`:

1. **One** `optixLaunch(myPipeline, myStream, …, W, H, 1)` (`OptixDemoTOP.cpp:905`) that does *everything* — primary rays, BSDF, NEE, fog, AOV writes — looped over all `Spp` and `Maxdepth` bounces **inside raygen**. Decomposed: **~10–12 ms** (~78% of the frame, ~85% of the cook), ~0.64 ms/spp.
2. **DLSS-RR** `myRR.evaluate` (`OptixDemoTOP.cpp:915`): 8 AOV array-copies + NGX eval + 1 copy-out. **~1.5–2.5 ms, fixed at output res** — does *not* shrink with the tracer.
3. Output `cudaMemcpy2DToArrayAsync` (`OptixDemoTOP.cpp:938`): **<0.1 ms** (the 8 AOV D2D copies total ~0.08 ms — *not* a cost worth touching).

**Current settings:** 720×1280 (921,600 px), **`Spp` = 18 per frame**, **`Maxdepth` = 4**, fog likely off for this capture, **DLSS-RR running in DLAA** (native, zero upscaling).

**Engine grounding — what the code already does right vs. what is left on the table:**

- ✅ **Pipeline/compile is already near-optimal.** `numPayloadValues=2` (pointer-packed PRD), `numAttributeValues=1`, `exceptionFlags=NONE`, `optLevel=DEFAULT` (-O3), `debugLevel=NONE`, `maxTraceDepth=1` (correct — all traces issue from raygen, CH never re-traces), `PREFER_FAST_TRACE` on both GAS, `DISABLE_ANYHIT` everywhere, all shadow rays use `TERMINATE_ON_FIRST_HIT`. (`OptixDemoTOP.cpp:318–348, 394–434`.) **The cheap pipeline wins are already taken.**
- ✅ **DLAA confirmed** at `RRDenoiser.cpp:93–94`: `InWidth==InTargetWidth==W`, `InPerfQualityValue = NVSDK_NGX_PerfQuality_Value_DLAA`, preset D (`:86–87`), `MVLowRes` flag already set (`:95`). The full upscaling plumbing exists and is unused.
- ❌ **SER absent** — no `optixReorder`, on a highly divergent megakernel (4-way BSDF switch + fog + sky branches).
- ❌ **Surface Russian roulette is dead code** — `if(b>=3)` (`demo_programs.cu:529`) at `Maxdepth=4` fires only on the final iteration, saving nothing.
- ❌ **No specialization / bound-value constants** — every disabled feature still costs branches + register pressure.
- ❌ **`Spp=18`/frame** sits on top of a progressive accumulator (`cu:570–578`) *and* DLSS-RR's neural temporal history — film-grade, not real-time.

---

## 1. The verdict

**Is this the maximum? No — not remotely. You are far from the hardware limit.**

At 14 ms you trace roughly 50–65 M rays + all shading/fog/NEE ≈ ~4–5 Grays/s. A 4080 Super's 3rd-gen RT cores can feed far more. The limiter is **megakernel divergence + brute-force 18 spp** — i.e. *SM-bound shading with under-fed RT cores*, not a BVH or RT-core wall. The current config is **quality-conservative brute force**, with large headroom.

Has NVIDIA "maximized it"? They maximized the *pipeline plumbing* you compiled (see §0) and they ship the *tools* (DLSS-RR upscaling, SER) — but the engine currently **declines** the two biggest ones: it runs DLSS at native DLAA (no upscaling) and never reorders. There is a lot to squeeze.

**Realistic ceiling (honest, with caveats):**

| Stack | Levers | Trace (T) | Frame | fps | Quality |
|---|---|---|---|---|---|
| **Now** | DLAA, 18 spp | ~11 ms | 17.95 ms | 55.7 | reference |
| **Conservative** | DLSS Quality + Spp 18→8 + minor tuning | ~2.1 ms | ~8.1 ms | **~95–125** | essentially indistinguishable |
| **Aggressive** | DLSS Performance + Spp 18→3 + SER | ~0.3 ms | ~6.3 ms | **~140–180** | softer/noisier on glass & fog |

**Caveats — do not multiply gains naively:**

- **There is a hard floor of ~6 ms** = DLSS-RR eval (~2 ms, fixed at output res) + TD overhead (~3.9 ms). Past it, cutting the tracer buys nothing. Total frame is capped at roughly **160–200 fps regardless** of how fast the trace gets.
- **Resolution and spp legitimately multiply** (both scale the same per-sample-per-pixel cost) — but they drive T toward zero *fast*, after which the floor dominates.
- **SER multiplies only the trace kernel.** Once you have already cut T below ~1 ms via res+spp, SER's *absolute* gain is negligible. **Pick a regime:** either keep quality high (DLAA + higher spp) and lean on SER, *or* cut spp/res and skip SER. Do **not** claim `4×·4.5×·1.5×·1.2× ≈ 32×`.
- Lower spp/res shifts variance onto the temporal reconstruction → ghosting/boil on motion, worst on glass refraction and fog. Quality is a real tradeoff, not free.

---

## 2. Lever — DLSS-RR upscaling (biggest gain, medium effort) ⭐ best quality-preserving lever

**What it is.** Today the megakernel renders at full 720×1280 and DLSS-RR runs as native reconstruction + AA (DLAA). Instead, render the path tracer at a *lower internal resolution* and let DLSS-RR reconstruct to 720×1280. RR is purpose-built to upsample noisy ray-traced input — this is the intended use, not a hack. Because T scales ~linearly with internal pixel count, this is the highest-leverage change that keeps quality high.

**Expected gain.**

| Mode | Per-axis | Px fraction | Trace ×factor | Internal res | Frame fps |
|---|---|---|---|---|---|
| DLAA (now) | 1.00 | 100% | 1.00× | 720×1280 | 55.7 |
| Quality | 0.667 | 44.5% | ×0.44 | ~480×854 | ~88 |
| Balanced | 0.58 | 33.6% | ×0.34 | ~418×742 | ~99 |
| Performance | 0.50 | 25% | ×0.25 | 360×640 | ~110 |
| Ultra-Perf | 0.333 | 11% | ×0.11 | ~240×427 | ~135 (not recommended) |

Recommended default sweet spot: **Balanced (~1.8×)**, with Quality for fidelity-critical shots and Performance for max fps. Ultra-Performance is too sparse for PT detail.

**Exact engine hooks.**
- `RRDenoiser::ensure()` `RRDenoiser.cpp:93–94`: split dims — `cp.InWidth/InHeight = renderW/H` (lower), `cp.InTargetWidth/Height = outW/H` (native), set `cp.InPerfQualityValue` from the mode. `MVLowRes` (`:95`) is already correct for upscaling.
- Query `NGX_DLSSD_GET_OPTIMAL_SETTINGS` at output res and use the returned render size **verbatim** (the SDK requires it; Performance/Ultra give non-round sizes).
- Launch `optixLaunch` (`OptixDemoTOP.cpp:905`) at `renderW×renderH`; move all render-res AOV/accumulator allocations (`myImage, myAccum, myAlbedo, myNormal, myRRColor, myRRSpecAlb, myRRRough, myRRDepth, myRRMotion, myRRHitDist, myFlow`) to render res; keep **only** the DLSS output (`myOutArr/myOutSurf`, `myDenoised`) at output res. Reset `myFrameIndex` on render-res change; extend `ensure()`'s cache key from `(W,H)` to `(renderW,renderH,outW,outH)`.
- Expose a `Dlssmode` menu param (DLAA / Quality / Balanced / Performance / Ultra Performance).
- Increase Halton jitter phases from 16 to ~32 (`OptixDemoTOP.cpp:832`) for Performance/Balanced to keep sub-pixel coverage dense.
- Gate the non-RR output paths: in any upscaling mode the OptiX AI denoiser and TAA produce render-res output that cannot fill the output-res TOP — force RR, and fix the transparent-matte restore + final copy (`OptixDemoTOP.cpp:936–938`) which currently assume input==output res.

**Quality / risk.** Medium risk. Re-verify the **MV-scale** (project's historical bug zone — confirm `myFlow` is render-pixel space, not NDC; `MVScale` tied to `flowscale` at `RRDenoiser.cpp:135`) and check camera-motion ghosting + god-ray streaking. Over-smoothing of high-frequency specular/NEE sparkle at low res. Verify glass spheres and fog with numpy A/B + Gemini Vision. A/B preset D vs the newer DLSS-4 transformer preset (verify the exact letter against the shipped `nvngx_dlssd.dll`).

**Effort:** medium (decouple render from output res; touches buffer lifetime).

---

## 3. Lever — Spp per frame 18 → 2–4 (huge gain, trivial effort) ⭐ single highest-ROI change

**What it is.** Raygen renders `nspp = params.spp` full paths every frame and averages them (`cu:540`), *then* feeds that into both a progressive accumulator (`cu:570–578`) and DLSS-RR's neural temporal history. So 18 spp/frame is stacked on top of two existing convergence mechanisms. It is largely wasted:
- DLSS-RR is trained for ~1 spp noisy input and reconstructs from the albedo/normal/roughness/depth guides you already feed.
- In RR mode the input is the per-frame `col`, **not** the accumulator (`rrColor = col`, `cu:587`) — RR's history does the temporal integration.
- Per-frame jitter is **constant across spp** in RR mode (`cu:431`), so extra spp don't even improve sub-pixel AA — the AA comes from jitter *across frames*. Direct evidence that high spp/frame is redundant here.

**Expected gain.** Near-linear in the per-sample portion until the floor:

| Spp | Trace | Frame | fps | vs now |
|---|---|---|---|---|
| 18 (now) | ~11.5 ms | 17.9 ms | 55.7 | 1.0× |
| 4 | ~2.6 ms | ~9.0 ms | ~111 | ~2.0× |
| 2 | ~1.3 ms | ~7.7 ms | ~130 | ~2.3× |
| 1 | ~0.65 ms | ~7.0 ms | ~143 | ~2.6× |

Below ~2–4 spp the **DLSS-RR + TD floor** dominates — spp is a ~2.6× lever, after which the bottleneck shifts to the fixed cost.

**Exact engine hooks.** `inputs->getParDouble("Spp")` → `lp.spp`, consumed by the spp loop `demo_programs.cu:428`. Add a **uniform cut-time spp ramp**: render 6–8 spp the frame `validPrev==0` (the `myFrameIndex` reset at `OptixDemoTOP.cpp:827`), then drop back to default — uniform across the frame, so **no warp divergence**.

**Quality / risk.** Low. 18→4 is essentially transparent in motion and identical when converged; 18→2 is RR's design point; 1 spp shows a brief noise/ghost burst on hard cuts (the ramp fixes this) and slightly more motion lag. Keep the firefly clamp on and tune `fireflyMax` *down* as spp drops (`cu:533–536`, indirect-only, ~zero cost — this is the enabler that keeps low spp clean). **Recommended default: 2 spp (sweet spot), 4 if conservative.**

**Effort:** trivial (one param + ~3-line ramp).

---

## 4. Lever — Shader Execution Reordering (SER) (medium gain, medium effort, pure HW win)

**What it is.** SER is an Ada (sm_89) hardware scheduling feature exposed in OptiX 8+ (you're on 9.1 ✓). As rays bounce they become incoherent — neighboring warp threads hit different geometry/materials and scattered memory, stalling on divergence. SER re-groups threads by a coherence key (hit object's GAS/SBT/instance + optional hint bits) so closest-hit shading and the subsequent material/NEE work run coherently. On pre-Ada it's a no-op (forward/backward safe).

**Why it fits THIS engine.** The main scatter ray (`cu:457`, `OPTIX_RAY_FLAG_NONE`) dispatches into `__closesthit__ch`/`__closesthit__tri`, then branches 4 ways on material `type` (`cu:654–675, 713–739`), each with different memory access — textbook divergence. Path termination is also divergent (RR, fog RR, emissive absorb, miss).

**Expected gain.** ~**10–25% on the 14 ms render TOP (~1.4–3.5 ms)**, concentrated on the primary+scatter trace and CH material work. Narrower than published headlines (Indiana Jones +11% SER-alone → +24% with live-state reduction; Alan Wake 2 +39% *partly OMM*; a glTF tracer +47%) because a large share of your cost is **coherent NEE shadow rays** that SER won't help. Few materials on screen → low end; complex multi-material + glass → high end.

**Exact engine hooks.** Replace the single `optixTrace` **for the main scatter ray only** (`cu:457`) with the split form:
```
optixTraverse(...);                                  // unordered: BVH traversal only
unsigned hint = (b+1u >= params.maxDepth) ? 1u : 0u; // single 1-bit termination hint (Indiana Jones pattern)
optixReorder(hint, 1);                               // reorder by hit-object + 1 hint bit
optixInvoke(...);                                    // ordered: runs __closesthit__ / __miss__
```
Keep the pointer-packed PRD and `numPayloadValues=2`. **Do NOT SER the shadow/occlusion rays** (`sampleSun/sampleEnv/sampleLights/sampleDirect/inscatterFog`, all `TERMINATE_ON_FIRST_HIT`) — already coherent, CH early-outs without material fetch; reordering only adds overhead. **Verify `numAttributeValues` (`OptixDemoTOP.cpp:321`):** builtin-triangle barycentrics via hit objects need 2 — bump from 1 to 2 if UVs/normals read wrong after the change. Use **at most 1 hint bit** (more steals sort bits from the hit object and can net-slow).

**Compounding sub-lever — live-state reduction** (Indiana Jones got the *other half* of its win here, +15%): shrink what is live across `optixReorder` — trim the ~120 B PRD (`cu:25–43`), demote denoiser-safe AOV/radiance accumulators to FP16, recompute cheap values after the reorder instead of carrying them. Raises warp occupancy → multiplies SER's effect.

**Quality / risk.** Medium. The rewrite touches the megakernel hot path — must preserve the DLSS-RR AOV writes, the OptiX-denoiser guides, and the Vulkan↔CUDA interop ordering; a regression breaks *quality*, not just perf. Gains are scene-dependent — measure per-scene, don't assume the headline. FP16 demotion can band AOVs — validate visually. **Only pursue SER in the high-quality regime (DLAA / higher spp); if you've already cut spp+res, its absolute gain collapses — skip it.**

**Effort:** medium (device rewrite of one trace + PTX recompile sm_89; no host pipeline-flag change).

---

## 5. Lever — Russian roulette + Maxdepth (modest gain, trivial effort)

**What it is.** Surface RR is **currently dead code**: `if(b>=3){ … if(rnd>p) break; }` (`cu:529`) at `Maxdepth=4` means `b` loops 0–3 and RR fires only at `b==3` — the last iteration, after the scatter, right before the loop exits anyway. **It never saves a single trace today.** To get its benefit, lower the start to `b>=1` or `b>=2` so it can cull low-throughput paths *before* tracing the next bounce. (Fog RR at `cu:496` is already correct/unbiased.)

**Expected gain.** Modest at depth 4 (~**5–15%**) — little depth to cull. The real payoff: RR makes higher `Maxdepth` affordable, so you can raise depth to 6–8 for better GI at roughly flat cost, *or* drop `Maxdepth` 4→3 (4→2 for a hero shot) since sun/env NEE captures most direct light.

**Exact engine hooks.** Change the threshold at `demo_programs.cu:529` and expose it; keep the `max(p, 0.05)` floor. `Maxdepth` is `inputs->getParDouble("Maxdepth")` → bounce loop `cu:452`.

**Quality / risk.** Low. RR adds path variance (more raw noise), absorbed by RR + accumulation — but could show on the first post-cut frame if combined with 1 spp. Lowering Maxdepth darkens indirect/GI; verify NEE still carries direct light so scenes don't lose energy.

**Effort:** trivial.

---

## 6. Lower-priority / conditional levers

- **Stochastic single-light NEE (conditional on light count).** `sampleLights` (`cu:205–247`, called per diffuse bounce) loops **all** `numPtLights` casting one shadow ray each — O(N). Replace with a power-CDF single-light pick (reuse the existing `lightCDF` from the sphere path) → O(1) shadow rays/bounce. Negligible at 1–2 lights, multi-× at 8+. This is the natural pre-ReSTIR DI step. Effort low-moderate; gain scales with scene light count.
- **Fog defaults (only when fog on, then large).** Prefer `fogSingleScatter=1` (`cu:491`), `fogMaxScatter=1–2` (`cu:493`), raise `fogRRStart`, set `fogSkyStr=0` to drop the env in-scatter ray. All already exposed — guidance/defaults, not code. Biased (loses multi-bounce glow in thick fog) but imperceptible in thin haze. Fog adds ~2–4 ms and much worse low-spp noise — it may need its own higher spp floor.
- **GAS compaction + stack trim (small, free-ish).** Sphere GAS (`OptixDemoTOP.cpp:396`) skips `ALLOW_COMPACTION` — adding it mainly saves VRAM + minor cache locality (low-single-digit trace gain). Compute the exact stack size instead of the hardcoded `8192` continuation (`:348`) for a small occupancy bump. Low priority.
- **Specialization / bound-value constants — SKIP for now.** Binding `fogEnable/neeEnable/skyMode/numPtLights/maxDepth` via `OptixModuleCompileBoundValueEntry` would dead-strip disabled features (~5–15%), but most are runtime TD toggles and bound values force a multi-hundred-ms module recompile on every change — a cook-time hitch, bad for live use. Low ROI.
- **OMM / DMM — SKIP (irrelevant).** All geometry is opaque (`DISABLE_ANYHIT` at `:394, :428`); no alpha cutouts or displacement → Ada micromaps give zero benefit. (This is why your SER number won't match Alan Wake 2's +39%, which was partly OMM.)
- **Per-pixel adaptive sampling — SKIP.** Causes warp divergence on the megakernel and duplicates what DLSS-RR already does. The only adaptive worth doing is the uniform cut-time spp ramp (§3).
- **Megakernel → wavefront rewrite — out of scope.** The architectural ceiling; big-but-expensive, not a "lever in the code."

**Possible existing bug to verify (orthogonal to perf):** OptiX 9.1's builtin sphere intersector now reports **front-face hits only** (Blackwell alignment). The glass-sphere refraction path (`cu:725–733`) relies on **back-face** hits for the exit interface — dielectric spheres may already be broken on 9.1. Verify; the fix needs a custom sphere intersector. Triangle geometry is unaffected.

---

## 7. Measurement plan

**Profile BEFORE and AFTER every change — the levers compound noise and the floor hides gains, so never assume.**

**CUDA events (true GPU ms, async-safe).** Place `cudaEventRecord(ev, myStream)` pairs *inside* the begin/endCUDAOperations block, then `cudaEventSynchronize` + `cudaEventElapsedTime` after the existing end-of-cook sync (no extra stalls). Brackets:
- `t_trace` around `optixLaunch` (`:905`) — trace+NEE+fog fused.
- `t_rr` around `myRR.evaluate` (`:915`); sub-bracket the 8 AOV copies vs `NGX_CUDA_EVALUATE_DLSSD_EXT` to separate copy overhead from tensor-core eval.
- `t_copy` around the output copy (`:938`).
Surface them as **Info CHOP channels** (you already expose 13 at `getNumInfoCHOPChans`) — live in TD, zero tooling.

**Separating trace vs NEE vs fog (they are ONE kernel — events can't split them).** Use differential A/B toggling:
- **Per-sample vs fixed:** measure `t_trace` at Spp=1 and Spp=18 → `t_trace ≈ fixed + spp·perSample`. Slope×18 = brute-force cost; intercept = BVH/setup.
- **NEE:** toggle `Nee` off → Δ = NEE cost.
- **Fog:** toggle `Fogenable` off → Δ = fog cost (largest single Δ when on).
- **DLAA vs none:** switch `Denoiser` Rr→none → cross-check on `t_rr`.

**Profilers.** Nsight Systems on `TouchDesigner.exe` for the timeline + GPU idle gaps + the present/interop gap. Nsight Compute on the single megakernel — the three numbers that decide strategy: **achieved occupancy**, **warp-state stalls (divergence / long-scoreboard)**, **RT-core (TTU) active %**. Expect SM-bound / divergent with under-fed RT cores → that confirms the limiter is megakernel brute force, not BVH, and is your SER + low-spp signal. Nsight Graphics is the wrong tool (the heavy work is CUDA/OptiX). OptiX has no per-shader GPU timer.

**The truth about the end-of-cook `cudaStreamSynchronize` (`:952`).** When GPU-bound (you are, 78%), it costs **little real frame time**: the output is consumed by TD's Vulkan present in the same frame via CUDA↔Vulkan interop, so TD must wait on a semaphore signaled after your final copy regardless. The sync only **moves the wait earlier** and forfeits overlap of the *next* cook's non-dependent CPU prep — realistically 1–3 ms recoverable, and only if TD pipelines cooks. **Keep it for measurement** (it makes the 14 ms honest and serves resize-safety). Then A/B **Total Frame Time** (not TOP cook time, which becomes meaningless without the sync) with it removed; keep it if Δ < ~1 ms. Do not expect it to be the win, and do not ship its removal without re-validating the resize path.

---

## 8. Phased, risk-ordered rollout

**Phase 0 — Profiling harness (do FIRST, prerequisite for everything).**
Add the CUDA-event brackets + Info CHOP channels (§7). No render change. This is how you'll prove each lever and catch quality regressions. **Touches:** none of the denoiser/NEE/interop logic — safe.

**Phase 1 — Free wins (trivial, low risk).** _Stack one at a time, verify with numpy A/B + Gemini Vision between each._
1. **Spp 18 → 4 (then → 2)** + the cut-time ramp (§3). Tune `fireflyMax` down. ~2× fps. _Risk: low (motion noise)._
2. **DLSS-RR upscaling → Balanced default** + `Dlssmode` param (§2). ~1.8×. _Risk: medium — touches the **denoiser path, buffer lifetimes, MV-scale, jitter**; re-verify ghosting and the matte/output-copy fix (`:936–938`)._
3. **Fix Russian roulette** start `b>=1`/`b>=2` + expose Maxdepth (§5). ~5–15%, enables higher GI. _Risk: low._

After Phase 1 the conservative-stack target (~95–125 fps at near-identical quality) is reached with no megakernel rewrite. **Decide the regime here** (§1): if quality is paramount, stay in DLAA/higher-spp and proceed to Phase 2; if max fps, you may already be floor-bound and Phase 2 buys little.

**Phase 2 — SER + live-state reduction (medium effort, medium risk).** Only worthwhile in the high-quality regime (T still large). Implement `optixTraverse/optixReorder/optixInvoke` on the **main scatter ray only** (§4), bump `numAttributeValues` to 2 if needed, trim live state / FP16 AOVs. ~10–25%. **Touches the megakernel hot path → DLSS-RR AOVs, OptiX denoiser guides, and Vulkan↔CUDA interop ordering — highest-risk change; profile per-scene, don't assume the headline.**

**Phase 3 — Tuning + conditional (low priority).** Stochastic single-light NEE (only if many lights), GAS compaction + stack-size trim, fog perf defaults. Re-measure the end-of-cook sync as a Total-Frame-Time A/B (do not ship removal without resize re-validation).

**Risk flags — changes that touch denoiser / NEE / interop:** §2 (DLSS path, buffers, MV/jitter), §4 (megakernel rewrite, AOV writes, interop ordering), §6 single-light NEE (NEE noise), end-of-cook sync removal (resize deadlock + async writes crossing cook boundaries). The profiling harness (§0/§7) and the Gemini-vision / pixel A/B workflow are mandatory gates on all of these.

---

## 9. Sources

**Engine code (this repo):**
- `phase2/OptixDemoTOP/demo_programs.cu` — `:411` raygen, `:428` spp loop, `:452` bounce loop, `:457` main scatter trace, `:516–523` NEE fan-out, `:205–247` `sampleLights` O(N), `:114–143` `sampleDirect` CDF, `:529` dead surface RR, `:533–536` firefly clamp, `:570–578` accumulator, `:587` `rrColor=col`, `:336–395`/`:478–497` fog, `:586–595` RR AOVs, `:654–742` CH material divergence, `:725–733` glass refraction, `:25–43` PRD.
- `phase2/OptixDemoTOP/OptixDemoTOP.cpp` — `:318–348` pipeline/module/link compile options, `:394–434` GAS build/refit, `:456` Spp/Maxdepth params, `:827` frameIndex reset, `:831–834` Halton jitter, `:905` optixLaunch, `:911–938` RR/denoiser branch + matte + output copy, `:952` end-of-cook sync.
- `phase2/OptixDemoTOP/RRDenoiser.cpp` — `:86–95` DLAA preset + create flags + MVLowRes, `:93–94` `InWidth==InTargetWidth`, `:109–145` evaluate (8 AOV copies + NGX eval + copy-out), `:135` MVScale=flowscale.
- `phase2/OptixDemoTOP/LaunchParamsDemo.h` — `:32–153` launch-param fields (spp/maxDepth/fireflyMax/fog controls).

**External:**
- NVIDIA — Path Tracing Optimization in Indiana Jones: SER + Live-State Reductions: https://developer.nvidia.com/blog/path-tracing-optimization-in-indiana-jones-shader-execution-reordering-and-live-state-reductions/
- NVIDIA — Improve Shader Performance with Shader Execution Reordering: https://developer.nvidia.com/blog/improve-shader-performance-and-in-game-frame-rates-with-shader-execution-reordering/
- NVIDIA SER whitepaper: https://d29g4g2dyqv443.cloudfront.net/sites/default/files/akamai/gameworks/ser-whitepaper.pdf
- OptiX 9 Programming Guide: https://raytracing-docs.nvidia.com/optix9/guide/index.html
- OptiX forums — what does optixReorder reorder: https://forums.developer.nvidia.com/t/what-does-optixreorder-reorder/319282
- OptiX forums — 9.1 SER builtin-sphere reported hits (front-face change): https://forums.developer.nvidia.com/t/optix-9-1-ser-builtin-sphere-intersection-reported-hits/357211
- NVIDIA DLSS SDK headers (DLSSD create params / GET_OPTIMAL_SETTINGS): https://github.com/NVIDIA/DLSS — `include/nvsdk_ngx_helpers.h`, `nvsdk_ngx_params.h`, `nvsdk_ngx_defs.h`
- Streamline DLSS programming guide: https://github.com/NVIDIAGameWorks/Streamline/blob/main/docs/ProgrammingGuideDLSS.md
- DLSS-RR overview: https://deepwiki.com/NVIDIA/DLSS/3-dlss-ray-reconstruction
- Integrate DLSS 4 with Streamline: https://developer.nvidia.com/blog/how-to-integrate-nvidia-dlss-4-into-your-game-with-nvidia-streamline/
- Rémi Adriaensen — Fast Path Tracing with DXR (RR + warp-coherent RR + SER measurements): https://medium.com/@remi.adriaensen/fast-path-tracing-with-dxr-ac22161b1876
- AMD GPUOpen — Neural supersampling & denoising for real-time path tracing (1-spp design point): https://gpuopen.com/learn/neural_supersampling_and_denoising_for_real-time_path_tracing/
- NVIDIA — SVGF (spatiotemporal variance-guided filtering): https://research.nvidia.com/publication/2017-07_spatiotemporal-variance-guided-filtering-real-time-reconstruction-path-traced
- Chips and Cheese — SER: NVIDIA Tackles Divergence: https://chipsandcheese.com/p/shader-execution-reordering-nvidia-tackles-divergence
- Khronos — VK_EXT_ray_tracing_invocation_reorder (SER): https://www.khronos.org/blog/boosting-ray-tracing-performance-with-shader-execution-reordering-introducing-vk-ext-ray-tracing-invocation-reorder
- TechSpot — DLSS 4 Ray Reconstruction analysis: https://www.techspot.com/article/2951-nvidia-dlss-4-ray-reconstruction/
