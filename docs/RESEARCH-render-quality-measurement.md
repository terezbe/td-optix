# Render-Quality Measurement — How to Objectively Judge Our Renders

**Status:** Proposed methodology / ready to adopt
**Date:** 2026-06-26
**Applies to:** `OptixDemoTOP` (OptiX 9.1 + CUDA 12.8, DLSS-RR), wrapped by `PT_Render`, read from the downstream tonemap chain (`PT_Render/tonemap` → `null2`/`level1`)

> **Why this document exists.** We have repeatedly judged render quality *by eye*, and it has burned us. The canonical failure: the agent looked at a **static** frame, saw no motion shimmer, and called it "clean" — but the frame was full of stable spatial grain (and in other cases over-blurred by DLSS-RR). Our one existing objective metric, **per-pixel temporal standard deviation under a fixed camera-motion path**, is *blind to anything that does not change between frames*: a noisy-but-stable frame reads ≈ 0 temporal-std, and an over-blurred frame reads *better* than a crisp one. Eyeballs plus a single stability metric cannot see static spatial noise, distance-from-converged, or denoiser over-blur. This document replaces "it looks clean" with numbers, and it exploits the one asset we have that a film renderer would envy: **our engine can render its own converged ground truth** via progressive accumulation. That turns "no reference" into a standard full-reference measurement problem for every static view.

---

## 0. What "render quality" actually decomposes into

"Quality" is not one axis. It is at least four, and each needs a *different kind* of metric. Our temporal-std covers exactly one of them.

| Axis | What it is | Symptom | Right kind of metric | Our prior coverage |
|---|---|---|---|---|
| **Spatial noise / convergence** | residual Monte-Carlo grain; distance from the fully-converged image | a *stable* frame that is still grainy; fireflies | **full-reference** vs a converged reference (relMSE, FLIP, PSNR-vs-spp curve), or **no-reference** noise-σ | ❌ blind |
| **Sharpness / detail (over-blur)** | denoiser eats high-frequency detail | DLSS-RR "clean" but soft / smeared texture | **structural** FR (MS-SSIM, FLIP feature term) + **sharpness** (variance-of-Laplacian *ratio* vs ref) | ❌ blind |
| **Temporal stability** | frame-to-frame instability under motion | boiling / flicker / shimmer while orbiting | **motion-compensated** temporal error; perceptual video metric (ColorVideoVDP) | ✅ temporal-std (the only thing we had) |
| **Perceptual / semantic error** | "does it look *right*" — wrong color, missing reflection, ringing, melted geometry | artifact a scalar can't name | perceptual FR map (**FLIP**) + **Gemini Vision** as the semantic tiebreak | partial (Gemini) |

The throughline: **noise and sharpness are the same high-frequency energy seen from two sides.** A noisy frame scores *high* on a sharpness measure (looks "sharp"); genuine detail reads as "noise" to a blind noise estimator. So you never read a single high-frequency number as "quality" — you measure **noise in flat regions** and **sharpness on edges** and read them as a pair. That pairing is what makes the panel diagnostic instead of misleading.

---

## 1. Full-reference metrics (vs a converged reference)

These are the gold standard, and they are *available to us* because of progressive accumulation (see §3 and §5 for how we manufacture the reference). All operate on two same-pose images that exit the **same** downstream TOP with **identical** tonemapping/exposure/resolution.

### relMSE — the convergence scalar
`relMSE = mean( (test − ref)² / (ref² + eps) )`, `eps = 1e-2` (Rousselle convention; report it always).
- **Catches:** residual MC noise / distance-from-converged, weighted by reference brightness so dark regions count as the eye sees them. This is *the* de-facto scalar in Monte-Carlo denoiser papers precisely because raw MSE is dominated by highlights/fireflies.
- **Misses:** structure — it is per-pixel and structure-blind. The `eps` and the asymmetric `ref²` denominator shift values (the asymmetric form over-penalizes test-side fireflies near dark reference pixels; a symmetric `(ref²+test²)/2` is an alternative). Keep eps + convention fixed across an A/B.
- **Tool:** ~3 lines of numpy (no package). Variant: **SMAPE** `mean(|t−r|/(|t|+|r|+eps))`, bounded 0..1, used in Disney/NVIDIA denoiser papers.

### PSNR — sanity number / convergence curve only
`20·log10(MAX) − 10·log10(MSE)`.
- **Catches:** raw intensity error; great as a **convergence tracker** — plot PSNR-vs-spp; it rises and plateaus, and the plateau defines "converged enough."
- **Misses:** perception. Dominated by a few bright pixels, structure-blind; an over-blurred frame can out-score a crisp slightly-noisy one. **Never** make the quality decision on PSNR alone.
- **Tool:** `skimage.metrics.peak_signal_noise_ratio(ref, test, data_range=1.0)`.

### SSIM / MS-SSIM — the over-blur / detail detector
Structural similarity (luminance × contrast × structure over local windows); MS-SSIM weights 5 scales.
- **Catches:** local structure/contrast loss — **this is our DLSS over-blur detector.** Smearing detail drops MS-SSIM even when MSE barely moves, because it kills local contrast. MS-SSIM is preferred here (multi-scale → frequency-aware).
- **Misses:** *where* (single global number); computed on tonemapped data; not a color/perceptual metric.
- **Tool:** SSIM in `skimage.metrics.structural_similarity(ref, test, channel_axis=-1, data_range=1.0)`. **MS-SSIM is NOT in scikit-image** → `pytorch-msssim` (`ms_ssim(t,r,data_range=1.0)`, NCHW) or `piq.multi_scale_ssim`. Report `1 − MS-SSIM` if you want "lower = better."

### LPIPS — secondary perceptual cross-check
Distance in deep-net (AlexNet/VGG) feature space, calibrated to human similarity.
- **Catches:** learned perceptual texture/blur/structure differences MSE misses; *the* loss if we ever train/tune a denoiser.
- **Misses:** it's trained on **natural-image** distortions (JPEG/blur/noise), **not** HDR/MC render grain or alternating-image artifacts; needs PyTorch + GPU model, RGB in **[−1,1]**; not physically interpretable. **Secondary to FLIP.**
- **Tool:** `pip install lpips`; `lpips.LPIPS(net='alex')`.

### NVIDIA FLIP — the PRIMARY perceptual metric (built for exactly our problem)
FLIP (Andersson et al., *Ray Tracing Gems II* / ACM CGIT 2020) models the perceptual difference a human notices when **flipping back and forth** between a render and its reference — literally our A/B-on-the-same-pose workflow. It runs a color/chroma pipeline (CIELAB + spatial contrast-sensitivity filtering) **and** a separate **feature term** for edges/points, so one metric catches both *color/noise* and *edge/detail (sharpness)* differences.
- **Catches:** residual MC noise, fireflies, denoiser over-blur, ghosting — where PSNR is blind and SSIM only partially sensitive. Beat PSNR/SSIM on human correlation in NVIDIA's study. Accounts for viewing conditions via **pixels-per-degree (ppd)** (default ~67; set it from our monitor resolution + viewing distance — this is what makes it a *calibrated* judgment instead of an uncalibrated eye).
- **Output is a deliverable:** a per-pixel error map in [0,1] **plus** a mean/weighted-median/quartile summary. The error map gives us the **3×5 localization grid** we already want — per-pixel and perceptual.
- **LDR vs HDR:** **LDR-FLIP** takes two tonemapped 0..1 images → **use this by default** on our downstream read. **HDR-FLIP** takes linear HDR, internally sweeps exposures, tonemaps each, runs LDR-FLIP, takes per-pixel max → use it **if/when** we tap the pre-tonemap linear buffer (reports error visible at *any* exposure).
- **Misses:** still full-reference (needs the converged reference); HDR-FLIP's exposure sweep is slower; Python API omits some CLI outputs.
- **Tool:** `pip install flip-evaluator`. Python: `import flip_evaluator as flip; errMap, meanFLIP, params = flip.evaluate(ref, test, "LDR")` (numpy HxWx3 float32 0..1 or file paths; `"HDR"` for linear). CLI: `flip -r ref.png -t test.png` writes the error-map PNG.

---

## 2. No-reference (live frames)

When a converged reference is impractical — live exploration, or a frame you can't pair to ground truth. These attack our two blind spots (static grain, over-blur) directly, but they are **relative A/B trend signals, never absolute verdicts.** Operate on tonemapped luminance; **`np.flipud` the bottom-up TD array first.**

### Blind noise (the "stable but grainy" detector)
1. **scikit-image `estimate_sigma`** (wavelet-MAD / Donoho-Johnstone): `from skimage.restoration import estimate_sigma; estimate_sigma(img, channel_axis=-1, average_sigmas=True)`. Robust σ from finest wavelet detail coefficients. A noisy *stable* static frame (temporal-std ≈ 0) now reads a real σ. Track it as the frame accumulates → a **convergence curve with no reference**.
2. **Immerkær fast σ** (3×3 Laplacian mask, ~5 lines `scipy.signal.convolve2d`): near-zero cost, second-derivative kills smooth ramps → the cheap real-time noise needle.
3. **Flat-patch residual std (recommended, de-confounded):** build a low-gradient mask (Sobel below a percentile), high-pass (subtract small Gaussian blur), take the std of the high-pass residual **only inside the flat mask**, reported on the **3×5 grid**. The most honest single-frame static-noise number, and it reuses our grid harness.

### Blind sharpness (the DLSS over-blur detector)
4. **Variance of Laplacian** — `cv2.Laplacian(luma, cv2.CV_64F).var()`; high = sharp.
5. **Tenengrad / Sobel energy** — `mean(gx²+gy²)`; more robust than Laplacian.
6. **Spectral slope / HF-energy ratio** — FFT log-power vs log-radial-frequency; steeper negative slope or lower high-band ratio = blurrier. Catches *global* softening local operators miss.

### Deep / NSS no-reference (relative only)
Via **`pyiqa`** (`pip install pyiqa`, runs on our CUDA GPU): `pyiqa.create_metric('niqe'|'brisque'|'musiq'|'clipiqa'|'topiq_nr', device='cuda')`. Also `piq.brisque` (lighter). MUSIQ/CLIP-IQA/TOPIQ-NR correlate with human opinion far better than NIQE/BRISQUE.

**Honest limit (well-supported in the literature):** NIQE/BRISQUE/IL-NIQE priors are fit to **natural photographs**; on synthetic + tonemapped renders they *misalign* — they cannot reliably separate natural from non-natural, and tonemapping/contrast shifts move their scores independent of true quality. Our frames are exactly that domain-shift case. **Use Tier-3 only as same-scene A/B deltas, cross-validated against Tier-1/2 + the accumulation reference — never as an absolute "this render is 4.2/good."** And remember the noise↔sharpness coupling: a noisy frame reads "sharp," detail reads as "noise." Only the flat-mask-noise + edge-sharpness pair is diagnostic.

---

## 3. Convergence + temporal stability

### Equal-time vs equal-sample (which A/B to run)
- **Equal-sample:** both conditions get the same spp. Isolates algorithmic quality-per-sample — use to compare two denoisers/sampling strategies on the science.
- **Equal-time:** both get the same wall-clock (same ms/frame or fps). **This is the primary A/B for us** — a real-time engine's user feels frame time, not sample count. Lock frame time / fps, then compare relMSE+FLIP (static) and motion-comp residual / ColorVideoVDP (motion). Report equal-sample too when you want to understand *why*.

### Convergence to reference
Plot **PSNR-vs-spp** and **relMSE-vs-spp** as accumulation climbs; the **plateau defines "converged enough."** Detect convergence automatically as the relMSE between successive accumulation checkpoints falling below ~1e-4 (the image stops changing). The no-reference σ curve (§2) gives the same "is it done" signal with no reference at all — this mirrors how Arnold/RenderMan/V-Ray auto-stop sampling on a per-pixel variance/noise threshold.

### Motion-compensated temporal error — the principled upgrade of temporal-std
Our per-pixel temporal-std on a fixed motion path is a legitimate **flicker** measure but (a) counts legitimate motion as error and (b) only works when both conditions follow the *exact* same path. The field's principled version: **reproject frame t−1 into t, then diff against the actual frame t** — real motion cancels, only true flicker/boiling remains.

`WE = mean( |frame_t − warp(frame_{t−1})| )` over non-disoccluded pixels.

**We have a big advantage: the engine already computes per-pixel motion vectors for DLSS-RR.** Warping with those MVs is an exact, cheap, correct reprojection — better than estimated optical flow. Mask disocclusions (MV points off-frame, or forward/backward flow disagree). Fallback without engine MVs: `cv2.calcOpticalFlowFarneback` (or RAFT `torchvision.models.optical_flow.raft_large` for quality). Report **both** raw temporal-std and the motion-comp residual, each as an overall number **and** a 3×5 grid.

Optional principled extras when a high-spp **reference sequence** exists (render the same path at high spp):
- **Relational Warping Error (RWE):** subtract the reference's own warp-diff so flow error + natural inter-frame change don't pollute the number.
- **tOF / tLP (TecoGAN, ToG 2020):** tOF compares optical flow of test-seq vs ref-seq; **tLP** = `|LPIPS(test_{t-1},test_t) − LPIPS(ref_{t-1},ref_t)|`, the principled lightweight "perceptual boiling" number.
- **ColorVideoVDP** (`pip install cvvdp`, JOD 0..10, 10 = identical) — the single most authoritative off-the-shelf temporal metric; it models spatial **and temporal** and color vision and is one of the only metrics that genuinely predicts *flicker* (most "temporal" metrics use a 2-frame window and can't distinguish temporal frequencies). Feed test-path vs high-spp reference-path at our real fps. `pycvvdp.cvvdp(display_name='standard_4k').predict(test, ref, dim_order="FCHW", frames_per_second=50)`. FovVideoVDP is the alternative.

> cvvdp / FovVideoVDP / LPIPS need torch + GPU and add real runtime — they're for periodic deep A/B checks, not the per-frame live loop. Keep the live loop on cheap numpy metrics (relMSE, warp-error, estimate_sigma).

---

## 4. How the industry does it

- **NVIDIA FLIP** is the de-facto render image-diff in graphics research and renderer teams — purpose-built for "alternate between render and ground truth and see what pops," HVS-modeled, beats PSNR/SSIM on human correlation, ships an error map. This is why it's our primary perceptual metric.
- **OpenImageIO `idiff` / `oiiotool --diff`** is the tool renderer / OpenEXR / OSL CI suites actually use to gate builds: reads EXR natively, reports mean/RMS/PSNR/max error with pixel location, has a perceptual mode (`-p`, Hector Yee), threshold flags, and **CI exit codes** (`0` pass / `1` warn / `2` fail / `3` size-mismatch / `4` file error). Example gate: `idiff -fail 0.004 -failpercent 10 -hardfail 0.25 -p A.exr B.exr`. `pip install OpenImageIO` ships the CLI + Python API. (`perceptualdiff`/pdiff is the same family — `idiff -p` subsumes it.)
- **Renderer adaptive-sampling noise thresholds:** Arnold (`AA_adaptive_threshold` ~0.01–0.05), RenderMan (per-pixel **variance** error metric is the default, can include all AOVs), V-Ray (noise threshold) all literally measure **per-pixel noise/variance to auto-stop sampling.** The transferable move for us: expose a **per-pixel variance / standard-error-of-the-mean "estimated noise" AOV/scalar** from our accumulation buffer — a reference-free convergence/noise meter that says "still at σ=0.03, keep accumulating," the principled fix for "called a noisy frame clean."
- **Automated image-regression CI:** render A and B on the identical path → `idiff` (or our `measure_render`) → fail the change on exit==2 / threshold breach. Reference frames are baked per scene+camera and reused.
- **Video-level temporal QC (Netflix VMAF):** fuses spatial (VIF/DLM) + temporal (motion) into one MOS-like score — right for a rendered *clip* on our fixed path vs a converged-path clip (`ffmpeg -i distorted -i reference -lavfi libvmaf`). Caveat: trained on broadcast content, real-time CG is out-of-distribution → **relative A/B only**, harmonic-mean pooling, and follow Netflix's rule: confirm the deficit is visible in real-time playback before acting.
- **Human-eye + tool hybrid:** the numbers are the decision; the trained eye (here, **Gemini Vision**) is the semantic gate for "is anything categorically wrong" that a scalar can't name. There is no rendering-specific SMPTE noise standard; the perceptual metrics descend from HVS contrast-sensitivity models (Barten/Daly CSF in FLIP, VIF/DLM in VMAF). Subjective-study standards (ITU-R BT.500 / ITU-T P.910) matter only if we ever run human studies.

---

## 5. THE PROTOCOL for this project (the actionable part)

Anchor *everything* to a self-rendered converged reference. Read frames from a **downstream regular TOP** (`PT_Render/tonemap` → `null2`/`level1`) via `numpyArray(delayed=False)` — the C++ CUDA TOP returns garbage if read directly — and `np.flipud` (TD is bottom-up) **identically** for test and reference before any metric or save.

### 5.1 Generate an in-engine converged reference
```
denoiser = None                       # PT_Render Denoise page; pure MC, no DLSS bias
freeze camera (null5 static); reset accumulation
loop:
  advance op('/').time.frame in-script so accumulation cooks (NOT a sleep loop — that freezes the main thread)
  every K=128 frames: grab buf_k (float32, flipud)
  if relMSE(buf_{k-1}, buf_k) < 1e-4: converged → stop
  hard ceiling ~4000–8000 accumulated spp
save reference.npy  (float32; .npy avoids EXR/OIIO install pain on Windows)
```
- Render the reference at the **DLSS OUTPUT resolution** (post-upscale equivalent) so FR metrics are pixel-aligned — a resolution mismatch is an automatic fail you can't blame on noise.
- Key the reference by **camera-transform hash** so it's reused across many A/B tests on that view.
- Optionally also bake a **`ref_dlss_settled`** (DLSS frame after it stops changing) when you want to grade against the *denoised target* rather than pure MC, and — if you can tap the pre-tonemap linear buffer — a linear `.exr` for HDR-FLIP / linear relMSE.
- **Verify the reference itself plateaued** (PSNR/relMSE-vs-spp flattened) before trusting any comparison. A too-few-spp "truth" biases every downstream number.

### 5.2 Static-quality check (the axis the agent missed)
Score the real-time / DLSS frame against `reference.npy` with **four FR metrics — never one** — plus the FLIP grid:

| Metric | Tool | Catches | "Good" |
|---|---|---|---|
| **FLIP mean (LDR)** + 3×5 grid | `flip_evaluator` | perceptual: grain + structure + over-blur, *localized* | < 0.08 |
| **relMSE** (eps 1e-2) | numpy | convergence / residual MC noise | < 5e-3 |
| **MS-SSIM** (or SSIM) | `piq` / `pytorch-msssim` / `skimage` | detail loss / over-blur | > 0.97 |
| **Sharpness ratio** = HF(test)/HF(ref) | numpy Laplacian-var | **noisy vs over-blurred disambiguation** | 0.85–1.15 |
| LPIPS (optional) | `piq` / `lpips` | learned perceptual texture loss | < 0.06 |

The **sharpness ratio** is the new axis that breaks the noisy/over-blurred tie FLIP+SSIM smear together:
- **CONVERGED & FAITHFUL:** relMSE<5e-3, FLIP<0.08, SSIM>0.97, sharp_ratio 0.85–1.15, NR-σ low.
- **NOISY** (the mis-called-"clean" case): relMSE high, FLIP>0.1, **sharp_ratio ≳ 1.2** (excess HF = grain, not detail), NR-σ high.
- **OVER-BLURRED** (DLSS-RR ate detail): **sharp_ratio < 0.75**, MS-SSIM down (esp. detailed tiles), NR-σ low, FLIP moderate.

Always emit the **3×5 FLIP error-map grid** to localize *where* it's wrong (the reflective sphere? only the HDRI background?).

### 5.3 Live no-reference panel (no converged GT available)
`estimate_sigma` (+ Immerkær cross-check, + flat-mask residual on the 3×5 grid) for grain; Laplacian-var / Tenengrad / spectral-slope for over-blur (read **relative**); optional pyiqa NIQE/BRISQUE/MUSIQ as **same-scene A/B deltas only**. NR thresholds (0..1, calibrate per scene against the reference's own σ): σ < 0.005 clean · 0.005–0.02 mild · > 0.02 visibly noisy.

### 5.4 Temporal-stability check
Keep per-pixel **temporal-std on the identical fixed motion path** (cheap boiling A/B). Upgrade to **motion-compensated residual** (warp t−1→t with engine DLSS MVs if tappable, else Farneback; fwd/bwd occlusion mask). Report both as overall + 3×5 grid. For a headline perceptual temporal number on a clip, run **ColorVideoVDP (JOD)** test-path vs high-spp reference-path; **tLP** is the lightweight alternative. Thresholds (0..1): mean residual/std < 0.01 stable · 0.01–0.03 some boiling · > 0.03 boiling. Both arms require the path be **stepped identically** between A and B (advance `time.frame` in-script) or the comparison is invalid.

### 5.5 Recommended thresholds (LDR, 0..1; calibrate per scene against the reference's own NR-σ + HF energy)
- **FLIP mean:** <0.05 excellent · 0.05–0.08 good · 0.08–0.15 noticeable · >0.15 bad
- **relMSE:** <1e-3 converged · 1e-3–5e-3 good · 5e-3–5e-2 noisy · >5e-2 very noisy
- **SSIM:** >0.98 excellent · 0.95–0.98 good · 0.90–0.95 soft/noisy · <0.90 bad (MS-SSIM ~0.01 higher)
- **Sharpness ratio (test/ref HF):** 0.85–1.15 faithful · <0.75 over-blur · >1.2 residual grain
- **NR σ:** <0.005 clean · 0.005–0.02 mild · >0.02 visibly noisy
- **Temporal mean / motion-comp residual:** <0.01 stable · 0.01–0.03 some boiling · >0.03 boiling
- **LPIPS:** <0.06 good · >0.2 bad

### 5.6 When to ALSO use Gemini Vision
Gemini Vision (`tools/gemini_vision.py`) is the **semantic tiebreak, never the primary number.** Trigger it when: the verdict is **MARGINAL**, FR and NR **disagree**, or there's an artifact a scalar can't name — fireflies, ringing/halos around edges, color/tonemap shift, melted reflections, missing geometry. The decision stays the numbers.

### 5.7 The harness
All of the above is one reusable call — see the `harness` field: `measure_static(img, ref)`, `measure_noref(img)`, `measure_temporal(frames)`, with thresholded pass/fail dicts. `pip install flip-evaluator scikit-image piq opencv-python imageio numpy` (optional `pyiqa lpips torch cvvdp`; `OpenImageIO` only if you want true EXR/CI `idiff`).

---

## 6. Pitfalls

- **Tonemapping / HDR confound.** Our read is **tonemapped, display-referred 0..1**, so relMSE/PSNR/FLIP run in tonemapped space (perceptually reasonable — it's what the user sees — but **not** scene-linear). Use **LDR-FLIP**; relMSE/HDR-FLIP are most canonical in **linear HDR**. **Never mix domains** — don't compare a linear number against a tonemapped one, and feed *both* images the same tonemapped pixels. Tap the pre-tonemap linear buffer if you want denoiser-paper-grade numbers.
- **PSNR / raw-MSE traps.** Perceptually weak, dominated by bright pixels, structure-blind; over-blur can *raise* PSNR while looking worse; a uniformly noisy frame and a single-artifact frame can score equally. Coarse sanity / convergence-curve only — never the verdict.
- **No-reference domain mismatch.** NIQE/BRISQUE/MUSIQ are trained on natural photos and **systematically misjudge synthetic + tonemapped + HDRI renders** in absolute terms. `estimate_sigma`/Immerkær assume additive Gaussian noise, but MC noise is signal-dependent, heavy-tailed (fireflies). All NR metrics are **relative same-scene A/B deltas**, never absolute pass/fail. And the **noise↔sharpness coupling**: a noisy frame reads "sharp," detail reads as "noise" — always read flat-region noise and edge sharpness as a pair.
- **Equal-time fairness.** For a real-time engine the meaningful A/B is **equal wall-clock**, not equal spp. Reserve equal-sample for understanding algorithmic quality-per-sample. Lock fps/ms before comparing.
- **Motion confound.** Raw temporal-std counts legitimate motion as error and requires bit-identical paths; **motion-compensate** (engine MVs preferred). But optical flow has its own errors — disocclusions, specular/refractive motion, DLSS ghosting break the flow assumption, so warped residual over/under-reports at boundaries and on mirrors. Use the fwd/bwd occlusion mask, treat reflective tiles with suspicion, prefer the engine's MV G-buffer or RAFT. DLSS-RR is itself a temporal neural denoiser — temporal-std under motion conflates DLSS reprojection with true MC noise, so grade a DLSS frame against a **DLSS-settled** reference unless you specifically want to measure DLSS *bias* vs pure path tracing.
- **Reference validity.** A self-reference is only "practically converged," not exact — it still carries firefly/clamping bias (and denoiser bias if DLSS-on). Converge far enough (verify the plateau), clamp fireflies, match resolution/tonemap/exposure/jitter exactly. Under motion you cannot accumulate — stop at keyframes for per-pose references, or fall back to NR + temporal.
- **TD plumbing.** `numpyArray` is **bottom-up** and the C++ CUDA TOP returns garbage if read directly → read a downstream regular TOP and `np.flipud` first, or every metric silently grades a flipped image. A hanging `numpyArray`/MCP call means **TD froze**. Drive accumulation by stepping `op('/').time.frame` in-script (a sleep loop blocks the main thread and freezes the timeline).

---

## Sources
- NVIDIA FLIP — pip `flip-evaluator`, NVlabs/flip, python README, "FLIP: A Difference Evaluator for Alternating Images" (dev blog + ACM 10.1145/3406183), *Ray Tracing Gems II*; HDR-FLIP arXiv 2210.05553.
- LPIPS — pip `lpips`, richzhang/PerceptualSimilarity.
- scikit-image metrics + `restoration.estimate_sigma`; pytorch-msssim (VainF); piq (photosynthesis-team) docs; torchmetrics MS-SSIM.
- relMSE/SMAPE conventions — KPCN/ANF/Disney/NVIDIA MC-denoiser literature.
- pyiqa / IQA-PyTorch (chaofengc), DeepWiki metrics list; BRISQUE (LearnOpenCV, LIVE NR-QA); NR-on-synthetic caveats (dataworlds, eu-opensci, arXiv 2510.13349 / 2602.08642).
- Immerkær fast noise (goldsequence note), medpy `filter.noise.immerkaer`; focus measures (OpenCV blog, pyimagesearch blur detection).
- Temporal — TecoGAN tOF/tLP (ge.in.tum.de, ToG 2020 / arXiv 1811.09393); ColorVideoVDP (gfxdisp/ColorVideoVDP, pip `cvvdp`, Mantiuk 2024); FovVideoVDP (gfxdisp).
- Production — OpenImageIO idiff docs + oiio idiff.rst; Arnold adaptive sampling (Autodesk), RenderMan sampling modes / variance metric (Pixar); Netflix VMAF (techblog, streaminglearningcenter best practices).
- ITU-R BT.500 / ITU-T P.910 (subjective-study methodology, only if human studies).