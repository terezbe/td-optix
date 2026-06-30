"""
render_quality.py — objective render-quality harness for OptixDemoTOP (OptiX + DLSS-RR in TD)

WHY: temporal-std under motion is blind to static spatial noise, distance-from-converged,
and DLSS over-blur. This harness scores frames against a SELF-rendered converged reference
(full-reference) + a no-reference panel + motion-compensated temporal error, and returns
dicts of numbers WITH pass/fail by threshold. The decision is the numbers, not the eye.

pip install:
    flip-evaluator scikit-image piq opencv-python imageio numpy
optional (deep metrics): pyiqa lpips torch cvvdp ; (true EXR/CI gate) OpenImageIO

ALL images: float32, HxWx3, range 0..1, TOP-DOWN.
  TD numpyArray is BOTTOM-UP -> np.flipud() the downstream-TOP read BEFORE calling anything here,
  identically for test, reference, and every temporal frame. Read a downstream regular TOP
  (PT_Render/tonemap -> null2/level1), NOT the C++ CUDA TOP (it returns garbage).
"""

import numpy as np, cv2
import flip_evaluator as flip
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.restoration import estimate_sigma

# ----------------------------------------------------------------------------- thresholds
THRESH = dict(
    flip=0.08, relmse=5e-3, ssim=0.97, lpips=0.06,
    sharp_lo=0.85, sharp_hi=1.15, sharp_blur=0.75, sharp_grain=1.20,
    nr_sigma=0.02, temporal=0.01,
)
RELMSE_EPS = 1e-2  # Rousselle convention; ALWAYS report it, keep fixed across an A/B

# ----------------------------------------------------------------------------- helpers
def _luma(x):  return 0.299*x[...,0] + 0.587*x[...,1] + 0.114*x[...,2]
def _hf(x):    return float(cv2.Laplacian(_luma(x).astype(np.float64), cv2.CV_64F).var())
def _relmse(t, r, eps=RELMSE_EPS):
    return float(np.mean((t - r)**2 / (r**2 + eps)))
def _grid(m, gy=3, gx=5):
    h, w = m.shape[:2]
    return [[float(m[i*h//gy:(i+1)*h//gy, j*w//gx:(j+1)*w//gx].mean())
             for j in range(gx)] for i in range(gy)]

# ============================================================================= 0. REFERENCE
def generate_reference_notes():
    """
    How to bake the in-engine converged reference (do this in TD over MCP, not here):

      1. PT_Render Denoise page: denoiser = None   (pure MC; no DLSS bias).
         (Optionally ALSO bake ref_dlss_settled with DLSS on, to grade the denoised target.)
      2. Freeze camera: null5 static. Reset accumulation.
      3. Loop, advancing op('/').time.frame IN-SCRIPT each step so accumulation cooks
         (a sleep/while loop blocks TD's main thread and FREEZES the timeline).
         every K=128 frames grab buf_k from the downstream TOP (float32) and np.flipud it.
      4. Converged when relMSE(buf_{k-1}, buf_k) < 1e-4 (image stopped changing);
         hard ceiling ~4000-8000 accumulated spp.
      5. Render the reference at the DLSS OUTPUT resolution (pixel-aligned with the test)
         and identical tonemap/exposure. Save float32 .npy (avoids EXR/OIIO pain on Windows);
         key the file by a hash of the camera transform so it's reused per view.
      6. VERIFY the reference plateaued (plot PSNR/relMSE vs spp) before trusting any score.
    Returns: np.load('reference_<camhash>.npy')  -> float32 HxWx3 0..1, top-down.
    """
    raise NotImplementedError("Bake the reference in TD; see docstring. Then np.load() it.")

# ============================================================================= 1. STATIC (full-reference)
def measure_static(img, ref, want_lpips=False, save_flip_map=None):
    """Score a real-time/DLSS frame vs the converged reference. Returns numbers + pass/fail + verdict."""
    img = img.astype(np.float32); ref = ref.astype(np.float32)
    assert img.shape == ref.shape, f"resolution mismatch {img.shape} vs {ref.shape} = auto-fail"

    errMap, meanFLIP, _ = flip.evaluate(ref, img, "LDR")   # LDR: our read is tonemapped 0..1
    errMap = np.asarray(errMap)
    out = {
        "flip":        float(meanFLIP),
        "flip_grid":   _grid(errMap),
        "relmse":      _relmse(img, ref),
        "relmse_eps":  RELMSE_EPS,
        "ssim":        float(ssim(ref, img, channel_axis=-1, data_range=1.0)),
        "psnr":        float(psnr(ref, img, data_range=1.0)),   # sanity / convergence-curve only
        "sharp_ratio": _hf(img) / (_hf(ref) + 1e-12),
        "nr_sigma":    float(np.mean(estimate_sigma(img, channel_axis=-1, average_sigmas=True))),
    }
    if want_lpips:
        try:
            import torch, lpips
            net = lpips.LPIPS(net='alex')
            f = lambda x: torch.tensor(x).permute(2,0,1)[None].float()*2 - 1  # -> [-1,1] NCHW
            out["lpips"] = float(net(f(img), f(ref)).item())
        except Exception as e:
            out["lpips_error"] = str(e)
    if save_flip_map:
        cv2.imwrite(save_flip_map, cv2.applyColorMap((errMap*255).astype(np.uint8), cv2.COLORMAP_INFERNO))

    out["pass"] = dict(
        flip = out["flip"] <= THRESH["flip"],
        relmse = out["relmse"] <= THRESH["relmse"],
        ssim = out["ssim"] >= THRESH["ssim"],
        sharp = THRESH["sharp_lo"] <= out["sharp_ratio"] <= THRESH["sharp_hi"],
    )
    out["verdict"] = _classify_static(out)
    return out

def _classify_static(m):
    sr = m["sharp_ratio"]
    if sr < THRESH["sharp_blur"] and m["ssim"] < THRESH["ssim"]:
        return "OVER-BLURRED (denoiser ate detail)"
    if (m["relmse"] > THRESH["relmse"] or m["flip"] > THRESH["flip"]) and sr > THRESH["sharp_grain"]:
        return "NOISY (residual MC grain)"
    if (m["flip"] <= THRESH["flip"] and m["relmse"] <= THRESH["relmse"]
            and THRESH["sharp_lo"] <= sr <= THRESH["sharp_hi"]):
        return "CONVERGED & FAITHFUL"
    return "MARGINAL -> escalate to Gemini Vision"

# ============================================================================= 2. NO-REFERENCE (live)
def measure_noref(img, want_pyiqa=False):
    """No converged reference (live/motion). RELATIVE same-scene A/B signals ONLY, never absolute."""
    img = img.astype(np.float32)
    Y = _luma(img)
    out = {
        "nr_sigma":     float(np.mean(estimate_sigma(img, channel_axis=-1, average_sigmas=True))),
        "nr_sigma_grid":_grid(_flat_noise_map(Y)),     # de-confounded: noise in flat regions only
        "sharp_lapvar": _hf(img),
        "sharp_tenengrad": _tenengrad(Y),
        "hf_ratio":     _spectral_hf_ratio(Y),
    }
    if want_pyiqa:
        try:
            import torch, pyiqa
            t = torch.tensor(img).permute(2,0,1)[None].float()
            for name in ("niqe", "brisque", "musiq", "clipiqa"):
                out[f"nr_{name}"] = float(pyiqa.create_metric(name)(t).item())
        except Exception as e:
            out["pyiqa_error"] = str(e)
    out["verdict"] = ("NOISY (NR, relative)" if out["nr_sigma"] > THRESH["nr_sigma"]
                      else "LOOKS CLEAN (NR, low confidence)")
    return out

def _tenengrad(Y):
    gx = cv2.Sobel(Y.astype(np.float64), cv2.CV_64F, 1, 0)
    gy = cv2.Sobel(Y.astype(np.float64), cv2.CV_64F, 0, 1)
    return float(np.mean(gx*gx + gy*gy))

def _spectral_hf_ratio(Y, cutoff=0.25):
    F = np.fft.fftshift(np.abs(np.fft.fft2(Y)))**2
    h, w = Y.shape; cy, cx = h//2, w//2
    yy, xx = np.ogrid[:h, :w]
    r = np.sqrt(((yy-cy)/cy)**2 + ((xx-cx)/cx)**2)
    return float(F[r > cutoff].sum() / (F.sum() + 1e-12))

def _flat_noise_map(Y, grad_pct=40, blur=1.0):
    """Std of high-pass residual restricted to low-gradient (flat) pixels — honest static grain."""
    g = np.abs(cv2.Sobel(Y.astype(np.float64), cv2.CV_64F, 1, 0)) + \
        np.abs(cv2.Sobel(Y.astype(np.float64), cv2.CV_64F, 0, 1))
    flat = g < np.percentile(g, grad_pct)
    hp = Y - cv2.GaussianBlur(Y, (0, 0), blur)
    out = np.zeros_like(Y, dtype=np.float64)
    out[flat] = np.abs(hp[flat])
    return out

# ============================================================================= 3. TEMPORAL
def measure_temporal(frames, mvs=None):
    """
    frames: list of consecutive output frames (float32 HxWx3 0..1) on the IDENTICAL fixed motion
    path (step op('/').time.frame in-script so A and B share the path). mvs[t]: optional per-frame
    engine motion-vector field (HxWx2, pixel offsets prev->cur from DLSS-RR G-buffer) -> exact warp.
    Returns raw temporal-std (legacy boiling A/B) AND motion-compensated residual, each + 3x5 grid.
    """
    fr = [f.astype(np.float32) for f in frames]
    stack = np.stack(fr, 0)
    tstd = stack.std(0).mean(-1)                       # legacy per-pixel temporal-std

    residuals = []
    for t in range(1, len(fr)):
        flow = mvs[t] if mvs is not None else _farneback(fr[t-1], fr[t])
        h, w = fr[t].shape[:2]
        gx, gy = np.meshgrid(np.arange(w), np.arange(h))
        mapx = (gx + flow[...,0]).astype(np.float32)
        mapy = (gy + flow[...,1]).astype(np.float32)
        warp = cv2.remap(fr[t-1], mapx, mapy, cv2.INTER_LINEAR)
        occ = (mapx >= 0) & (mapx < w) & (mapy >= 0) & (mapy < h)   # mask disocclusions
        resid = np.abs(fr[t] - warp).mean(-1)
        resid[~occ] = 0.0
        residuals.append(resid)
    mc = np.mean(residuals, 0) if residuals else np.zeros_like(tstd)

    out = {
        "temporal_std":       float(tstd.mean()),
        "temporal_std_grid":  _grid(tstd),
        "motion_comp_resid":  float(mc.mean()),
        "motion_comp_grid":   _grid(mc),
    }
    out["pass"] = dict(
        temporal_std = out["temporal_std"] <= THRESH["temporal"],
        motion_comp  = out["motion_comp_resid"] <= THRESH["temporal"],
    )
    out["verdict"] = ("STABLE" if out["motion_comp_resid"] <= THRESH["temporal"]
                      else "SOME BOILING" if out["motion_comp_resid"] <= 0.03 else "BOILING")
    return out

def _farneback(a, b):
    g0 = cv2.cvtColor((a*255).astype('uint8'), cv2.COLOR_RGB2GRAY)
    g1 = cv2.cvtColor((b*255).astype('uint8'), cv2.COLOR_RGB2GRAY)
    return cv2.calcOpticalFlowFarneback(g0, g1, None, 0.5, 3, 15, 3, 5, 1.2, 0)

# Optional headline temporal perceptual score over a clip vs a high-spp reference path:
#   import pycvvdp
#   JOD, _ = pycvvdp.cvvdp(display_name='standard_4k').predict(
#       np.stack(test), np.stack(ref), dim_order="FHWC", frames_per_second=50)  # JOD 10=identical

# ============================================================================= demo
if __name__ == "__main__":
    ref  = np.load("reference_<camhash>.npy")          # see generate_reference_notes()
    test = np.flipud(np.load("test_dlss.npy"))         # remember: TD is bottom-up
    print("STATIC  :", measure_static(test, ref, save_flip_map="flip_map.png"))
    print("NO-REF  :", measure_noref(test))
    # frames = [np.flipud(f) for f in path_capture]    # identical motion path, A vs B
    # print("TEMPORAL:", measure_temporal(frames, mvs=engine_mvs_or_None))