"""Parameter-validation metrics for the TD path tracer (system-python: numpy+cv2+skimage).
Consumes float32 .npy frames dumped from TD (already np.flipud'd, RGB 0..1).
Differential principle: a param works iff changing it moves the right metric the right way."""
import numpy as np, cv2

def luma(x): return (0.299*x[...,0]+0.587*x[...,1]+0.114*x[...,2]).astype(np.float64)

def _flat_mask(Y, pct=45):
    g=np.abs(cv2.Sobel(Y,cv2.CV_64F,1,0))+np.abs(cv2.Sobel(Y,cv2.CV_64F,0,1))
    m=g < np.percentile(g, pct)
    return m if m.sum()>500 else np.ones_like(m)  # guard: never empty (memory TODO)

def flat_noise(img):
    """std of high-pass residual in flat regions = honest spatial grain."""
    Y=luma(img); m=_flat_mask(Y)
    hp=Y-cv2.GaussianBlur(Y,(0,0),1.0)
    return float(np.std(hp[m]))

def sharpness_hf(img):
    """variance of Laplacian over the whole frame (HF energy; confounded by noise)."""
    return float(cv2.Laplacian(luma(img),cv2.CV_64F).var())

def edge_sharpness(img, region=None):
    """mean gradient magnitude at the strongest 2% of edges (AA / DOF / over-blur)."""
    Y=luma(img)
    if region: y0,y1,x0,x1=region; Y=Y[y0:y1,x0:x1]
    gx=cv2.Sobel(Y,cv2.CV_64F,1,0); gy=cv2.Sobel(Y,cv2.CV_64F,0,1)
    g=np.sqrt(gx*gx+gy*gy); t=np.percentile(g,98)
    return float(g[g>=t].mean())

def mean_lum(img, mask=None):
    Y=luma(img); return float(Y.mean() if mask is None else Y[mask].mean())

def firefly_count(img, k=4.0):
    Y=luma(img); thr=np.percentile(Y,99.0)*k
    return int((Y>thr).sum())

def image_delta(a,b):
    d=np.abs(a.astype(np.float64)-b.astype(np.float64))
    return {'rmse':round(float(np.sqrt(np.mean(d**2))),5),'max':round(float(d.max()),4),
            'changed':bool(np.mean(d)>2e-4)}

def relmse(test, ref, eps=1e-2):
    t,r=test.astype(np.float64),ref.astype(np.float64)
    return float(np.mean((t-r)**2/(r**2+eps)))

def _grid35(M):
    H,W=M.shape; gy,gx=3,5
    return [[round(float(M[H*r//gy:H*(r+1)//gy, W*c//gx:W*(c+1)//gx].mean()),4) for c in range(gx)] for r in range(gy)]

def temporal_boil(frames):
    """motion-compensated frame-to-frame residual in flat regions across a camera-motion sequence.
    Warps consecutive frames via Farneback flow so legitimate motion cancels; residual = boiling/noise."""
    Ys=[luma(f).astype(np.float32) for f in frames]
    m=_flat_mask(Ys[0],55)
    res=[]
    for i in range(1,len(Ys)):
        prev,cur=Ys[i-1],Ys[i]
        flow=cv2.calcOpticalFlowFarneback(prev,cur,None,0.5,3,25,3,5,1.2,0)
        h,w=cur.shape
        gx,gy=np.meshgrid(np.arange(w),np.arange(h))
        warped=cv2.remap(cur,(gx+flow[...,0]).astype(np.float32),(gy+flow[...,1]).astype(np.float32),cv2.INTER_LINEAR)
        res.append(np.abs(warped-prev))
    R=np.mean(res,axis=0)
    return {'boil_mean':round(float(R[m].mean()),5),'boil_p90':round(float(np.percentile(R[m],90)),5),'grid':_grid35(R)}

def load(path): return np.clip(np.load(path).astype(np.float32),0,1)
