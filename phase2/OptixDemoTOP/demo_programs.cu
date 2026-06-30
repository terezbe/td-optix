// Standalone OptiX path tracer — procedural sphere scene (no external geometry).
// Built to PTX by nvcc; loaded by OptixDemoTOP.cpp. Hardware sphere primitives.
#include <optix.h>
#include <cuda_runtime.h>
#include "LaunchParamsDemo.h"

extern "C" { __constant__ LaunchParamsDemo params; }

// ---------------- small vector helpers ----------------
static __forceinline__ __device__ float3 V(float x,float y,float z){ return make_float3(x,y,z); }
static __forceinline__ __device__ float3 add(float3 a,float3 b){ return make_float3(a.x+b.x,a.y+b.y,a.z+b.z); }
static __forceinline__ __device__ float3 sub(float3 a,float3 b){ return make_float3(a.x-b.x,a.y-b.y,a.z-b.z); }
static __forceinline__ __device__ float3 mul(float3 a,float3 b){ return make_float3(a.x*b.x,a.y*b.y,a.z*b.z); }
static __forceinline__ __device__ float3 scl(float3 a,float s){ return make_float3(a.x*s,a.y*s,a.z*s); }
static __forceinline__ __device__ float  dot3(float3 a,float3 b){ return a.x*b.x+a.y*b.y+a.z*b.z; }
static __forceinline__ __device__ float3 cross3(float3 a,float3 b){ return make_float3(a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x); }
static __forceinline__ __device__ float3 norm(float3 a){ float l=rsqrtf(fmaxf(dot3(a,a),1e-20f)); return scl(a,l); }
static __forceinline__ __device__ float  maxc(float3 a){ return fmaxf(a.x,fmaxf(a.y,a.z)); }

// ---------------- rng (pcg) ----------------
static __forceinline__ __device__ unsigned int pcg(unsigned int& s){ s=s*747796405u+2891336453u; unsigned int w=((s>>((s>>28u)+4u))^s)*277803737u; return (w>>22u)^w; }
static __forceinline__ __device__ float rnd(unsigned int& s){ return (pcg(s)&0x00FFFFFFu)/16777216.0f; }

// ---------------- per-ray data + payload pointer packing ----------------
struct PRD
{
    float3       attenuation;   // surface throughput multiplier for the next bounce
    float3       emitted;       // emitted radiance at this hit
    float3       skyColor;      // sky radiance if this ray missed
    float3       hitAlbedo;     // surface albedo at the hit (denoiser guide AOV)
    float3       hitNormal;     // world normal at the hit (denoiser guide AOV)
    float3       hitPos;        // world position at the hit (temporal motion vectors)
    float3       hitSpecAlb;    // specular albedo / F0 (RR guide)
    float        hitRough;      // linear roughness (RR guide)
    float3       nextOrigin;    // scattered ray origin
    float3       nextDir;       // scattered ray direction
    unsigned int seed;
    int          miss;          // 1 = ray escaped to sky
    int          absorbed;      // 1 = terminal (emissive)
    int          isDiffuse;     // 1 = lambert hit (NEE applies)
    int          isShadow;      // 1 = this is an occlusion query
    int          occluded;      // shadow-ray result
};
static __forceinline__ __device__ void* unpackPtr(unsigned int i0,unsigned int i1){ return reinterpret_cast<void*>(((unsigned long long)i0<<32)|i1); }
static __forceinline__ __device__ void  packPtr(void* p,unsigned int& i0,unsigned int& i1){ unsigned long long u=(unsigned long long)p; i0=(unsigned int)(u>>32); i1=(unsigned int)(u&0xffffffffu); }
static __forceinline__ __device__ PRD*  getPRD(){ return reinterpret_cast<PRD*>(unpackPtr(optixGetPayload_0(),optixGetPayload_1())); }

// ---------------- sampling ----------------
static __forceinline__ __device__ float3 cosineHemisphere(float3 n,float u1,float u2)
{
    float r=sqrtf(u1), phi=6.28318530718f*u2;
    float3 t = fabsf(n.x)>0.1f ? V(0,1,0) : V(1,0,0);
    float3 b1=norm(cross3(t,n)); float3 b2=cross3(n,b1);
    float x=r*cosf(phi), y=r*sinf(phi), z=sqrtf(fmaxf(0.0f,1.0f-u1));
    return norm(add(add(scl(b1,x),scl(b2,y)),scl(n,z)));
}
static __forceinline__ __device__ float3 reflect3(float3 v,float3 n){ return sub(v,scl(n,2.0f*dot3(v,n))); }
static __forceinline__ __device__ bool refract3(float3 v,float3 n,float eta,float3& out)
{
    float ci=-dot3(v,n); float k=1.0f-eta*eta*(1.0f-ci*ci);
    if(k<0.0f) return false;
    out=add(scl(v,eta),scl(n,eta*ci-sqrtf(k))); return true;
}
static __forceinline__ __device__ float schlick(float ci,float ior){ float r0=(1.0f-ior)/(1.0f+ior); r0*=r0; return r0+(1.0f-r0)*powf(fmaxf(0.0f,1.0f-ci),5.0f); }
static __forceinline__ __device__ float2 diskRand(unsigned int& s){ float r=sqrtf(rnd(s)), a=6.28318530718f*rnd(s); return make_float2(r*cosf(a),r*sinf(a)); }

// ---------------- sky (gradient dome + sharp sun disk, kept separable) ----------------
// ---- xyY -> linear sRGB (Preetham physical sky) ----
static __forceinline__ __device__ float3 xyY_to_rgb(float x,float y,float Y)
{
    if(y<1e-5f) return V(0.0f,0.0f,0.0f);
    float X=(x/y)*Y, Z=((1.0f-x-y)/y)*Y;
    float r= 3.2406f*X -1.5372f*Y -0.4986f*Z;   // CIE XYZ (D65) -> linear sRGB
    float g=-0.9689f*X +1.8758f*Y +0.0415f*Z;
    float b= 0.0557f*X -0.2040f*Y +1.0570f*Z;
    return V(fmaxf(r,0.0f),fmaxf(g,0.0f),fmaxf(b,0.0f));
}
static __forceinline__ __device__ float3 preethamSky(float3 d)   // analytic daylight sky (no solar disk)
{
    float cosT=fmaxf(d.y,0.0015f);                              // view zenith cos (clamp at horizon)
    float cosG=fminf(fmaxf(dot3(d,params.sunDir),-1.0f),1.0f);
    float g=acosf(cosG);
    float3 A=params.perezA,B=params.perezB,C=params.perezC,D=params.perezD,E=params.perezE;
    float Fx=(1.0f+A.x*__expf(B.x/cosT))*(1.0f+C.x*__expf(D.x*g)+E.x*cosG*cosG);   // Perez F per channel
    float Fy=(1.0f+A.y*__expf(B.y/cosT))*(1.0f+C.y*__expf(D.y*g)+E.y*cosG*cosG);
    float FY=(1.0f+A.z*__expf(B.z/cosT))*(1.0f+C.z*__expf(D.z*g)+E.z*cosG*cosG);
    float x=params.perezZenith.x*Fx/params.perezNorm.x;
    float y=params.perezZenith.y*Fy/params.perezNorm.y;
    float Y=params.perezZenith.z*FY/params.perezNorm.z;
    return scl(xyY_to_rgb(x,y,Y), 0.05f*params.skyStrength);    // 0.05 = exposure normalization (kcd/m^2 -> ~1)
}
static __forceinline__ __device__ float3 skyDome(float3 d)   // image-based light: gradient / HDRI / Preetham
{
    if(params.skyMode==1 && params.hdriTex){                  // equirectangular HDRI lookup
        float u=atan2f(d.z,d.x)*0.15915494f + 0.5f + params.hdriRot;   // 1/(2pi); U wraps in the sampler
        float v=acosf(fminf(fmaxf(d.y,-1.0f),1.0f))*0.31830989f;       // 1/pi ; v=0 at zenith (top row)
        float4 c=tex2D<float4>(params.hdriTex,u,v);
        return scl(V(c.x,c.y,c.z), params.skyStrength);
    }
    if(params.skyMode==2) return preethamSky(d);             // Preetham physical sky
    float t=0.5f*(d.y+1.0f);
    float3 sky=add(scl(params.skyHorizon,1.0f-t),scl(params.skyZenith,t));
    return scl(sky,params.skyStrength);
}
static __forceinline__ __device__ float3 sunDisk(float3 d)   // sharp analytic sun disk (gradient + physical modes)
{
    if(params.skyMode==1) return V(0.0f,0.0f,0.0f);           // HDRI supplies its own sun visual -> no double disk
    float s=fmaxf(0.0f,dot3(d,params.sunDir));
    return scl(params.sunColor, powf(s,1500.0f)*40.0f);       // tight, bright sun disk
}
static __forceinline__ __device__ float3 skyColor(float3 d){ return add(skyDome(d), sunDisk(d)); }

// ---- next-event estimation: direct lighting from one power-sampled emissive sphere ----
static __forceinline__ __device__ float3 sampleDirect(float3 P, float3 N, float3 albedo, unsigned int& seed)
{
    if(!params.neeEnable || params.numLights<=0 || params.totalPower<=0.0f) return V(0,0,0);
    float u = rnd(seed) * params.totalPower;                    // pick a light proportional to power
    int lo=0, hi=params.numLights-1;
    while(lo<hi){ int mid=(lo+hi)>>1; if(params.lightCDF[mid] < u) lo=mid+1; else hi=mid; }
    int li=lo;
    float prevC = li>0 ? params.lightCDF[li-1] : 0.0f;
    float power = params.lightCDF[li] - prevC;
    if(power<=0.0f) return V(0,0,0);
    float pdfLight = power / params.totalPower;
    float4 lp=params.lightPos[li]; float3 C=V(lp.x,lp.y,lp.z); float R=lp.w;
    float4 le=params.lightEmit[li]; float3 E=V(le.x,le.y,le.z);
    float z=1.0f-2.0f*rnd(seed); float rr=sqrtf(fmaxf(0.0f,1.0f-z*z)); float phi=6.28318530718f*rnd(seed);
    float3 nq=V(rr*cosf(phi), rr*sinf(phi), z);                 // uniform point on the light sphere
    float3 Q=add(C, scl(nq,R));
    float3 dvec=sub(Q,P); float dist2=dot3(dvec,dvec); float dist=sqrtf(dist2);
    if(dist<1e-4f) return V(0,0,0);
    float3 wi=scl(dvec, 1.0f/dist);
    float cosS=dot3(N,wi); float cosL=-dot3(nq,wi);
    if(cosS<=0.0f || cosL<=0.0f) return V(0,0,0);
    PRD sh; sh.isShadow=1; sh.occluded=0; sh.miss=0;            // occlusion test (stop before the light)
    unsigned int u0,u1; packPtr(&sh,u0,u1);
    optixTrace(params.handle, add(P,scl(N,1e-3f)), wi, 1e-3f, dist-1e-3f, 0.0f,
               OptixVisibilityMask(255), OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT, 0,1,0, u0,u1);
    if(sh.occluded) return V(0,0,0);
    float area=4.0f*3.14159265f*R*R;
    float est = 0.31830989f * (cosS*cosL/dist2) * (area/pdfLight);   // (1/pi)*G*Area/pdfLight
    return scl(mul(albedo,E), est);
}

// ---- next-event estimation: the sun as an analytic directional light (delta light, pdf=1) ----
// crisp shadows + no fireflies (one direction, no 1/dist^2). Soft penumbra from a small angular cone.
static __forceinline__ __device__ float3 sampleSun(float3 P, float3 N, float3 albedo, unsigned int& seed)
{
    float3 E=params.sunColor;                                  // irradiance (Suncolor * Sunstrength)
    if(E.x+E.y+E.z <= 0.0f) return V(0,0,0);                   // sun off
    float3 wi=params.sunDir;
    if(params.sunAngle > 0.0f){                                // jitter inside a cone -> soft shadows
        float ca=cosf(params.sunAngle);
        float cosT=1.0f-rnd(seed)*(1.0f-ca);
        float sinT=sqrtf(fmaxf(0.0f,1.0f-cosT*cosT));
        float phi=6.28318530718f*rnd(seed);
        float3 t = fabsf(wi.x)>0.1f ? V(0,1,0) : V(1,0,0);
        float3 b1=norm(cross3(t,wi)); float3 b2=cross3(wi,b1);
        wi=norm(add(add(scl(b1,sinT*cosf(phi)), scl(b2,sinT*sinf(phi))), scl(wi,cosT)));
    }
    float ndl=dot3(N,wi);
    if(ndl<=0.0f) return V(0,0,0);                             // sun below the surface horizon
    PRD sh; sh.isShadow=1; sh.occluded=0; sh.miss=0;           // occlusion ray toward the sun
    unsigned int u0,u1; packPtr(&sh,u0,u1);
    optixTrace(params.handle, add(P,scl(N,1e-3f)), wi, 1e-3f, 1e16f, 0.0f,
               OptixVisibilityMask(255), OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT, params.useInput?1u:0u,1,0, u0,u1);
    if(sh.occluded) return V(0,0,0);
    return scl(mul(albedo,E), 0.31830989f*ndl);               // lambert f=albedo/pi : f * E * cos
}

// ---- next-event estimation: importance-sample the HDRI via its 2D CDF (kills bright-spot fireflies) ----
static __forceinline__ __device__ float3 sampleEnv(float3 P, float3 N, float3 albedo, unsigned int& seed)
{
    int W=params.envW, H=params.envH;
    if(W<=0 || H<=0 || params.envFuncInt<=0.0f) return V(0,0,0);
    float u1=rnd(seed);                                        // pick a row j via the marginal CDF
    int lo=0,hi=H-1; while(lo<hi){ int m=(lo+hi)>>1; if(params.envMargCdf[m]<u1) lo=m+1; else hi=m; } int j=lo;
    float u2=rnd(seed);                                        // pick a column i via row j's conditional CDF
    const float* row=params.envCondCdf + (unsigned)j*W;
    lo=0; hi=W-1; while(lo<hi){ int m=(lo+hi)>>1; if(row[m]<u2) lo=m+1; else hi=m; } int i=lo;
    float uu=((float)i+rnd(seed))/(float)W;                   // jittered position inside the texel
    float vv=((float)j+rnd(seed))/(float)H;
    float theta=vv*3.14159265f, sinT=sinf(theta);
    if(sinT<1e-4f) return V(0,0,0);
    float phi=(uu-0.5f-params.hdriRot)*6.28318530718f;        // inverse of skyDome's equirect mapping
    float3 wi=V(sinT*cosf(phi), cosf(theta), sinT*sinf(phi));
    float ndl=dot3(N,wi); if(ndl<=0.0f) return V(0,0,0);
    float func=params.envFunc[(unsigned)j*W+i];               // pdf(solid angle) = (func*W*H/funcInt)/(2pi^2 sinT)
    float pdf=(func*(float)W*(float)H/params.envFuncInt)/(6.28318530718f*3.14159265f*sinT);
    if(pdf<=1e-8f) return V(0,0,0);
    float4 c=tex2D<float4>(params.hdriTex,uu,vv);             // radiance from the full-res HDRI
    float3 L=scl(V(c.x,c.y,c.z), params.skyStrength);
    PRD sh; sh.isShadow=1; sh.occluded=0; sh.miss=0;          // occlusion ray toward the chosen env direction
    unsigned int o0,o1; packPtr(&sh,o0,o1);
    optixTrace(params.handle, add(P,scl(N,1e-3f)), wi, 1e-3f, 1e16f, 0.0f,
               OptixVisibilityMask(255), OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT, params.useInput?1u:0u,1,0, o0,o1);
    if(sh.occluded) return V(0,0,0);
    return scl(mul(albedo,L), 0.31830989f*ndl/pdf);          // lambert f=albedo/pi : f * L * cos / pdf
}

static __forceinline__ __device__ float smooth01(float a,float b,float x){ float t=fminf(fmaxf((x-a)/(b-a+1e-8f),0.0f),1.0f); return t*t*(3.0f-2.0f*t); }

// ---- next-event estimation over the analytic Light COMPs (point/cone/distant delta lights, pdf=1) ----
// Light data is a small device buffer parsed host-side from the render's "Lightdata" string param.
static __forceinline__ __device__ float3 sampleLights(float3 P, float3 Nf, float3 albedo, unsigned int& seed)
{
    float3 sum=V(0,0,0);
    float master=params.lightMaster;
    for(int i=0;i<params.numPtLights;i++){
        PTLight L=params.ptLights[i];
        if(L.radiance.x+L.radiance.y+L.radiance.z<=0.0f) continue;       // off / zero light
        float3 wi; float tmax; float atten;
        if(L.type==2){                                       // distant (sun) — no 1/d^2
            wi=scl(L.dir,-1.0f);                             // toward the light = opposite its pointing dir
            if(L.radius>0.0f){                               // angular soft shadow
                float ca=cosf(L.radius), cz=1.0f-rnd(seed)*(1.0f-ca), sz=sqrtf(fmaxf(0.0f,1.0f-cz*cz)), ph=6.28318530718f*rnd(seed);
                float3 t=fabsf(wi.x)>0.1f?V(0,1,0):V(1,0,0); float3 b1=norm(cross3(t,wi)); float3 b2=cross3(wi,b1);
                wi=norm(add(add(scl(b1,sz*cosf(ph)),scl(b2,sz*sinf(ph))),scl(wi,cz)));
            }
            tmax=1e16f; atten=master;
        } else {                                             // point / cone — 1/d^2, balanced by 50 so TD dimmer~1 reads
            float3 Q=L.pos;
            if(L.radius>0.0f){                               // sample the light sphere (penumbra)
                float z=1.0f-2.0f*rnd(seed), rr=sqrtf(fmaxf(0.0f,1.0f-z*z)), ph=6.28318530718f*rnd(seed);
                Q=add(L.pos, scl(V(rr*cosf(ph),rr*sinf(ph),z), L.radius));
            }
            float3 toL=sub(Q,P); float d2=dot3(toL,toL); float d=sqrtf(fmaxf(d2,1e-12f));
            wi=scl(toL,1.0f/d);
            atten=master*50.0f/(d2 + L.radius*L.radius);     // 50 = built-in per-type balance (point/cone vs distant)
            tmax=d-1e-3f;
            if(L.type==1){                                   // cone (spotlight) angular falloff
                float cd=dot3(scl(wi,-1.0f), L.dir);
                float cone=smooth01(L.cosOuter, L.cosInner, cd);
                if(cone<=0.0f) continue;
                atten*=cone;
            }
        }
        float ndl=dot3(Nf,wi); if(ndl<=0.0f) continue;
        PRD sh; sh.isShadow=1; sh.occluded=0; sh.miss=0;
        unsigned int o0,o1; packPtr(&sh,o0,o1);
        optixTrace(params.handle, add(P,scl(Nf,1e-3f)), wi, 1e-3f, tmax, 0.0f,
                   OptixVisibilityMask(255), OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT, params.useInput?1u:0u,1,0, o0,o1);
        if(sh.occluded) continue;
        sum=add(sum, scl(mul(albedo,L.radiance), 0.31830989f*ndl*atten));   // lambert f=albedo/pi
    }
    return sum;
}

// ---- Henyey-Greenstein phase function (mu = cos angle to the forward/propagation dir; g>0 = forward) ----
static __forceinline__ __device__ float phaseHG(float mu, float g)
{
    float d = 1.0f + g*g - 2.0f*g*mu; d = fmaxf(d, 1e-6f);
    return 0.0795774715f * (1.0f-g*g) / (d*sqrtf(d));    // 1/(4pi) * (1-g^2)/(1+g^2-2g*mu)^1.5
}
// sample a scattered direction from HG around the forward direction (matched pair with phaseHG)
static __forceinline__ __device__ float3 sampleHGdir(float3 fwd, float g, float u0, float u1)
{
    float mu;
    if(fabsf(g)<1e-3f) mu = 1.0f-2.0f*u0;                // isotropic
    else { float s=(1.0f-g*g)/(1.0f-g+2.0f*g*u0); mu=(1.0f+g*g-s*s)/(2.0f*g); }
    mu=fminf(fmaxf(mu,-1.0f),1.0f);
    float st=sqrtf(fmaxf(0.0f,1.0f-mu*mu)), ph=6.28318530718f*u1;
    float3 t=fabsf(fwd.x)>0.1f?V(0,1,0):V(1,0,0); float3 b1=norm(cross3(t,fwd)); float3 b2=cross3(fwd,b1);
    return norm(add(add(scl(b1,st*cosf(ph)),scl(b2,st*sinf(ph))),scl(fwd,mu)));
}
// ---- medium in-scatter NEE toward the HDRI environment (phase-weighted; mirrors sampleEnv's 2D-CDF pick) ----
// Same importance sample as the surface sampleEnv, but with the HG phase function instead of the cosine/BRDF
// and no surface normal. Returns phaseHG(wi.rd,g) * L_env / pdf for one sample.
static __forceinline__ __device__ float3 mediumEnvNEE(float3 P, float3 rd, float g, unsigned int& seed)
{
    int W=params.envW, H=params.envH;
    if(W<=0 || H<=0 || params.envFuncInt<=0.0f) return V(0,0,0);
    float u1=rnd(seed);                                       // pick a row j via the marginal CDF
    int lo=0,hi=H-1; while(lo<hi){ int m=(lo+hi)>>1; if(params.envMargCdf[m]<u1) lo=m+1; else hi=m; } int j=lo;
    float u2=rnd(seed);                                       // pick a column i via row j's conditional CDF
    const float* row=params.envCondCdf + (unsigned)j*W;
    lo=0; hi=W-1; while(lo<hi){ int m=(lo+hi)>>1; if(row[m]<u2) lo=m+1; else hi=m; } int i=lo;
    float uu=((float)i+rnd(seed))/(float)W, vv=((float)j+rnd(seed))/(float)H;
    float theta=vv*3.14159265f, sinT=sinf(theta);
    if(sinT<1e-4f) return V(0,0,0);
    float phi=(uu-0.5f-params.hdriRot)*6.28318530718f;       // inverse of skyDome's equirect mapping
    float3 wi=V(sinT*cosf(phi), cosf(theta), sinT*sinf(phi));
    float func=params.envFunc[(unsigned)j*W+i];
    float pdf=(func*(float)W*(float)H/params.envFuncInt)/(6.28318530718f*3.14159265f*sinT);
    if(pdf<=1e-8f) return V(0,0,0);
    float4 c=tex2D<float4>(params.hdriTex,uu,vv);
    float3 L=scl(V(c.x,c.y,c.z), params.skyStrength);
    PRD sh; sh.isShadow=1; sh.occluded=0; sh.miss=0; unsigned int o0,o1; packPtr(&sh,o0,o1);
    optixTrace(params.handle, add(P,scl(wi,1e-3f)), wi, 1e-3f, 1e16f, 0.0f,
               OptixVisibilityMask(255), OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT, params.useInput?1u:0u,1,0, o0,o1);
    if(sh.occluded) return V(0,0,0);
    return scl(L, phaseHG(dot3(wi,rd), g) / pdf);            // phase * L_env / pdf (no cosine, no albedo)
}

// ---- medium in-scatter NEE toward EMISSIVE TRIANGLE GEOMETRY (emitter glow) ----
// Picks one emissive triangle UNIFORMLY, samples a point Q uniformly by area, traces an occlusion ray
// P->Q, and returns the phase-weighted area-light in-scatter (un-albedo'd; the caller multiplies by fogColor,
// matching the sun/PTLight path). NO surface cosine/BRDF: phaseHG replaces f*cos. Single-sample NEE (no MIS).
// Uniform pick: pdf_area(Q) = (1/n)*(1/area) -> estimator = phase * Le * (cosL/dist^2) * area * n.
static __forceinline__ __device__ float3 mediumEmitNEE(float3 P, float3 rd, float g, unsigned int& seed)
{
    int n = params.numEmitTri;
    if(!params.fogEmitNEE || n <= 0 || !params.emitTriIdx) return V(0,0,0);
    int k = (int)(rnd(seed)*(float)n); if(k>=n) k=n-1;            // uniform triangle pick
    unsigned int i0 = 3u*(unsigned)params.emitTriIdx[k];
    float4 a4=params.triVerts[i0+0u], b4=params.triVerts[i0+1u], c4=params.triVerts[i0+2u];
    float3 v0=V(a4.x,a4.y,a4.z), v1=V(b4.x,b4.y,b4.z), v2=V(c4.x,c4.y,c4.z);
    float3 cd  = params.triCd ? V(params.triCd[i0].x,params.triCd[i0].y,params.triCd[i0].z) : V(0.82f,0.80f,0.78f);
    float  estr= (params.triMat && params.triMat[i0].w>0.0f) ? params.triMat[i0].w : 1.0f;   // matches __closesthit__tri
    float3 Le  = scl(cd, estr);

    float r1=rnd(seed), r2=rnd(seed), sq=sqrtf(r1);              // uniform-area barycentrics
    float3 Q = add(add(scl(v0,1.0f-sq), scl(v1,sq*(1.0f-r2))), scl(v2,sq*r2));
    float3 crs = cross3(sub(v1,v0), sub(v2,v0));
    float area = 0.5f*sqrtf(fmaxf(dot3(crs,crs),0.0f));
    if(area <= 1e-12f) return V(0,0,0);                          // degenerate triangle
    float3 nL = scl(crs, 1.0f/fmaxf(2.0f*area,1e-12f));          // unit geometric normal

    float3 dvec=sub(Q,P); float dist2=dot3(dvec,dvec); float dist=sqrtf(dist2);
    if(dist < 2.1e-3f) return V(0,0,0);                          // avoid inverted shadow-ray interval (tmin>tmax)
    float3 wi = scl(dvec, 1.0f/dist);
    float cosL = fabsf(dot3(nL, wi));                            // double-sided emitter (closest-hit emits both faces)
    if(cosL <= 1e-6f) return V(0,0,0);

    PRD sh; sh.isShadow=1; sh.occluded=0; sh.miss=0; unsigned int o0,o1; packPtr(&sh,o0,o1);
    optixTrace(params.handle, add(P,scl(wi,1e-3f)), wi, 1e-3f, dist-1e-3f, 0.0f,
               OptixVisibilityMask(255), OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT, params.useInput?1u:0u,1,0, o0,o1);
    if(sh.occluded) return V(0,0,0);

    float phase = phaseHG(dot3(wi, rd), g);                      // medium phase replaces f*cos (incl. 1/4pi)
    float est   = phase * (cosL / dist2) * area * (float)n;      // uniform pick: 1/pdf_area = area * n
    return scl(Le, est);                                         // un-albedo'd; caller mul's fogColor
}

// ---- in-scatter at a medium point: phase-weighted direct lighting (sun + analytic lights). No surface/BRDF. ----
static __forceinline__ __device__ float3 inscatterFog(float3 P, float3 rd, unsigned int& seed)
{
    float3 sum=V(0,0,0); float g=params.fogG; float gShaft=params.fogShaftG;
    // sun (distant directional) — full strength (treated as outside the fog; the occlusion ray makes the shafts)
    float3 E=params.sunColor;
    if(E.x+E.y+E.z>0.0f){
        float3 wi=params.sunDir;
        if(params.sunAngle>0.0f){
            float ca=cosf(params.sunAngle), cz=1.0f-rnd(seed)*(1.0f-ca), sz=sqrtf(fmaxf(0.0f,1.0f-cz*cz)), ph=6.28318530718f*rnd(seed);
            float3 t=fabsf(wi.x)>0.1f?V(0,1,0):V(1,0,0); float3 b1=norm(cross3(t,wi)); float3 b2=cross3(wi,b1);
            wi=norm(add(add(scl(b1,sz*cosf(ph)),scl(b2,sz*sinf(ph))),scl(wi,cz)));
        }
        PRD sh; sh.isShadow=1; sh.occluded=0; sh.miss=0; unsigned int o0,o1; packPtr(&sh,o0,o1);
        optixTrace(params.handle, add(P,scl(wi,1e-3f)), wi, 1e-3f, 1e16f, 0.0f,
                   OptixVisibilityMask(255), OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT, params.useInput?1u:0u,1,0, o0,o1);
        if(!sh.occluded) sum=add(sum, scl(E, phaseHG(dot3(wi,rd),gShaft)));   // SHAFT g (decoupled from bulk haze)
    }
    // analytic Light COMPs (point/cone/distant); finite lights attenuate through the fog over their distance
    float master=params.lightMaster;
    for(int i=0;i<params.numPtLights;i++){
        PTLight Lt=params.ptLights[i];
        if(Lt.radiance.x+Lt.radiance.y+Lt.radiance.z<=0.0f) continue;
        float3 wi; float tmax, atten, segTr=1.0f;
        if(Lt.type==2){ wi=scl(Lt.dir,-1.0f); tmax=1e16f; atten=master; }
        else {
            float3 Q=Lt.pos;
            if(Lt.radius>0.0f){ float z=1.0f-2.0f*rnd(seed), rr=sqrtf(fmaxf(0.0f,1.0f-z*z)), ph=6.28318530718f*rnd(seed); Q=add(Lt.pos,scl(V(rr*cosf(ph),rr*sinf(ph),z),Lt.radius)); }
            float3 toL=sub(Q,P); float d2=dot3(toL,toL), d=sqrtf(fmaxf(d2,1e-12f)); wi=scl(toL,1.0f/d);
            atten=master*50.0f/(d2+Lt.radius*Lt.radius); tmax=d-1e-3f; segTr=__expf(-params.fogDensity*d);
            if(Lt.type==1){ float cd=dot3(scl(wi,-1.0f),Lt.dir), cone=smooth01(Lt.cosOuter,Lt.cosInner,cd); if(cone<=0.0f) continue; atten*=cone; }
        }
        PRD sh; sh.isShadow=1; sh.occluded=0; sh.miss=0; unsigned int o0,o1; packPtr(&sh,o0,o1);
        optixTrace(params.handle, add(P,scl(wi,1e-3f)), wi, 1e-3f, tmax, 0.0f,
                   OptixVisibilityMask(255), OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT, params.useInput?1u:0u,1,0, o0,o1);
        if(sh.occluded) continue;
        sum=add(sum, scl(Lt.radiance, phaseHG(dot3(wi,rd),gShaft)*atten*segTr));   // SHAFT g (decoupled)
    }
    // environment in-scatter: HDRI importance-sampled if a CDF exists, else a 2-sample uniform-sphere fallback
    // for the gradient/Preetham dome (so fog is correctly sky-lit under every sky mode). The fog-continuation
    // double-count is suppressed via the fromFogScatter flag in raygen (NOT here).
    if(params.fogSkyStr > 0.0f){
        if(params.envImportance){
            sum=add(sum, scl(mediumEnvNEE(P, rd, g, seed), params.fogSkyStr));
        } else {
            for(int s=0;s<2;s++){
                float z=1.0f-2.0f*rnd(seed), rr=sqrtf(fmaxf(0.0f,1.0f-z*z)), ph=6.28318530718f*rnd(seed);
                float3 wi=V(rr*cosf(ph), z, rr*sinf(ph));                     // uniform on the sphere, pdf=1/(4pi)
                PRD sh; sh.isShadow=1; sh.occluded=0; sh.miss=0; unsigned int o0,o1; packPtr(&sh,o0,o1);
                optixTrace(params.handle, add(P,scl(wi,1e-3f)), wi, 1e-3f, 1e16f, 0.0f,
                           OptixVisibilityMask(255), OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT, params.useInput?1u:0u,1,0, o0,o1);
                if(sh.occluded) continue;
                float3 L=skyDome(wi);                                        // dome only (sun disk is NEE'd above)
                sum=add(sum, scl(L, phaseHG(dot3(wi,rd),g) * 12.5663706f * 0.5f * params.fogSkyStr));   // *4pi(1/pdf) *0.5(avg) *strength
            }
        }
    }
    // emitter glow: in-scatter toward emissive triangle geometry (area-light NEE in the medium)
    sum = add(sum, mediumEmitNEE(P, rd, g, seed));
    // God-ray contrast: luminance-preserving power curve about a fixed pivot -> steepens the lit/shadowed
    // transition (deepens the air between shafts). fogContrast==1.0 -> identity (exact original look).
    if(params.fogContrast != 1.0f){
        const float p = 0.18f;                                   // tonal pivot (~mid in-scatter level)
        float Y = 0.2126f*sum.x + 0.7152f*sum.y + 0.0722f*sum.z; // Rec.709 luminance
        if(Y > 1e-6f){ float Yc = p*powf(Y/p, params.fogContrast); sum = scl(sum, Yc/Y); }   // scale RGB -> preserve hue
    }
    return sum;
}

// ---- triplanar projection sample (for geometry WITHOUT UVs): blend 3 axis projections by the normal ----
static __forceinline__ __device__ float3 triplanar(cudaTextureObject_t t, float3 P, float3 N, float scale, int layer)
{
    float inv = (scale>1e-4f) ? 1.0f/scale : 1.0f;          // tile every 'scale' world units
    float3 w = V(fabsf(N.x), fabsf(N.y), fabsf(N.z)); w = mul(w,w);   // sharpen the blend
    float ws = w.x+w.y+w.z+1e-6f;
    float4 cx = tex2DLayered<float4>(t, P.y*inv, P.z*inv, layer);   // x-facing -> project onto yz
    float4 cy = tex2DLayered<float4>(t, P.x*inv, P.z*inv, layer);   // y-facing -> project onto xz
    float4 cz = tex2DLayered<float4>(t, P.x*inv, P.y*inv, layer);   // z-facing -> project onto xy
    float3 r = add(add(scl(V(cx.x,cx.y,cx.z),w.x), scl(V(cy.x,cy.y,cy.z),w.y)), scl(V(cz.x,cz.y,cz.z),w.z));
    return scl(r, 1.0f/ws);
}

// ======================================================================
extern "C" __global__ void __raygen__rg()
{
    const uint3 idx=optixGetLaunchIndex();
    const unsigned int w=params.width, h=params.height;
    if(idx.x>=w || idx.y>=h) return;
    const unsigned int pix=idx.y*w + idx.x;

    unsigned int sfc = params.sampleSeed ? params.sampleSeed : params.frameIndex;   // 0 = use frameIndex (exact prior); else free-running -> decorrelate noise under motion
    unsigned int seed=(idx.x*1973u)^(idx.y*9277u)^(sfc*26699u)^0x9e3779b9u;
    float3 unitU=norm(params.cam_u), unitV=norm(params.cam_v);

    float3 col=V(0,0,0);
    float3 fogSum=V(0,0,0);                                   // per-frame fog in-scatter, separated for motion-robust smoothing
    float  alphaSum=0.0f;                                     // matte: 1 = opaque (hit/bg), 0 = transparent primary miss
    float3 aoAlb=V(0,0,0), aoNrm=V(0,0,0), aoPos=V(0,0,0);   // first-hit AOVs (denoiser guides)
    float3 aoDir=V(0,0,1);                                   // primary view direction (for the infinite-distance sky motion vector)
    float3 aoSpecAlb=V(0.04f,0.04f,0.04f); float aoRough=1.0f;
    int    aoHit=0;
    float  aoSpecHitT=0.0f;                 // reflected/refracted-ray hit distance from the FIRST specular surface (RR reproject)
    float3 specSurfPos=V(0,0,0);            // world pos of the first specular surface (the mirror) on the path
    float3 specHitPos=V(0,0,0);             // world pos the reflected/refracted ray lands on (virtual-image point, for the spec MV)
    int    specState=0;                     // 0=searching, 1=mirror found-arm reflection capture next bounce, 2=resolved
    int    gotSpecHit=0;                    // 1 once aoSpecHitT + specHitPos are valid
    const unsigned int nspp = params.spp>0u?params.spp:1u;
    for(unsigned int sp=0; sp<nspp; ++sp)
    {
        float jx, jy;
        if(params.rrEnable){ jx=params.rrJitterX; jy=params.rrJitterY; }   // deterministic per-frame jitter -> RR temporal AA
        else { jx=0.5f+(rnd(seed)-0.5f)*params.jitter; jy=0.5f+(rnd(seed)-0.5f)*params.jitter; }
        float sx=((idx.x+jx)/(float)w)*2.0f-1.0f;
        float sy=((idx.y+jy)/(float)h)*2.0f-1.0f;
        float3 ro=params.cam_eye;
        float3 dir=add(add(params.cam_w, scl(params.cam_u,sx)), scl(params.cam_v,sy));
        if(params.aperture>0.0f)
        {
            float2 ld=diskRand(seed);
            float3 lens=add(scl(unitU,ld.x*params.aperture), scl(unitV,ld.y*params.aperture));
            ro=add(ro,lens); dir=sub(dir,lens);          // keep focus plane fixed
        }
        float3 rd=norm(dir);

        float3 thr=V(1,1,1), L=V(0,0,0), Ldir=V(0,0,0); int dcap=0;   // Ldir = direct (b==0); firefly clamp only the indirect
        float3 fogS=V(0,0,0);                         // this sample's fog in-scatter (tracked separately for smoothing)
        bool addEmit=true;                            // add emission this hit? (false right after a NEE'd diffuse bounce)
        bool lastSpec=true;                           // last bounce specular/primary? -> show the sun disk on miss (else NEE owns the sun)
        bool fromFogScatter=false;                    // did THIS ray come from a fog scatter? -> its env was NEE'd, suppress env-on-miss
        int  fogScatterCount=0;                        // medium scatter events on THIS path (for fogMaxScatter cap)
        float sampA=1.0f;                             // this sample's matte (set to 0 on a transparent primary miss)
        PRD prd;
        for(unsigned int b=0; b<params.maxDepth; ++b)
        {
            if(b==1u && !dcap){ Ldir=L; dcap=1; }      // snapshot direct light before the first indirect bounce
            prd.miss=0; prd.absorbed=0; prd.seed=seed; prd.isShadow=0;
            unsigned int u0,u1; packPtr(&prd,u0,u1);
            optixTrace(params.handle, ro, rd, 1e-3f, 1e16f, 0.0f,
                       OptixVisibilityMask(255), OPTIX_RAY_FLAG_NONE, params.useInput?1u:0u,1,0, u0,u1);
            seed=prd.seed;
            if(sp==0u && b==0u){                       // capture primary-hit AOVs once
                aoDir=rd;                              // primary view direction (rd is still the primary ray here) -> sky motion vector
                if(prd.miss){ aoAlb=prd.skyColor; aoNrm=V(0,0,0); aoHit=0; }
                else { aoAlb=prd.hitAlbedo; aoNrm=prd.hitNormal; aoPos=prd.hitPos; aoHit=1; aoSpecAlb=prd.hitSpecAlb; aoRough=prd.hitRough; }
            }
            // SPECULAR REPROJECTION (RR): the bounce traced AFTER the first specular surface IS the reflected/refracted
            // ray, so prd.hitPos here is the virtual-image point. Capture its travel distance (rrHitDist) and world pos
            // (for the specular motion vector) for the FIRST specular interaction at ANY bounce -> b>=1 mirrors and
            // mirrors reached behind fog now get a valid hit-T (was hard-zero -> DLSS-RR smear). Reproduces the OLD
            // value EXACTLY for a primary-visible mirror (mirror @ b==0 -> resolved @ b==1).
            if(sp==0u && specState==1){
                aoSpecHitT = prd.miss ? 1000.0f : sqrtf(dot3(sub(prd.hitPos,specSurfPos),sub(prd.hitPos,specSurfPos)));
                specHitPos = prd.miss ? add(ro, scl(rd, 1000.0f)) : prd.hitPos;   // miss -> point ~at infinity along the reflected dir
                gotSpecHit = 1; specState = 2;          // first specular interaction resolved (do not re-arm on deeper mirrors)
            }
            // ---- homogeneous volumetric fog: free-flight single scattering (Phase 1) ----
            // sample a scatter distance; if it falls before the surface (or sky), the path scatters in the
            // medium instead of reaching the hit. The not-scattering probability == transmittance, so the
            // surface branch needs no explicit exp() (it is accounted for implicitly).
            // skip in-scatter (and its shadow rays) entirely if no light source can contribute
            bool fogHasLight = (params.sunColor.x+params.sunColor.y+params.sunColor.z > 0.0f)
                               || (params.numPtLights > 0)
                               || (params.fogEmitNEE && params.numEmitTri > 0)
                               || (params.fogSkyStr > 0.0f);
            if(params.fogEnable && params.fogDensity>0.0f){
                float hitDist = prd.miss ? 1e16f : sqrtf(dot3(sub(prd.hitPos,ro),sub(prd.hitPos,ro)));
                float ts = -__logf(1.0f-rnd(seed))/params.fogDensity;
                if(ts < hitDist){
                    float3 pT = add(ro, scl(rd, ts));
                    if(fogHasLight){                                       // only trace in-scatter shadow rays if a light exists
                        float3 ins = inscatterFog(pT, rd, seed);          // direct in-scatter (phase-weighted)
                        float3 fc = mul(mul(thr, params.fogColor), ins);  // (T*sigma_s)/pdf == albedo == fogColor
                        if(params.fogFireflyMax>0.0f){                    // clamp the in-scatter spike: b==0 fog is DIRECT and bypasses the indirect-only fireflyMax
                            float ffl = 0.2126f*fc.x + 0.7152f*fc.y + 0.0722f*fc.z;
                            if(ffl > params.fogFireflyMax) fc = scl(fc, params.fogFireflyMax/ffl);
                        }
                        L = add(L, fc);
                        fogS = add(fogS, fc);                             // same CLAMPED fc into both -> the fog EMA rebuild stays consistent
                    }
                    thr = mul(thr, params.fogColor);                      // carry single-scatter albedo to the continuation
                    addEmit = (params.fogEmitNEE && params.numEmitTri>0) ? false : true;  // emitter NEE'd -> don't re-add on continuation
                    lastSpec = false;                                     // sun in-scattered -> suppress its disk on next miss
                    fromFogScatter = true;                                // env NEE'd -> suppress env-on-miss for the continuation
                    if(params.fogSingleScatter){ if(!dcap){Ldir=L;dcap=1;} break; }   // fast biased mode: no continuation bounce
                    fogScatterCount++;                                    // cap multi-scatter (mild bias in dense fog, bounds path length)
                    if(params.fogMaxScatter>0 && fogScatterCount>=params.fogMaxScatter){ if(!dcap){Ldir=L;dcap=1;} break; }
                    rd = sampleHGdir(rd, params.fogG, rnd(seed), rnd(seed));
                    ro = pT;
                    { float p=fminf(fmaxf(maxc(thr)/fmaxf(params.fogRRStart,1e-3f),0.02f),1.0f);   // RR from the 1st scatter (unbiased)
                      if(rnd(seed)>p){ if(!dcap){Ldir=L;dcap=1;} break; } thr=scl(thr,1.0f/p); }
                    continue;                                             // this bounce was consumed by a medium scatter
                }
            }
            if(prd.miss){
                float3 envc;
                if(b==0u){                                       // primary miss = the VISIBLE background (mode-controlled)
                    if(params.bgMode==2){ envc=V(0,0,0); sampA=0.0f; }       // Transparent -> matte alpha 0
                    else if(params.bgMode==1){ envc=params.bgSolid; }        // Solid flat color
                    else { envc=add(skyDome(rd), sunDisk(rd)); }             // Environment (sky + sun disk)
                } else if(lastSpec){                              // specular bounce: see the full environment
                    envc=add(skyDome(rd), sunDisk(rd));
                } else {                                          // diffuse bounce: env owned by NEE (importance) or passive sky
                    envc = (params.envImportance || fromFogScatter) ? V(0.0f,0.0f,0.0f) : skyDome(rd);
                }
                L=add(L, mul(thr, envc)); if(!dcap){Ldir=L;dcap=1;} break;
            }
            if(addEmit) L=add(L, mul(thr, prd.emitted));   // skip if NEE already counted this light
            if(prd.absorbed){ if(!dcap){Ldir=L;dcap=1;} break; }
            if(sp==0u && specState==0 && !prd.isDiffuse){ specSurfPos=prd.hitPos; specState=1; }   // first specular SURFACE on the path -> arm reflection capture (post-fog, post-miss, non-emitter)
            if(prd.isDiffuse)                              // sun directional NEE (crisp shadows, both sphere + tri modes)
                L=add(L, mul(thr, sampleSun(prd.hitPos, prd.hitNormal, prd.hitAlbedo, seed)));
            if(params.envImportance && prd.isDiffuse)      // HDRI importance-sampled env NEE (low-noise IBL)
                L=add(L, mul(thr, sampleEnv(prd.hitPos, prd.hitNormal, prd.hitAlbedo, seed)));
            if(params.numPtLights>0 && prd.isDiffuse)      // analytic Light COMPs NEE (point/cone/distant)
                L=add(L, mul(thr, sampleLights(prd.hitPos, prd.hitNormal, prd.hitAlbedo, seed)));
            if(params.neeEnable && prd.isDiffuse)
                L=add(L, mul(thr, sampleDirect(prd.hitPos, prd.hitNormal, prd.hitAlbedo, seed)));
            thr=mul(thr, prd.attenuation);
            ro=prd.nextOrigin; rd=prd.nextDir;
            fromFogScatter = false;                        // a surface bounce: its continuation keeps the passive sky ambient on miss
            lastSpec = prd.isDiffuse ? false : true;       // diffuse did sun-NEE -> suppress its sun disk on the next miss
            addEmit = (params.neeEnable && prd.isDiffuse) ? false : true;
            if(b>=3){ float p=fmaxf(maxc(thr),0.05f); if(rnd(seed)>p) break; thr=scl(thr,1.0f/p); }
        }
        if(!dcap) Ldir=L;                             // no indirect (maxDepth 1 / broke at primary)
        float3 Lind=sub(L,Ldir);                      // indirect = light gathered at bounces b>=1
        if(params.fireflyMax > 0.0f){                 // clamp INDIRECT only -> emitters & direct highlights stay bright
            float fl = 0.2126f*Lind.x + 0.7152f*Lind.y + 0.0722f*Lind.z;
            if(fl > params.fireflyMax) Lind = scl(Lind, params.fireflyMax/fl);
        }
        col=add(col, add(Ldir,Lind));
        fogSum=add(fogSum, fogS);                    // sum this sample's fog into the per-frame fog
        alphaSum += sampA;
    }
    col=scl(col, 1.0f/(float)nspp);
    fogSum=scl(fogSum, 1.0f/(float)nspp);
    // FOG TEMPORAL STABILITY: EMA the per-frame fog into a persistent buffer that does NOT reset on camera motion
    // (only on fogReset = fog-param / resolution change). The fog haze is low-frequency, so temporally averaging it
    // removes the boiling DLSS-RR can't (volumetric in-scatter has no surface to reproject). col is rebuilt as
    // (surface = col - fogSum) + smoothed fog -> the rrColor / image / accum writes below stay untouched.
    // fogStability==0 -> a==1 -> fogOut==fogSum -> col unchanged == exact original behavior (fully reversible).
    if(params.fogEnable && params.fogStability>0.0f && params.fogAccum){
        float a = 1.0f - 0.95f*fminf(fmaxf(params.fogStability,0.0f),1.0f);   // 1=off, ~0.05=max smoothing
        float4 fp = params.fogAccum[pix];
        float3 fogOut = (params.fogReset!=0) ? fogSum
                        : add(scl(V(fp.x,fp.y,fp.z), 1.0f-a), scl(fogSum, a));
        params.fogAccum[pix] = make_float4(fogOut.x, fogOut.y, fogOut.z, 1.0f);
        col = add(sub(col, fogSum), fogOut);          // surface + smoothed fog
    }
    float frameA = alphaSum/(float)nspp;             // this frame's matte (avg over spp -> AA'd alpha edges)

    // ---- previous-frame screen location of this pixel's surface point (reproject thru prev camera) ----
    bool  haveProj=false; float ppx=0.0f, ppy=0.0f;
    if(params.validPrev){
        // SKY MOTION VECTOR FIX: surface hit -> vector from the previous eye to the world hit point (parallax).
        // sky/miss -> the primary VIEW DIRECTION instead (a point at infinity has ZERO parallax under translation),
        // reprojected through the previous camera basis -> a correct ROTATION-induced motion vector for the
        // background. Without this the sky MV was hard-zero, so under camera rotation DLSS-RR could not reproject
        // its clean sky history and the moving background boiled into pixelated noise.
        float3 dpv = aoHit ? sub(aoPos, params.prev_eye) : aoDir;
        float awp=dot3(dpv,params.prev_w)/dot3(params.prev_w,params.prev_w);
        if(awp>1e-4f){
            ppx=((dot3(dpv,params.prev_u)/dot3(params.prev_u,params.prev_u))/awp*0.5f+0.5f)*(float)w;
            ppy=((dot3(dpv,params.prev_v)/dot3(params.prev_v,params.prev_v))/awp*0.5f+0.5f)*(float)h;
            haveProj=true;
        }
    }
    float2 flow = haveProj ? make_float2(((float)idx.x+0.5f)-ppx, ((float)idx.y+0.5f)-ppy) : make_float2(0.0f,0.0f);

    // ---- SPECULAR motion vector: reproject the reflected/refracted hit point (virtual image) through the prev camera ----
    // A mirror pixel otherwise inherits the STATIC mirror-plane MV (flow above) -> the reflection swims/boils under
    // camera motion. Same scale-invariant reprojection as the surface MV, with specHitPos in place of aoPos.
    float2 flowSpec = flow;                              // default to the surface MV (used when no spec hit / behind prev cam)
    if(params.validPrev && gotSpecHit){
        float3 dps = sub(specHitPos, params.prev_eye);
        float aws  = dot3(dps,params.prev_w)/dot3(params.prev_w,params.prev_w);
        if(aws>1e-4f){
            float spx=((dot3(dps,params.prev_u)/dot3(params.prev_u,params.prev_u))/aws*0.5f+0.5f)*(float)w;
            float spy=((dot3(dps,params.prev_v)/dot3(params.prev_v,params.prev_v))/aws*0.5f+0.5f)*(float)h;
            flowSpec = make_float2(((float)idx.x+0.5f)-spx, ((float)idx.y+0.5f)-spy);
        }
    }

    // ---- temporal reprojection accumulation: running average that follows the surface ----
    float3 outc; float outA=frameA;
    if(params.taaEnable){
        float n=1.0f; float3 hist=col;
        if(haveProj){
            int ix=(int)floorf(ppx), iy=(int)floorf(ppy);
            if(ix>=0 && ix<(int)w && iy>=0 && iy<(int)h){
                float4 pc=params.taaPrevCol[(unsigned)iy*w+(unsigned)ix];
                if(pc.w>0.0f){ hist=make_float3(pc.x,pc.y,pc.z); n=fminf(pc.w+1.0f, params.taaMaxHist); }
            }
        }
        float a=1.0f/n;
        outc=add(scl(hist,1.0f-a), scl(col,a));
        params.taaCurCol[pix]=make_float4(outc.x,outc.y,outc.z,n);
    } else {
        // plain running-sum accumulation (best for a static hero shot); accum.w carries the matte sum
        float4 pa=(params.frameIndex<=1u)?make_float4(0.0f,0.0f,0.0f,0.0f):params.accum[pix];
        float3 sum=add(V(pa.x,pa.y,pa.z),col);
        float  sumA=pa.w + frameA;
        params.accum[pix]=make_float4(sum.x,sum.y,sum.z,sumA);
        outc=scl(sum, 1.0f/(float)params.frameIndex);
        outA=sumA/(float)params.frameIndex;
    }

    params.image[pix]=make_float4(outc.x,outc.y,outc.z,outA);
    params.albedo[pix]=make_float4(fminf(fmaxf(aoAlb.x,0.0f),1.0f),fminf(fmaxf(aoAlb.y,0.0f),1.0f),fminf(fmaxf(aoAlb.z,0.0f),1.0f),1.0f);
    params.normal[pix]=make_float4(aoNrm.x,aoNrm.y,aoNrm.z,0.0f);
    params.flow[pix]=make_float4(flow.x,flow.y,0.0f,0.0f);

    // ---- DLSS Ray Reconstruction G-buffer (raw color + specular/roughness/depth/motion) ----
    if(params.rrEnable){
        params.rrColor[pix]      = make_float4(col.x, col.y, col.z, 1.0f);     // raw per-frame HDR
        params.rrSpecAlbedo[pix] = make_float4(aoSpecAlb.x, aoSpecAlb.y, aoSpecAlb.z, 1.0f);
        params.rrRoughness[pix]  = aoRough;
        float3 dd = sub(aoPos, params.cam_eye);
        float  vz = dot3(dd, params.cam_w) * rsqrtf(fmaxf(dot3(params.cam_w,params.cam_w),1e-20f));  // view-space linear Z
        params.rrDepth[pix]      = aoHit ? vz : 10000.0f;
        float specW = params.rrSpecMV * (1.0f - fminf(fmaxf(aoRough*4.0f,0.0f),1.0f));   // specular-dominated (low-roughness) AND enabled; 0 for diffuse -> rrMotion unchanged
        params.rrMotion[pix]     = make_float2(flow.x + specW*(flowSpec.x-flow.x), flow.y + specW*(flowSpec.y-flow.y));
        params.rrHitDist[pix]    = aoSpecHitT;
    }
}

// ---- triangle closest-hit (input geometry: no-index soup, flat normals, default matte material) ----
extern "C" __global__ void __closesthit__tri()
{
    PRD* prd=getPRD();
    if(prd->isShadow){ prd->occluded=1; return; }
    unsigned int seed=prd->seed;

    const unsigned int pid=optixGetPrimitiveIndex();
    float4 a4=params.triVerts[3u*pid+0u], b4=params.triVerts[3u*pid+1u], c4=params.triVerts[3u*pid+2u];
    float3 v0=V(a4.x,a4.y,a4.z), v1=V(b4.x,b4.y,b4.z), v2=V(c4.x,c4.y,c4.z);
    float3 Ng=norm(cross3(sub(v1,v0),sub(v2,v0)));            // flat (geometric) normal

    float3 ro=optixGetWorldRayOrigin(), rd=norm(optixGetWorldRayDirection());
    float3 P=add(ro, scl(rd,optixGetRayTmax()));
    bool   front=dot3(rd,Ng)<0.0f;
    float3 Nf=front?Ng:scl(Ng,-1.0f);
    unsigned int i0=3u*pid;
    float4 mm = params.triMat ? params.triMat[i0] : make_float4(0.0f,0.0f,0.0f,0.0f);
    int rawtype=(int)(mm.x+0.5f); bool smoothObj=(rawtype>=10); int type=rawtype%10;   // type>=10 => per-material smooth normals
    if(params.triN && smoothObj){                            // smooth shading normal (per-object): barycentric-interp the 3 vert normals
        float2 bc=optixGetTriangleBarycentrics();
        float4 n0=params.triN[3u*pid+0u], n1=params.triN[3u*pid+1u], n2=params.triN[3u*pid+2u];
        float3 sn=norm(add(add(scl(V(n0.x,n0.y,n0.z),1.0f-bc.x-bc.y), scl(V(n1.x,n1.y,n1.z),bc.x)), scl(V(n2.x,n2.y,n2.z),bc.y)));
        Nf=(dot3(rd,sn)<0.0f)?sn:scl(sn,-1.0f);              // face the ray (use geometric front for offsets is fine on convex)
    }

    // per-object material (i0 / mm / type / smoothObj already decoded above with the normal)
    float3 color = params.triCd  ? V(params.triCd[i0].x, params.triCd[i0].y, params.triCd[i0].z) : V(0.82f,0.80f,0.78f);
    if(params.triUV || params.useBaseColor || params.showUV){
        float uu=0.0f, vv=0.0f; bool haveUV=false;
        if(params.triUV){                                    // interpolate the UV (Tex) + check it's real (non-degenerate)
            float2 bc=optixGetTriangleBarycentrics();
            float4 q0=params.triUV[3u*pid+0u], q1=params.triUV[3u*pid+1u], q2=params.triUV[3u*pid+2u];
            uu=q0.x*(1.0f-bc.x-bc.y)+q1.x*bc.x+q2.x*bc.y;
            vv=q0.y*(1.0f-bc.x-bc.y)+q1.y*bc.x+q2.y*bc.y;
            float span=fabsf(q1.x-q0.x)+fabsf(q1.y-q0.y)+fabsf(q2.x-q0.x)+fabsf(q2.y-q0.y);
            haveUV=(span>1e-5f);
        }
        bool useUV = params.triUV && (params.texMode==1 || (params.texMode==0 && haveUV));  // auto-fallback
        if(params.showUV) color = useUV ? V(uu,vv,0.0f) : V(0.6f,0.0f,0.6f);   // debug: UV gradient, or magenta = projection
        if(params.useBaseColor){
            int matID = 0;                                   // CP5: per-material texture layer from triCd.w (Cd alpha)
            if(params.triCd && params.numMaterials>1){ int m=(int)(params.triCd[i0].w+0.5f); matID = m<0?0:(m>=params.numMaterials?params.numMaterials-1:m); }
            float3 tx;
            if(useUV){ float4 c=tex2DLayered<float4>(params.baseColorTex,uu,vv,matID); tx=V(c.x,c.y,c.z); }
            else       tx=triplanar(params.baseColorTex, P, Nf, params.projScale, matID);
            color = mul(color, tx);                          // modulate the base color by the map
        }
    }
    float rough  = mm.y;
    float ior    = (mm.z>1.01f) ? mm.z : 1.5f;
    float estr   = (mm.w>0.0f)  ? mm.w : 1.0f;

    prd->miss=0; prd->emitted=V(0,0,0); prd->absorbed=0;
    prd->hitNormal=Nf; prd->hitPos=P;

    if(type==3){                                             // emitter
        prd->emitted=scl(color,estr); prd->absorbed=1; prd->isDiffuse=0;
        prd->hitAlbedo=color; prd->hitSpecAlb=V(0,0,0); prd->hitRough=1.0f; prd->seed=seed; return;
    }
    else if(type==1){                                        // metal
        float3 r=reflect3(rd,Nf);
        if(rough>0.0f){ float3 j=cosineHemisphere(Nf,rnd(seed),rnd(seed)); r=norm(add(r,scl(sub(j,r),rough))); }
        prd->nextOrigin=add(P,scl(Nf,1e-3f)); prd->nextDir=norm(r); prd->attenuation=color;
        prd->hitAlbedo=color; prd->hitSpecAlb=color; prd->hitRough=fmaxf(rough,0.02f); prd->isDiffuse=0;
    }
    else if(type==2){                                        // glass (dielectric)
        float eta=front?(1.0f/ior):ior; float ci=fminf(-dot3(rd,Nf),1.0f);
        float3 T; bool can=refract3(rd,Nf,eta,T); float F=can?schlick(ci,ior):1.0f;
        if(!can||rnd(seed)<F){ prd->nextDir=norm(reflect3(rd,Nf)); prd->nextOrigin=add(P,scl(Nf,1e-3f)); }
        else                { prd->nextDir=norm(T);                prd->nextOrigin=sub(P,scl(Nf,1e-3f)); }
        prd->attenuation=color;
        prd->hitAlbedo=color; prd->hitSpecAlb=V(0.04f,0.04f,0.04f); prd->hitRough=0.05f; prd->isDiffuse=0;
    }
    else {                                                   // lambert (diffuse)
        prd->nextDir=cosineHemisphere(Nf,rnd(seed),rnd(seed)); prd->nextOrigin=add(P,scl(Nf,1e-3f)); prd->attenuation=color;
        prd->hitAlbedo=color; prd->hitSpecAlb=V(0.04f,0.04f,0.04f); prd->hitRough=1.0f; prd->isDiffuse=1;
    }
    prd->seed=seed;
}

extern "C" __global__ void __miss__ms()
{
    PRD* prd=getPRD();
    prd->miss=1;
    prd->skyColor=skyColor(norm(optixGetWorldRayDirection()));
}

extern "C" __global__ void __closesthit__ch()
{
    PRD* prd=getPRD();
    if(prd->isShadow){ prd->occluded=1; return; }   // occlusion query: just report a hit
    unsigned int seed=prd->seed;

    const unsigned int pid=optixGetPrimitiveIndex();
    float4 sph; optixGetSphereData(&sph);            // {center.xyz, radius}
    float3 center=V(sph.x,sph.y,sph.z);

    float3 ro=optixGetWorldRayOrigin(), rd=norm(optixGetWorldRayDirection());
    float  t=optixGetRayTmax();
    float3 P=add(ro, scl(rd,t));
    float3 N=norm(sub(P,center));
    bool   front=dot3(rd,N)<0.0f;
    float3 Nf=front?N:scl(N,-1.0f);

    DemoMaterial m=params.materials[pid];
    prd->miss=0; prd->emitted=V(0,0,0); prd->absorbed=0;
    prd->isDiffuse = (m.type==0) ? 1 : 0;
    prd->hitNormal=Nf;
    prd->hitPos=P;
    prd->hitSpecAlb = (m.type==1) ? m.albedo : (m.type==3) ? V(0.0f,0.0f,0.0f) : V(0.04f,0.04f,0.04f);
    prd->hitRough   = (m.type==1) ? fmaxf(m.fuzz,0.02f) : (m.type==2) ? 0.05f : 1.0f;
    prd->hitAlbedo = (m.type==3) ? make_float3(fminf(m.emission.x,1.0f),fminf(m.emission.y,1.0f),fminf(m.emission.z,1.0f))
                   : (m.type==2) ? V(1.0f,1.0f,1.0f) : m.albedo;

    if(m.type==3)                                    // emissive (light)
    {
        prd->emitted=m.emission; prd->absorbed=1; prd->seed=seed; return;
    }
    else if(m.type==1)                               // metal
    {
        float3 r=reflect3(rd,Nf);
        if(m.fuzz>0.0f){ float3 j=cosineHemisphere(Nf,rnd(seed),rnd(seed)); r=norm(add(r, scl(sub(j,r),m.fuzz))); }
        prd->nextOrigin=add(P, scl(Nf,1e-3f));
        prd->nextDir=norm(r);
        prd->attenuation=m.albedo;
    }
    else if(m.type==2)                               // dielectric (glass)
    {
        float eta = front ? (1.0f/m.ior) : m.ior;
        float ci=fminf(-dot3(rd,Nf),1.0f);
        float3 T; bool can=refract3(rd,Nf,eta,T);
        float F = can ? schlick(ci,m.ior) : 1.0f;
        if(!can || rnd(seed)<F){ prd->nextDir=norm(reflect3(rd,Nf)); prd->nextOrigin=add(P,scl(Nf,1e-3f)); }
        else                  { prd->nextDir=norm(T);               prd->nextOrigin=sub(P,scl(Nf,1e-3f)); }
        prd->attenuation=m.albedo;                   // white = clear glass
    }
    else                                             // lambert (diffuse)
    {
        prd->nextDir=cosineHemisphere(Nf,rnd(seed),rnd(seed));
        prd->nextOrigin=add(P, scl(Nf,1e-3f));
        prd->attenuation=m.albedo;                   // cosine pdf cancels with 1/pi
    }
    prd->seed=seed;
}
