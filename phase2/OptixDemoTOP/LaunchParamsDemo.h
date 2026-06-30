#pragma once
// Shared between host (OptixDemoTOP.cpp) and device (demo_programs.cu).
// Standalone procedural OptiX path tracer — a field of spheres, no external geometry.
#include <vector_types.h>
#include <optix_types.h>
#include <texture_types.h>   // cudaTextureObject_t (HDRI environment)

// One material per sphere (indexed by primitive index in the closest-hit program).
struct DemoMaterial
{
    float3  albedo;      // diffuse / reflection tint (glass: white = clear)
    float3  emission;    // emissive radiance (0,0,0 = not a light)
    int     type;        // 0 = lambert, 1 = metal, 2 = dielectric, 3 = emissive
    float   fuzz;        // metal roughness (0 = mirror)
    float   ior;         // dielectric index of refraction (~1.5)
    float   _pad;
};

// Analytic delta light from a referenced TD Light COMP (point / cone / distant).
struct PTLight
{
    float3  pos;         // world position (point / cone)
    float3  dir;         // normalized axis the light points along (cone axis / distant direction)
    float3  radiance;    // emitted color * dimmer (raw TD units; master multiplier applied on device)
    int     type;        // 0 = point, 1 = cone (spot), 2 = distant (directional)
    float   radius;      // soft-shadow radius (world units, point/cone) or angular radians (distant); 0 = hard
    float   cosInner;    // cone: cos(inner half-angle)
    float   cosOuter;    // cone: cos(outer half-angle)
};


struct LaunchParamsDemo
{
    float4*                 accum;        // HDR running-sum buffer (W*H)
    float4*                 image;        // output linear HDR (W*H) -> TD, tonemapped downstream
    float4*                 albedo;       // first-hit surface albedo AOV (OptiX denoiser guide)
    float4*                 normal;       // first-hit world normal AOV (OptiX denoiser guide)
    unsigned int            width;
    unsigned int            height;
    unsigned int            frameIndex;   // 1-based accumulated frame count (progressive convergence)
    unsigned int            sampleSeed;   // free-running RNG seed source (0 = use frameIndex == exact prior). Under camera motion
                                          //   frameIndex is pinned to 1, freezing the noise pattern; a free-running counter here
                                          //   decorrelates per-frame noise so DLSS-RR + the fog EMA can actually average it.
    unsigned int            spp;          // samples per pixel this frame
    unsigned int            maxDepth;     // max path bounces
    OptixTraversableHandle  handle;       // sphere GAS

    // thin-lens camera basis (cam_w length = focus distance)
    float3                  cam_eye;
    float3                  cam_u;        // right  * halfWidth  (at focus plane)
    float3                  cam_v;        // up     * halfHeight (at focus plane)
    float3                  cam_w;        // forward * focusDist
    float                   aperture;     // lens radius for depth of field (0 = pinhole)
    // previous-frame camera (temporal denoiser motion vectors / flow)
    float3                  prev_eye;
    float3                  prev_u;
    float3                  prev_v;
    float3                  prev_w;
    int                     validPrev;
    float4*                 flow;         // output: per-pixel screen-space motion (xy used)

    // temporal reprojection accumulation (TAA) — accumulate radiance through camera motion
    float4*                 taaPrevCol;   // prev accumulated radiance (.xyz) + history count (.w)
    float4*                 taaCurCol;    // output: current accumulation
    int                     taaEnable;
    float                   taaMaxHist;   // cap on accumulated frames (lower = ghosts fade faster)

    // DLSS Ray Reconstruction G-buffer (written by raygen when rrEnable; consumed by NGX)
    int                     rrEnable;
    float4*                 rrColor;       // raw per-frame HDR radiance (no temporal accumulation)
    float4*                 rrSpecAlbedo;  // specular albedo (F0)
    float*                  rrRoughness;   // linear roughness
    float*                  rrDepth;       // linear view-space depth (>0 in front of camera)
    float2*                 rrMotion;      // screen-space motion vectors (pixels)
    float*                  rrHitDist;     // specular hit distance (reflection reprojection)
    float                   rrJitterX;     // per-frame subpixel jitter [0,1] (RR temporal AA alignment)
    float                   rrJitterY;
    float                   rrSpecMV;      // specular-MV / cross-bounce spec-hitT strength [0..1] (0 = exact prior rrMotion)

    DemoMaterial*           materials;    // per-sphere, parallel to the GAS primitive order

    // input geometry (triangle soup from a TD positions texture)
    float4*                 triVerts;     // soup verts (float4, xyz used) — triangle closest-hit reads 3*pid..+2
    int                     useInput;     // 1 = render the input triangle GAS (SBT record 1); 0 = procedural spheres
    float4*                 triCd;        // per-vertex base/emission color (Cd attr), or null -> default grey
    float4*                 triMat;       // per-vertex (type, roughness, ior, emitStrength), or null -> lambert
    float4*                 triN;         // per-vertex smooth normal (N attr), or null -> flat geometric normal
    float4*                 triUV;        // per-vertex UV (Tex attribute; xy used), or null -> untextured
    int                     showUV;       // debug: render the texture coords as color (red/green UV, magenta=projection)

    // base color: LAYERED 2D texture array (input 5), one layer per material, keyed by triCd.w matID
    // (numMaterials==1 => a single global map = the pre-CP5 behavior). UV-sampled if geometry has UVs, else triplanar.
    cudaTextureObject_t     baseColorTex; // layered texObject; 0 = no map
    int                     useBaseColor; // 1 = a base color map is wired
    float                   projScale;    // triplanar projection: world units per texture tile
    int                     texMode;      // 0 = auto (UV if present, else projection), 1 = force UV, 2 = force projection
    int                     numMaterials; // CP5: number of layers in baseColorTex (matID clamped to [0,numMaterials-1])

    // Next-event estimation (direct light sampling of the emissive spheres)
    int                     neeEnable;
    int                     numLights;
    float                   totalPower;   // sum of light powers (for the CDF)
    float4*                 lightPos;     // xyz = center, w = radius
    float4*                 lightEmit;    // xyz = emission
    float*                  lightCDF;     // cumulative power, size numLights (power-weighted pick)

    // analytic Light COMPs (point/cone/distant) — NEE, looped every diffuse bounce (pdf=1).
    // Data arrives via the render's "Lightdata" STRING param (parsed host-side -> this device buffer),
    // NOT a TOP input — feeding a Python Script TOP into a CUDA input deadlocks TD's cook thread.
    PTLight*                ptLights;
    int                     numPtLights;
    float                   lightMaster;   // master intensity multiplier (interprets TD's small dimmer units)

    float3                  sunDir;       // normalized direction toward the sun
    float3                  sunColor;     // sun radiance / irradiance (NEE key light + sharp disk)
    float                   sunAngle;     // sun angular radius (radians) for soft shadows (0 = hard)
    float                   skyStrength;  // sky dome / HDRI multiplier (exposure)
    float3                  skyZenith;    // sky gradient: top color
    float3                  skyHorizon;   // sky gradient: horizon color
    int                     skyMode;      // 0 = gradient sky, 1 = equirect HDRI, 2 = Preetham physical sky
    cudaTextureObject_t     hdriTex;      // equirectangular HDRI (bilinear, U-wrap / V-clamp), 0 = none
    float                   hdriRot;      // HDRI horizontal rotation in turns [0,1) (added to u)

    // --- ENV 3: HDRI importance sampling (2D CDF, NEE on diffuse) ---
    int                     envImportance; // 1 = importance-sample the HDRI (diffuse env owned by NEE)
    int                     envW, envH;    // CDF grid resolution
    float*                  envCondCdf;    // [envH*envW] per-row conditional CDF over columns (normalized)
    float*                  envMargCdf;    // [envH] marginal CDF over rows (normalized)
    float*                  envFunc;       // [envH*envW] importance function (lum*sinθ), for the pdf
    float                   envFuncInt;    // sum of envFunc (pdf normalization denominator)

    // --- ENV 4: Preetham physical sky (skyMode==2) ---
    float3                  perezA, perezB, perezC, perezD, perezE; // per-channel (x,y,Y) Perez coefficients
    float3                  perezZenith;   // zenith (xz, yz, Yz)
    float3                  perezNorm;     // F(0, thetaS) per channel (normalization)
    int                     bgMode;       // primary-miss background: 0=Environment, 1=Solid, 2=Transparent
    float3                  bgSolid;      // Solid-mode flat background color
    float                   jitter;       // AA subpixel jitter amount (0 = stable/aliased, 1 = full AA)
    float                   fireflyMax;   // clamp per-sample radiance luminance (0 = off) -> kills fireflies at source

    // --- Volumetric fog (Phase 1: homogeneous, free-flight single scattering) ---
    int                     fogEnable;    // 1 = participating medium on
    float                   fogDensity;   // extinction sigma_t (1/world-units); mean free path = 1/sigma_t
    float3                  fogColor;     // single-scatter albedo (sigma_s/sigma_t) per channel [0,1]
    float                   fogG;         // Henyey-Greenstein anisotropy [-1,1] (0 = isotropic, >0 = forward/sun halo)

    // --- Emitter glow: medium in-scatter NEE toward emissive triangle geometry (v1: uniform triangle pick) ---
    int                     fogEmitNEE;   // 1 = also NEE emissive geometry from the medium (needs useInput + emitters)
    int                     numEmitTri;   // count of emissive triangles in the soup (0 = list empty)
    int*                    emitTriIdx;   // [numEmitTri] primitive index of each emissive triangle (host-built, uploaded)

    // --- Fog quality / performance controls ---
    float                   fogSkyStr;        // sky/env in-scatter strength (0 = no sky in fog, 1 = physical) -> tames bright-HDRI wash
    int                     fogSingleScatter; // 1 = single-scatter only (fast, biased): add in-scatter, attenuate, terminate
    int                     fogMaxScatter;    // hard cap on medium scatter events per path (0 = uncapped)
    float                   fogRRStart;       // throughput level below which fog Russian-roulette kicks in (lower = kill sooner)
    float                   fogFireflyMax;    // clamp the fog in-scatter luminance (0 = off, exact prior). The b==0 fog in-scatter
                                              //   lands in the DIRECT bucket and BYPASSES the indirect-only fireflyMax, so bright
                                              //   single-sample env/emitter in-scatter survives as 1-frame fog fireflies; this kills them.

    // --- Fog temporal stability: motion-robust fog smoothing (stops the fog haze boiling under camera motion) ---
    // DLSS-RR is a SURFACE denoiser; volumetric fog in-scatter has no surface to reproject, so it boils when the
    // camera moves. We EMA the per-frame fog into a persistent buffer that survives camera motion (resets only on
    // fog-param / resolution change), then feed RR surface+smoothed-fog. fogStability=0 == byte-for-byte original.
    float4*                 fogAccum;         // persistent per-pixel fog in-scatter EMA (W*H); NOT reset on camera motion
    int                     fogReset;         // 1 = reset the fog EMA this frame (res / fog-param change ONLY)
    float                   fogStability;     // 0 = off (per-frame fog, exact original); ->1 = heavier temporal smoothing
    // --- God-ray "Crispiness": decoupled shaft anisotropy + in-scatter contrast curve (RESEARCH §A.3) ---
    float                   fogShaftG;        // HG g used ONLY at the sun + PTLight in-scatter (shaft tightness). Host resolves the
                                              //   -2 sentinel to fogG, so fogShaftG==fogG == byte-exact original shafts.
    float                   fogContrast;      // luminance-preserving contrast exponent on the in-scatter term about pivot 0.18; 1.0 = off
};
