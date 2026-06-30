#include "RRDenoiser.h"
#include <cstring>
#include <cuda.h>

#include "nvsdk_ngx.h"
#include "nvsdk_ngx_helpers.h"
#include "nvsdk_ngx_helpers_dlssd_cuda.h"

// Random project UUID (fine for R&D; a registered ID is only needed at commercial ship).
static const char* kProjId = "dddbee68-a452-4fab-9371-f9575480a154";

RRDenoiser::RRDenoiser()
	: myInit(false), myLastResult(0), myW(0), myH(0), myHandle(0), myParams(0), myOutArr(0), myOutSurf(0)
{
	for (int i = 0; i < NINPUT; i++) { myArr[i] = 0; myTex[i] = 0; }
}
RRDenoiser::~RRDenoiser() { shutdown(); }

bool RRDenoiser::init(const wchar_t* featureDllDir)
{
	if (myInit) return true;
	const wchar_t* paths[1] = { featureDllDir };
	NVSDK_NGX_FeatureCommonInfo fci; memset(&fci, 0, sizeof(fci));
	fci.PathListInfo.Path = paths; fci.PathListInfo.Length = 1;

	NVSDK_NGX_Result r = NVSDK_NGX_CUDA_Init_with_ProjectID(
		kProjId, NVSDK_NGX_ENGINE_TYPE_CUSTOM, "1.0.0", featureDllDir, &fci);
	myLastResult = (int)r;
	if (r != NVSDK_NGX_Result_Success) return false;

	NVSDK_NGX_Parameter* p = nullptr;
	r = NVSDK_NGX_CUDA_GetCapabilityParameters(&p);
	myLastResult = (int)r;
	if (r != NVSDK_NGX_Result_Success) return false;

	myParams = p;
	myInit = true;
	return true;
}

static cudaTextureObject_t mkTex(int w, int h, const cudaChannelFormatDesc& fmt, cudaArray_t* outA)
{
	cudaArray_t a = 0; cudaMallocArray(&a, &fmt, w, h, cudaArrayDefault);
	cudaResourceDesc rd; memset(&rd, 0, sizeof(rd)); rd.resType = cudaResourceTypeArray; rd.res.array.array = a;
	cudaTextureDesc td; memset(&td, 0, sizeof(td)); td.readMode = cudaReadModeElementType; td.filterMode = cudaFilterModePoint; td.normalizedCoords = 0;
	cudaTextureObject_t t = 0; cudaCreateTextureObject(&t, &rd, &td, nullptr);
	*outA = a; return t;
}

void RRDenoiser::freeArrays()
{
	for (int i = 0; i < NINPUT; i++) {
		if (myTex[i]) { cudaDestroyTextureObject(myTex[i]); myTex[i] = 0; }
		if (myArr[i]) { cudaFreeArray(myArr[i]); myArr[i] = 0; }
	}
	if (myOutSurf) { cudaDestroySurfaceObject(myOutSurf); myOutSurf = 0; }
	if (myOutArr) { cudaFreeArray(myOutArr); myOutArr = 0; }
}

bool RRDenoiser::ensure(int W, int H)
{
	if (!myInit) return false;
	if (myHandle && W == myW && H == myH) return true;

	if (myHandle) { NVSDK_NGX_CUDA_ReleaseFeature((NVSDK_NGX_Handle*)myHandle); myHandle = 0; }
	freeArrays();
	myW = W; myH = H;

	cudaChannelFormatDesc f4 = cudaCreateChannelDesc<float4>();
	cudaChannelFormatDesc f2 = cudaCreateChannelDesc<float2>();
	cudaChannelFormatDesc f1 = cudaCreateChannelDesc<float>();
	myTex[COLOR] = mkTex(W, H, f4, &myArr[COLOR]);
	myTex[DALB]  = mkTex(W, H, f4, &myArr[DALB]);
	myTex[SALB]  = mkTex(W, H, f4, &myArr[SALB]);
	myTex[NRM]   = mkTex(W, H, f4, &myArr[NRM]);
	myTex[RGH]   = mkTex(W, H, f1, &myArr[RGH]);
	myTex[DEP]   = mkTex(W, H, f1, &myArr[DEP]);
	myTex[MV]    = mkTex(W, H, f2, &myArr[MV]);
	myTex[HIT]   = mkTex(W, H, f1, &myArr[HIT]);

	cudaMallocArray(&myOutArr, &f4, W, H, cudaArraySurfaceLoadStore);
	cudaResourceDesc ord; memset(&ord, 0, sizeof(ord)); ord.resType = cudaResourceTypeArray; ord.res.array.array = myOutArr;
	cudaCreateSurfaceObject(&myOutSurf, &ord);

	NVSDK_NGX_Parameter* params = (NVSDK_NGX_Parameter*)myParams;
	NVSDK_NGX_Parameter_SetUI(params, NVSDK_NGX_Parameter_RayReconstruction_Hint_Render_Preset_DLAA,
	                          (unsigned)NVSDK_NGX_RayReconstruction_Hint_Render_Preset_D);

	NVSDK_NGX_DLSSD_Create_Params cp; memset(&cp, 0, sizeof(cp));
	cp.InDenoiseMode   = NVSDK_NGX_DLSS_Denoise_Mode_DLUnified;
	cp.InRoughnessMode = NVSDK_NGX_DLSS_Roughness_Mode_Unpacked;
	cp.InUseHWDepth    = NVSDK_NGX_DLSS_Depth_Type_Linear;
	cp.InWidth = W; cp.InHeight = H; cp.InTargetWidth = W; cp.InTargetHeight = H;
	cp.InPerfQualityValue   = NVSDK_NGX_PerfQuality_Value_DLAA;
	cp.InFeatureCreateFlags = NVSDK_NGX_DLSS_Feature_Flags_IsHDR | NVSDK_NGX_DLSS_Feature_Flags_MVLowRes;

	CUcontext ctx = 0; cuCtxGetCurrent(&ctx);   // share whatever context is current (TD's)
	NVSDK_NGX_CUDA_DLSSD_Create_Params ccp; memset(&ccp, 0, sizeof(ccp));
	ccp.Feature = cp; ccp.InCUContext = (void*)ctx; ccp.InCUStream = 0;

	NVSDK_NGX_Handle* h = nullptr;
	NVSDK_NGX_Result r = NGX_CUDA_CREATE_DLSSD_EXT(&h, params, &ccp);
	myLastResult = (int)r;
	if (r != NVSDK_NGX_Result_Success) return false;
	myHandle = h;
	return true;
}

bool RRDenoiser::evaluate(const float4* color, const float4* dAlb, const float4* sAlb, const float4* nrm,
                          const float* rgh, const float* dep, const float2* mv, const float* hit,
                          float4* outDenoised, const float* view16, const float* proj16,
                          float jx, float jy, int reset,
                          float mvScaleX, float mvScaleY, int invertY, cudaStream_t s)
{
	if (!myInit || !myHandle) return false;
	const int W = myW, H = myH;
	const size_t r4 = (size_t)W * sizeof(float4), r2 = (size_t)W * sizeof(float2), r1 = (size_t)W * sizeof(float);

	cudaMemcpy2DToArrayAsync(myArr[COLOR], 0, 0, color, r4, r4, H, cudaMemcpyDeviceToDevice, s);
	cudaMemcpy2DToArrayAsync(myArr[DALB],  0, 0, dAlb,  r4, r4, H, cudaMemcpyDeviceToDevice, s);
	cudaMemcpy2DToArrayAsync(myArr[SALB],  0, 0, sAlb,  r4, r4, H, cudaMemcpyDeviceToDevice, s);
	cudaMemcpy2DToArrayAsync(myArr[NRM],   0, 0, nrm,   r4, r4, H, cudaMemcpyDeviceToDevice, s);
	cudaMemcpy2DToArrayAsync(myArr[RGH],   0, 0, rgh,   r1, r1, H, cudaMemcpyDeviceToDevice, s);
	cudaMemcpy2DToArrayAsync(myArr[DEP],   0, 0, dep,   r1, r1, H, cudaMemcpyDeviceToDevice, s);
	cudaMemcpy2DToArrayAsync(myArr[MV],    0, 0, mv,    r2, r2, H, cudaMemcpyDeviceToDevice, s);
	cudaMemcpy2DToArrayAsync(myArr[HIT],   0, 0, hit,   r1, r1, H, cudaMemcpyDeviceToDevice, s);

	NVSDK_NGX_CUDA_DLSSD_Eval_Params ep; memset(&ep, 0, sizeof(ep));
	ep.pInColor = &myTex[COLOR]; ep.pInOutput = &myOutSurf; ep.pInDepth = &myTex[DEP]; ep.pInMotionVectors = &myTex[MV];
	ep.pInDiffuseAlbedo = &myTex[DALB]; ep.pInSpecularAlbedo = &myTex[SALB]; ep.pInNormals = &myTex[NRM]; ep.pInRoughness = &myTex[RGH];
	ep.pInSpecularHitDistance = &myTex[HIT];
	ep.pInWorldToViewMatrix = (float*)view16; ep.pInViewToClipMatrix = (float*)proj16;
	ep.InJitterOffsetX = jx; ep.InJitterOffsetY = jy;
	ep.InReset = reset;
	ep.InMVScaleX = mvScaleX;          // host passes flowscale/W (sign + scale tunable from the TOP)
	ep.InMVScaleY = mvScaleY;
	ep.InIndicatorInvertYAxis = invertY;
	ep.InRenderSubrectDimensions.Width = (unsigned)W; ep.InRenderSubrectDimensions.Height = (unsigned)H;

	NVSDK_NGX_Result r = NGX_CUDA_EVALUATE_DLSSD_EXT((NVSDK_NGX_Handle*)myHandle, (NVSDK_NGX_Parameter*)myParams, &ep);
	myLastResult = (int)r;
	if (r != NVSDK_NGX_Result_Success) return false;

	cudaMemcpy2DFromArrayAsync(outDenoised, r4, myOutArr, 0, 0, r4, H, cudaMemcpyDeviceToDevice, s);
	return true;
}

void RRDenoiser::shutdown()
{
	if (myHandle) { NVSDK_NGX_CUDA_ReleaseFeature((NVSDK_NGX_Handle*)myHandle); myHandle = 0; }
	freeArrays();
	if (myParams) { NVSDK_NGX_CUDA_DestroyParameters((NVSDK_NGX_Parameter*)myParams); myParams = 0; }
	// IMPORTANT (reload-safety): NVSDK_NGX_CUDA_Shutdown() resets/destroys CUDA driver state on
	// the shared context, which breaks the *reloaded* plugin's optixDeviceContextCreate (observed:
	// ready=0/optixOK=0 after an unloadplugin reload). We release per-feature state only and leave
	// NGX resident for the process lifetime. (NGX re-init on the next load is idempotent.)
	myInit = false;
}
