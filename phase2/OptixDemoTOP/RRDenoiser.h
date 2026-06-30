#pragma once
// NVIDIA NGX CUDA DLSS Ray Reconstruction wrapper. Pure CUDA — runs on the caller's
// CURRENT CUDA context (share TouchDesigner's: call from execute() inside begin/endCUDAOperations).
// All eval inputs are LINEAR device buffers at render resolution; this wrapper copies them into
// internal cudaArrays + texture objects each frame and hands the texture objects to NGX.
// DLAA mode (output resolution == render resolution: denoise + AA, no upscaling).
#include <cuda_runtime.h>

class RRDenoiser
{
public:
	RRDenoiser();
	~RRDenoiser();

	// NGX init (once). featureDllDir = directory that contains nvngx_dlssd.dll (NGX searches it).
	bool init(const wchar_t* featureDllDir);
	bool initialized() const { return myInit; }
	int  lastResult() const { return myLastResult; }   // last NVSDK_NGX_Result (0x1 == Success)

	// (Re)create the RR feature + (re)allocate input/output arrays for W x H. Safe to call every frame.
	bool ensure(int W, int H);
	bool ready() const { return myInit && myHandle != 0; }

	// Run RR for one frame. All inputs LINEAR device buffers at W x H. view16/proj16 = row-major 4x4.
	// jitter in pixels. reset=1 on a scene cut. Result written to outDenoised (linear float4, W*H).
	bool evaluate(const float4* color, const float4* diffuseAlbedo, const float4* specularAlbedo,
	              const float4* normal, const float* roughness, const float* depth,
	              const float2* motion, const float* specularHitDist,
	              float4* outDenoised,
	              const float* view16, const float* proj16,
	              float jitterX, float jitterY, int reset,
	              float mvScaleX, float mvScaleY, int invertY, cudaStream_t stream);

	void shutdown();

private:
	enum { COLOR = 0, DALB, SALB, NRM, RGH, DEP, MV, HIT, NINPUT };
	bool   myInit;
	int    myLastResult;
	int    myW, myH;
	void*  myHandle;     // NVSDK_NGX_Handle*
	void*  myParams;     // NVSDK_NGX_Parameter*
	cudaArray_t         myArr[NINPUT];
	cudaTextureObject_t myTex[NINPUT];
	cudaArray_t         myOutArr;
	cudaSurfaceObject_t myOutSurf;
	void freeArrays();
};
