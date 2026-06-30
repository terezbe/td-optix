/* Standalone OptiX path-tracing showcase for TouchDesigner (CPlusPlus TOP).
 * Procedural sphere scene generated entirely in code — no external geometry. */
#include "TOP_CPlusPlusBase.h"
#include "cuda_runtime.h"
#include <optix.h>

using namespace TD;

#include "LaunchParamsDemo.h"
#include "RRDenoiser.h"

class OptixDemoTOP : public TOP_CPlusPlusBase
{
public:
	OptixDemoTOP(const OP_NodeInfo* info, TOP_Context* context);
	virtual ~OptixDemoTOP();

	virtual void		getGeneralInfo(TOP_GeneralInfo*, const OP_Inputs*, void*) override;
	virtual void		execute(TOP_Output*, const OP_Inputs*, void*) override;
	virtual int32_t		getNumInfoCHOPChans(void*) override;
	virtual void		getInfoCHOPChan(int32_t, OP_InfoCHOPChan*, void*) override;
	virtual void		getErrorString(OP_String*, void*) override;
	virtual void		setupParameters(OP_ParameterManager*, void*) override;
	virtual void		pulsePressed(const char*, void*) override;

private:
	const OP_NodeInfo*		myNodeInfo;
	int32_t					myExecuteCount;
	cudaStream_t			myStream;
	TOP_Context*			myContext;
	const char*				myError;

	// OptiX
	OptixDeviceContext		myOptixContext;
	bool					myOptixTried;
	int						myOptixResult;
	bool					myReady;
	int						myReadyResult;     // -999 = not tried
	OptixModule				myModule;
	OptixModule				mySphereIS;        // built-in sphere intersector
	OptixProgramGroup		myRaygenPG, myMissPG, myHitPG;
	OptixPipeline			myPipeline;
	OptixShaderBindingTable	mySbt;
	CUdeviceptr				myParamsBuf;
	char					myLog[2048];

	// procedural scene
	CUdeviceptr				myCenters;     // float3[ mySphereCount ]
	CUdeviceptr				myRadii;       // float [ mySphereCount ]
	CUdeviceptr				myMaterials;   // DemoMaterial[ mySphereCount ]
	unsigned int			mySphereCount;
	OptixTraversableHandle	myGasHandle;
	CUdeviceptr				myGasBuffer;

	// next-event estimation lights (emissive spheres, power-weighted)
	CUdeviceptr				myLightPos;    // float4[ myNumLights ] (xyz=center, w=radius)
	CUdeviceptr				myLightEmit;   // float4[ myNumLights ] (xyz=emission)
	CUdeviceptr				myLightCDF;    // float [ myNumLights ] (cumulative power)
	int						myNumLights;
	float					myTotalPower;

	// input triangle-soup geometry (live from a TD positions texture)
	OptixProgramGroup		myTriHitPG;
	float4*					myTriVerts;        // device soup verts (padded to input texture size)
	int						myTriVertCap;      // allocated capacity (padded texels)
	float4*					myTriCd;           // per-vertex color (Cd)
	float4*					myTriMat;          // per-vertex (type, roughness, ior, emitStrength)
	float4*					myTriN;            // per-vertex smooth normal (N)
	float4*					myTriUV;           // per-vertex UV (Tex attribute)
	CUdeviceptr				myTriGasBuffer;
	CUdeviceptr				myTriGasTemp;
	size_t					myTriGasTempSize;
	size_t					myTriGasOutSize;
	unsigned int			myTriGasN;         // vert count the current tri GAS was built for
	int						myTriGasRefits;    // consecutive refits; force a full rebuild every 30 to keep BVH quality
	// Emitter glow: emissive-triangle index list (v1: host readback of triMat -> scan -> upload, gated on Fogemitnee)
	int*					myEmitTriIdx;      // device [cap] emissive-triangle primitive indices
	int						myEmitCap;         // device/host idx capacity (padded verts)
	int						myNumEmitTri;      // emissive triangle count this cook
	float4*					myTriMatHost;      // host scratch: readback of triMat to find emissive tris
	int*					myEmitIdxHost;     // host scratch: emissive index list before upload
	int						myDbgScanTri;      // diag: #triangles scanned for emitters last cook (-1 = block skipped)
	float					myDbgFirstMat;     // diag: triMat[0].x readback (sanity: is material data valid?)
	OptixTraversableHandle	myTriGasHandle;

	// accumulation / output
	float4*					myImage;
	CUdeviceptr				myAccum;
	int						myW, myH;
	CUdeviceptr				myFogAccum;   // persistent fog in-scatter EMA (motion-robust fog smoothing; NOT reset on camera motion)
	bool					myFogReset;   // reset the fog EMA next cook (res / fog-param change ONLY — never on camera motion)
	unsigned int			myFrameIndex;

	// OptiX AI denoiser
	OptixDenoiser			myDenoiser;
	CUdeviceptr				myDenoiserState, myDenoiserScratch, myIntensity;
	size_t					myStateSize, myScratchSize;
	float4*					myAlbedo;
	float4*					myNormal;
	float4*					myDenoised;
	float4*					myFlow;            // per-pixel motion vectors (temporal denoiser)
	float4*					myPrevDenoised;    // previous frame's denoised output (temporal)
	bool					myHavePrevDenoised;
	float4*					myTaaCol0;         // temporal reprojection accumulation (ping-pong)
	float4*					myTaaCol1;
	int						myTaaParity;

	// DLSS Ray Reconstruction (NGX CUDA) — destructor calls NVSDK_NGX_CUDA_Shutdown on unload
	RRDenoiser				myRR;
	bool					myRRInitTried;
	float4*					myRRColor;     // raw HDR for RR
	float4*					myRRSpecAlb;   // specular albedo
	float*					myRRRough;     // roughness
	float*					myRRDepth;     // linear depth
	float2*					myRRMotion;    // motion vectors (pixels)
	float*					myRRHitDist;   // specular hit distance
	bool					myRRReset;     // RR: discard temporal history (res change / reset / off->on)
	bool					myPrevRR;      // RR toggle state last frame (off->on detect)

	// camera / animation
	float					myOrbitAngle;
	float3					myPrevEye, myPrevU, myPrevV, myPrevW;   // previous camera basis (temporal motion vectors)
	bool					myHavePrev;
	bool					myResetReq;
	int						myPrevBgMode;   // last frame's background mode (switch -> reset accumulation)
	int						myPrevSkyMode;  // last frame's sky mode (gradient/HDRI switch -> reset)
	float					myPrevHdriRot;  // last frame's HDRI rotation (change -> reset)
	float					myPrevProjScale; // last frame's triplanar Projscale (change -> reset accum + RR history so edits show live)
	int						myPrevTexMode;   // last frame's Texmode (change -> reset)
	long long				myPrevGeoCooks;  // geometry POP totalCooks last frame (re-cook = geometry/UV edit -> reset)
	float					myPrevFogSig;    // last frame's fog-params signature (change -> reset accum so edits show live)

	// environment HDRI (input 4) as a CUDA texture object
	cudaTextureObject_t		myHdriTex;
	cudaArray_t				myHdriArray;    // bound array (recreate the tex object only when this changes)

	// base color: LAYERED CUDA texture array (input 5) — N material layers, keyed by triCd.w matID
	cudaTextureObject_t		myBaseColorTex;       // layered texObject
	cudaArray_t				myBaseColorArray;     // last input-5 source array seen (change-detect)
	cudaArray_t				myBaseColorLayered;   // CP5: the allocated layered array (owned)
	int						myBcW, myBcLayerH, myBcNumMat;  // CP5: change-detect dims for the layered build
	void*					myTexStage;       // device staging for the CPU-bounced texPop (Vulkan-origin toptoPOP -> CPU readback -> HtoD; CUDA getBuffer on it deadlocks begin)
	size_t					myTexStageBytes;  // current capacity of myTexStage

	// ENV 3: HDRI importance-sampling 2D CDF (host-built, throttled)
	float*					myEnvCondCdf;   // [EH*EW] per-row conditional CDF (device)
	float*					myEnvMargCdf;   // [EH] marginal CDF (device)
	float*					myEnvFunc;      // [EH*EW] importance func lum*sinθ (device)
	float					myEnvFuncInt;   // sum of the func (pdf normalization)
	int						myEnvW, myEnvH; // CDF grid resolution
	bool					myEnvReady;
	cudaArray_t				myEnvBuiltArray;// the array the current CDF was built from
	int						myEnvLastBuild; // execute count of last build (throttle)
	bool					myEnvRebuildReq;// "Rebuild Env Map" pulse

	// analytic Light COMPs — parsed host-side from the render's "Lightdata" string param into a device buffer
	PTLight*				myPtLights;     // device buffer
	int						myPtLightCap;   // allocated count
	int						myNumPtLights;  // current count

	bool					buildAll();
	bool					buildInputTriGAS(unsigned int N);
	bool					buildEnvCDF(const OP_TOPInput* hdriIn, const OP_CUDAArrayInfo* hdriArr);
};
