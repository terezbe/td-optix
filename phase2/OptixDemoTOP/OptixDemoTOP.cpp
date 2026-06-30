/* Standalone OptiX path-tracing showcase — procedural sphere scene, no external geometry. */
#include "OptixDemoTOP.h"

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <string>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include "cuda_runtime.h"
#include <optix.h>
#include <optix_stubs.h>
#include <optix_function_table_definition.h>
#include <cstdarg>

// ---- freeze diagnostics: breadcrumb log (OutputDebugString -> DebugView + flush-per-line file) ----
// Each line is emitted BEFORE the call it describes and flushed to disk, so after a main-thread
// hang the LAST line names the exact step that wedged. Set OXD_LOG 0 to compile it all out.
#define OXD_LOG 0
#if OXD_LOG
extern "C" __declspec(dllimport) void __stdcall OutputDebugStringA(const char*);
static void oxdLog(const char* fmt, ...){
	char buf[640];
	va_list ap; va_start(ap, fmt); vsnprintf(buf, sizeof(buf)-2, fmt, ap); va_end(ap);
	size_t n=strlen(buf); buf[n]='\n'; buf[n+1]='\0';
	OutputDebugStringA(buf);
	char lpath[1024]; DWORD tl=GetTempPathA(1024,lpath);
	if(tl==0u||tl>1000u){ lpath[0]='.'; lpath[1]='\\'; lpath[2]=0; }
	strcat_s(lpath,sizeof(lpath),"oxd_trace.log");
	FILE* f=fopen(lpath,"a");
	if(f){ fputs(buf,f); fflush(f); fclose(f); }
}
#define OXD_LOGF(...) oxdLog(__VA_ARGS__)
#else
#define OXD_LOGF(...) ((void)0)
#endif

// ---- BREADCRUMB: memory-mapped, MAIN-THREAD deadlock locator --------------------------------
// Two watchdog *threads* (heap-based, then lock-free) each captured only ONE sample (step=0) then
// went silent: a separate thread samples too early and is then caught by the same hang. So the
// MAIN THREAD now records its own progress, as a single mov into a memory-mapped page (a few ns
// -> cannot mask the race). Windows flushes file-backed mapped pages to disk when the process is
// force-killed, so after the freeze oxd_bread.bin holds the last step the main thread reached.
// 4 little-endian int32: [0]=magic 0xB0B, [1]=cook, [2]=last step, [3]=#OXD_STEP updates.
// (CreateFileA / CreateFileMappingA / MapViewOfFile / FlushViewOfFile come from <windows.h>,
//  already pulled in by cuda_runtime.h — do NOT redeclare them; their structptr args clash.)
#define OXD_WATCHDOG 1
#if OXD_WATCHDOG
static volatile long  g_oxdStep = 0;
static volatile long  g_oxdCook = 0;
static volatile long* g_oxdMap  = nullptr;
static void*          g_oxdFileH = nullptr;          // kept so unload can release the mapping (else it leaks + locks the file)
static void*          g_oxdMapH  = nullptr;
static void oxdWatchdogStart(){
	if(g_oxdMap) return;
	// GENERIC_READ|WRITE=0xC0000000, SHARE_READ|WRITE=3, OPEN_ALWAYS=4, ATTR_NORMAL=0x80.
	// OPEN_ALWAYS (NOT CREATE_ALWAYS): never truncate. CREATE_ALWAYS FAILS if a prior leaked mapping
	// still holds this file (a mapped file can't be truncated) -> it silently nulled the breadcrumb.
	char bpath[1024]; DWORD tl=GetTempPathA(1024,bpath);    // %TEMP%\oxd_bread.bin — portable, no install-dir write
	if(tl==0u||tl>1000u){ bpath[0]='.'; bpath[1]='\\'; bpath[2]=0; }
	strcat_s(bpath,sizeof(bpath),"oxd_bread.bin");
	void* h = CreateFileA(bpath,
	                      0xC0000000u, 3u, nullptr, 4u, 0x80u, nullptr);
	if(h==(void*)(long long)-1) return;                 // INVALID_HANDLE_VALUE
	void* m = CreateFileMappingA(h, nullptr, 4u /*PAGE_READWRITE*/, 0u, 4096u, nullptr);
	if(!m){ CloseHandle(h); return; }
	void* v = MapViewOfFile(m, 0xF001Fu /*FILE_MAP_ALL_ACCESS*/, 0u, 0u, 4096u);
	if(!v){ CloseHandle(m); CloseHandle(h); return; }
	g_oxdFileH=h; g_oxdMapH=m; g_oxdMap=(volatile long*)v;
	for(int i=0;i<32;i++) g_oxdMap[i]=0;                 // OPEN_ALWAYS doesn't truncate -> clear stale slots
	g_oxdMap[0]=0xB0B;
	FlushViewOfFile((const void*)g_oxdMap, 128);
}
static void oxdWatchdogStop(){                          // call on plugin unload -> no leak, file unlocked, CREATE works again
	if(g_oxdMap){ FlushViewOfFile((const void*)g_oxdMap,128); UnmapViewOfFile((void*)g_oxdMap); g_oxdMap=nullptr; }
	if(g_oxdMapH){ CloseHandle(g_oxdMapH); g_oxdMapH=nullptr; }
	if(g_oxdFileH){ CloseHandle(g_oxdFileH); g_oxdFileH=nullptr; }
}
#define OXD_STEP(n) do{ g_oxdStep=(long)(n); if(g_oxdMap){ g_oxdMap[2]=(long)(n); g_oxdMap[3]++; } }while(0)
#define OXD_COOK(n) do{ g_oxdCook=(long)(n); if(g_oxdMap){ g_oxdMap[1]=(long)(n); } }while(0)
#define OXD_SET(slot,val) do{ if(g_oxdMap) g_oxdMap[(slot)]=(long)(val); }while(0)   // store a diagnostic value (slots 4..1023)
#else
static void oxdWatchdogStart(){}
static void oxdWatchdogStop(){}
#define OXD_STEP(n) ((void)0)
#define OXD_COOK(n) ((void)0)
#define OXD_SET(slot,val) ((void)0)
#endif

namespace
{
	// Resolve runtime files relative to THIS plugin DLL (portability: no hardcoded user paths).
	// GetModuleHandleEx(FROM_ADDRESS) on a function compiled into this DLL yields the DLL's own
	// HMODULE, so paths are correct wherever the .dll is installed. demo_programs.ptx and
	// nvngx_dlssd.dll must sit in the SAME folder as OptixDemoTOP.dll.
	static std::wstring pluginDirW(){
		HMODULE h=nullptr;                                   // 0x4=FROM_ADDRESS, 0x2=UNCHANGED_REFCOUNT
		if(!GetModuleHandleExW(0x4u|0x2u,(LPCWSTR)&pluginDirW,&h)) return L".";
		wchar_t buf[1024]; DWORD n=GetModuleFileNameW(h,buf,1024);
		if(n==0||n>=1024) return L".";
		std::wstring p(buf,n); size_t pos=p.find_last_of(L"\\/");
		return pos==std::wstring::npos ? L"." : p.substr(0,pos);
	}

	struct EmptyData { int dummy; };
	struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) EmptyRecord { char header[OPTIX_SBT_RECORD_HEADER_SIZE]; EmptyData data; };

	static std::string readFile(const char* p){ std::ifstream f(p,std::ios::binary); if(!f) return std::string(); std::ostringstream ss; ss<<f.rdbuf(); return ss.str(); }
	static std::string readFileW(const std::wstring& p){ std::ifstream f(p.c_str(),std::ios::binary); if(!f) return std::string(); std::ostringstream ss; ss<<f.rdbuf(); return ss.str(); }

	static OptixImage2D mkImg(CUdeviceptr d,int W,int H,OptixPixelFormat fmt=OPTIX_PIXEL_FORMAT_FLOAT4){ OptixImage2D im={}; im.data=d; im.width=(unsigned)W; im.height=(unsigned)H; im.rowStrideInBytes=(unsigned)(W*sizeof(float4)); im.pixelStrideInBytes=(unsigned)sizeof(float4); im.format=fmt; return im; }

	// IEEE half -> float (for downloading 16-bit-float HDRIs to build the importance CDF)
	static float halfToFloat(unsigned short h){
		unsigned int s=(h>>15)&1u, e=(h>>10)&0x1Fu, m=h&0x3FFu, f;
		if(e==0u){ if(m==0u) f=s<<31; else { e=1u; while(!(m&0x400u)){ m<<=1; e--; } m&=0x3FFu; f=(s<<31)|((e-15u+127u)<<23)|(m<<13); } }
		else if(e==0x1Fu) f=(s<<31)|(0xFFu<<23)|(m<<13);
		else f=(s<<31)|((e-15u+127u)<<23)|(m<<13);
		float r; memcpy(&r,&f,4); return r;
	}

	// ---- host vector helpers ----
	static float3 H3(float x,float y,float z){ float3 v; v.x=x; v.y=y; v.z=z; return v; }
	static float3 hsub(float3 a,float3 b){ return H3(a.x-b.x,a.y-b.y,a.z-b.z); }
	static float3 hscl(float3 a,float s){ return H3(a.x*s,a.y*s,a.z*s); }
	static float3 hcross(float3 a,float3 b){ return H3(a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x); }
	static float  hlen(float3 a){ return sqrtf(a.x*a.x+a.y*a.y+a.z*a.z); }
	static float3 hnorm(float3 a){ float l=hlen(a); return l>0?H3(a.x/l,a.y/l,a.z/l):a; }
	static float  hrnd(unsigned int& s){ s=s*747796405u+2891336453u; unsigned int w=((s>>((s>>28u)+4u))^s)*277803737u; return (((w>>22u)^w)&0x00FFFFFFu)/16777216.0f; }

	static DemoMaterial matLambert(float3 a){ DemoMaterial m={}; m.type=0; m.albedo=a; return m; }
	static DemoMaterial matMetal(float3 a,float f){ DemoMaterial m={}; m.type=1; m.albedo=a; m.fuzz=f; return m; }
	static DemoMaterial matGlass(float ior){ DemoMaterial m={}; m.type=2; m.albedo=H3(1,1,1); m.ior=ior; return m; }
	static DemoMaterial matEmit(float3 e){ DemoMaterial m={}; m.type=3; m.emission=e; return m; }

	static void genScene(std::vector<float3>& C, std::vector<float>& R, std::vector<DemoMaterial>& M,
	                     std::vector<float4>& Lp, std::vector<float4>& Le)
	{
		auto add=[&](float3 c,float r,DemoMaterial m){
			C.push_back(c); R.push_back(r); M.push_back(m);
			if(m.type==3){                                  // emissive -> register as an NEE light
				float4 p; p.x=c.x; p.y=c.y; p.z=c.z; p.w=r;            Lp.push_back(p);
				float4 e; e.x=m.emission.x; e.y=m.emission.y; e.z=m.emission.z; e.w=0; Le.push_back(e);
			}
		};
		add(H3(0,-1000,0),1000.0f, matLambert(H3(0.5f,0.5f,0.5f)));      // ground (giant sphere)
		unsigned int s=1234567u;
		for(int a=-11;a<11;a++) for(int b=-11;b<11;b++)
		{
			float choose=hrnd(s);
			float3 c=H3(a+0.9f*hrnd(s), 0.2f, b+0.9f*hrnd(s));
			if(hlen(hsub(c,H3(4,0.2f,0)))<0.9f) continue;
			if(choose<0.62f)      { float3 al=H3(hrnd(s)*hrnd(s),hrnd(s)*hrnd(s),hrnd(s)*hrnd(s)); add(c,0.2f,matLambert(al)); }
			else if(choose<0.80f) { float3 al=H3(0.5f+0.5f*hrnd(s),0.5f+0.5f*hrnd(s),0.5f+0.5f*hrnd(s)); add(c,0.2f,matMetal(al,0.3f*hrnd(s))); }
			else if(choose<0.90f) { add(c,0.2f,matGlass(1.5f)); }
			else                  { float3 e=H3((2.0f+4.0f*hrnd(s))*(0.3f+0.7f*hrnd(s)),(2.0f+4.0f*hrnd(s))*(0.3f+0.7f*hrnd(s)),(2.0f+4.0f*hrnd(s))*(0.3f+0.7f*hrnd(s))); add(c,0.2f,matEmit(e)); }
		}
		add(H3( 0,1,0),1.0f, matGlass(1.5f));                            // hero glass
		add(H3(-4,1,0),1.0f, matLambert(H3(0.4f,0.2f,0.1f)));            // hero diffuse
		add(H3( 4,1,0),1.0f, matMetal(H3(0.7f,0.6f,0.5f),0.0f));         // hero mirror
		add(H3(-2,0.6f, 4),0.6f, matEmit(H3(7.0f,2.2f,1.0f)));           // warm key light
		add(H3( 3,0.6f,-3),0.6f, matEmit(H3(1.0f,2.6f,7.0f)));           // cool fill light
	}
}

extern "C"
{
DLLEXPORT void FillTOPPluginInfo(TOP_PluginInfo* info)
{
	if (!info->setAPIVersion(TOPCPlusPlusAPIVersion)) return;
	info->executeMode = TOP_ExecuteMode::CUDA;
	info->customOPInfo.opType->setString("Optixdemo");
	info->customOPInfo.opLabel->setString("OptiX Demo");
	info->customOPInfo.opIcon->setString("OXD");
	info->customOPInfo.authorName->setString("Erez");
	info->customOPInfo.authorEmail->setString("erezc.media@gmail.com");
	info->customOPInfo.minInputs = 0;
	info->customOPInfo.maxInputs = 7;
}
DLLEXPORT TOP_CPlusPlusBase* CreateTOPInstance(const OP_NodeInfo* info, TOP_Context* c) { return new OptixDemoTOP(info,c); }
DLLEXPORT void DestroyTOPInstance(TOP_CPlusPlusBase* i, TOP_Context* c) { delete (OptixDemoTOP*)i; }
};

OptixDemoTOP::OptixDemoTOP(const OP_NodeInfo* info, TOP_Context* context) :
	myNodeInfo(info), myExecuteCount(0), myStream(0), myContext(context), myError(nullptr)
{
	myOptixContext=0; myOptixTried=false; myOptixResult=-1;
	myReady=false; myReadyResult=-999;
	myModule=0; mySphereIS=0; myRaygenPG=0; myMissPG=0; myHitPG=0; myPipeline=0; mySbt={}; myParamsBuf=0; myLog[0]=0;
	myCenters=0; myRadii=0; myMaterials=0; mySphereCount=0; myGasHandle=0; myGasBuffer=0;
	myLightPos=0; myLightEmit=0; myLightCDF=0; myNumLights=0; myTotalPower=0.0f;
	myTriHitPG=0; myTriVerts=0; myTriVertCap=0; myTriGasBuffer=0; myTriGasTemp=0; myTriGasTempSize=0; myTriGasOutSize=0; myTriGasN=0; myTriGasHandle=0;
	myTriCd=0; myTriMat=0; myTriN=0; myTriUV=0; myTriGasRefits=0;
	myEmitTriIdx=0; myEmitCap=0; myNumEmitTri=0; myTriMatHost=0; myEmitIdxHost=0; myDbgScanTri=-1; myDbgFirstMat=-99.0f;
	myImage=0; myAccum=0; myFogAccum=0; myFogReset=true; myW=0; myH=0; myFrameIndex=0;
	myDenoiser=0; myDenoiserState=0; myDenoiserScratch=0; myIntensity=0; myStateSize=0; myScratchSize=0; myAlbedo=0; myNormal=0; myDenoised=0;
	myFlow=0; myPrevDenoised=0; myHavePrevDenoised=false; myPrevU=H3(0,0,0); myPrevV=H3(0,0,0); myPrevW=H3(0,0,0);
	myTaaCol0=0; myTaaCol1=0; myTaaParity=0;
	myRRInitTried=false;
	myRRColor=0; myRRSpecAlb=0; myRRRough=0; myRRDepth=0; myRRMotion=0; myRRHitDist=0;
	myRRReset=true; myPrevRR=false;
	myOrbitAngle=0.0f; myHavePrev=false; myResetReq=false; myPrevEye=H3(0,0,0);
	myPrevBgMode=0; myPrevSkyMode=0; myPrevHdriRot=0.0f; myPrevProjScale=-1.0f; myPrevTexMode=-1; myPrevGeoCooks=-1; myPrevFogSig=-1.0f;
	myHdriTex=0; myHdriArray=nullptr;
	myBaseColorTex=0; myBaseColorArray=nullptr;
	myBaseColorLayered=nullptr; myBcW=0; myBcLayerH=0; myBcNumMat=0;
	myTexStage=nullptr; myTexStageBytes=0;
	myEnvCondCdf=0; myEnvMargCdf=0; myEnvFunc=0; myEnvFuncInt=0.0f; myEnvW=0; myEnvH=0;
	myEnvReady=false; myEnvBuiltArray=nullptr; myEnvLastBuild=-1000; myEnvRebuildReq=false;
	myPtLights=0; myPtLightCap=0; myNumPtLights=0;
	cudaStreamCreate(&myStream);
}

OptixDemoTOP::~OptixDemoTOP()
{
	oxdWatchdogStop();      // release the breadcrumb mapping so unload doesn't leak/lock oxd_bread.bin
	if (myHdriTex)          cudaDestroyTextureObject(myHdriTex);
	if (myBaseColorTex)     cudaDestroyTextureObject(myBaseColorTex);
	if (myBaseColorLayered) cudaFreeArray(myBaseColorLayered);
	if (myTexStage)         cudaFree(myTexStage);
	if (myPtLights)         cudaFree(myPtLights);
	if (myEnvCondCdf)       cudaFree(myEnvCondCdf);
	if (myEnvMargCdf)       cudaFree(myEnvMargCdf);
	if (myEnvFunc)          cudaFree(myEnvFunc);
	if (myImage)            cudaFree(myImage);
	if (myAccum)            cudaFree((void*)myAccum);
	if (myFogAccum)         cudaFree((void*)myFogAccum);
	if (myAlbedo)           cudaFree(myAlbedo);
	if (myNormal)           cudaFree(myNormal);
	if (myDenoised)         cudaFree(myDenoised);
	if (myFlow)             cudaFree(myFlow);
	if (myPrevDenoised)     cudaFree(myPrevDenoised);
	if (myTaaCol0)          cudaFree(myTaaCol0);
	if (myTaaCol1)          cudaFree(myTaaCol1);
	if (myRRColor)          cudaFree(myRRColor);
	if (myRRSpecAlb)        cudaFree(myRRSpecAlb);
	if (myRRRough)          cudaFree(myRRRough);
	if (myRRDepth)          cudaFree(myRRDepth);
	if (myRRMotion)         cudaFree(myRRMotion);
	if (myRRHitDist)        cudaFree(myRRHitDist);
	if (myDenoiserState)    cudaFree((void*)myDenoiserState);
	if (myDenoiserScratch)  cudaFree((void*)myDenoiserScratch);
	if (myIntensity)        cudaFree((void*)myIntensity);
	if (myDenoiser)         optixDenoiserDestroy(myDenoiser);
	if (myParamsBuf)        cudaFree((void*)myParamsBuf);
	if (mySbt.raygenRecord) cudaFree((void*)mySbt.raygenRecord);
	if (mySbt.missRecordBase) cudaFree((void*)mySbt.missRecordBase);
	if (mySbt.hitgroupRecordBase) cudaFree((void*)mySbt.hitgroupRecordBase);
	if (myGasBuffer)        cudaFree((void*)myGasBuffer);
	if (myCenters)          cudaFree((void*)myCenters);
	if (myRadii)            cudaFree((void*)myRadii);
	if (myMaterials)        cudaFree((void*)myMaterials);
	if (myLightPos)         cudaFree((void*)myLightPos);
	if (myLightEmit)        cudaFree((void*)myLightEmit);
	if (myLightCDF)         cudaFree((void*)myLightCDF);
	if (myTriVerts)         cudaFree((void*)myTriVerts);
	if (myTriCd)            cudaFree((void*)myTriCd);
	if (myTriMat)           cudaFree((void*)myTriMat);
	if (myTriN)             cudaFree((void*)myTriN);
	if (myTriUV)            cudaFree((void*)myTriUV);
	if (myEmitTriIdx)       cudaFree((void*)myEmitTriIdx);
	if (myTriMatHost)       free(myTriMatHost);
	if (myEmitIdxHost)      free(myEmitIdxHost);
	if (myTriGasBuffer)     cudaFree((void*)myTriGasBuffer);
	if (myTriGasTemp)       cudaFree((void*)myTriGasTemp);
	if (myTriHitPG)         optixProgramGroupDestroy(myTriHitPG);
	if (myPipeline)         optixPipelineDestroy(myPipeline);
	if (myRaygenPG)         optixProgramGroupDestroy(myRaygenPG);
	if (myMissPG)           optixProgramGroupDestroy(myMissPG);
	if (myHitPG)            optixProgramGroupDestroy(myHitPG);
	if (mySphereIS)         optixModuleDestroy(mySphereIS);
	if (myModule)           optixModuleDestroy(myModule);
	if (myOptixContext)     optixDeviceContextDestroy(myOptixContext);
	if (myStream)           cudaStreamDestroy(myStream);
}

// Build the HDRI importance-sampling 2D CDF (host side; called throttled). Downloads the equirect env,
// downsamples to EW*EH, weights by luminance*sinθ, builds marginal + per-row conditional CDFs, uploads.
bool OptixDemoTOP::buildEnvCDF(const OP_TOPInput* hdriIn, const OP_CUDAArrayInfo* hdriArr)
{
	if(!hdriIn || !hdriArr || !hdriArr->cudaArray) return false;
	int sw=(int)hdriIn->textureDesc.width, sh=(int)hdriIn->textureDesc.height;
	if(sw<=0 || sh<=0) return false;
	OP_PixelFormat fmt=hdriIn->textureDesc.pixelFormat;
	int bpp = (fmt==OP_PixelFormat::RGBA32Float)?16 : (fmt==OP_PixelFormat::RGBA16Float)?8 : 0;
	if(bpp==0) return false;                                    // need a float HDRI (16F / 32F)

	std::vector<unsigned char> host((size_t)sw*sh*bpp);         // download the full equirect to host
	cudaMemcpy2DFromArrayAsync(host.data(),(size_t)sw*bpp,hdriArr->cudaArray,0,0,(size_t)sw*bpp,sh,cudaMemcpyDeviceToHost,myStream);
	cudaStreamSynchronize(myStream);

	const int EW=256, EH=128; const float PI=3.14159265358979f;
	std::vector<float> func((size_t)EW*EH), cond((size_t)EW*EH), marg(EH), rowSum(EH);
	double total=0.0;
	for(int j=0;j<EH;j++){
		float st=sinf(((float)j+0.5f)/EH*PI);                  // equirect solid-angle weight
		int sj=(int)(((float)j+0.5f)/EH*sh); if(sj>=sh)sj=sh-1;
		double acc=0.0;
		for(int i=0;i<EW;i++){
			int si=(int)(((float)i+0.5f)/EW*sw); if(si>=sw)si=sw-1;
			size_t t=((size_t)sj*sw+si)*4; float r,g,b;
			if(bpp==16){ const float* p=(const float*)host.data()+t; r=p[0]; g=p[1]; b=p[2]; }
			else { const unsigned short* p=(const unsigned short*)host.data()+t; r=halfToFloat(p[0]); g=halfToFloat(p[1]); b=halfToFloat(p[2]); }
			float lum=0.2126f*r+0.7152f*g+0.0722f*b; if(lum<0.0f) lum=0.0f;
			float fv=lum*st; func[(size_t)j*EW+i]=fv;
			acc+=fv; cond[(size_t)j*EW+i]=(float)acc;           // unnormalized cumulative (per row)
		}
		rowSum[j]=(float)acc;
		if(acc>0.0){ for(int i=0;i<EW;i++) cond[(size_t)j*EW+i]/=(float)acc; }     // normalize the row CDF
		else       { for(int i=0;i<EW;i++) cond[(size_t)j*EW+i]=(float)(i+1)/EW; }
		total+=acc;
	}
	if(total<=0.0) return false;
	double macc=0.0;
	for(int j=0;j<EH;j++){ macc+=rowSum[j]; marg[j]=(float)(macc/total); }

	if(!myEnvFunc){ cudaMalloc((void**)&myEnvFunc,(size_t)EW*EH*sizeof(float)); cudaMalloc((void**)&myEnvCondCdf,(size_t)EW*EH*sizeof(float)); cudaMalloc((void**)&myEnvMargCdf,(size_t)EH*sizeof(float)); }
	cudaMemcpy(myEnvFunc,func.data(),(size_t)EW*EH*sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(myEnvCondCdf,cond.data(),(size_t)EW*EH*sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(myEnvMargCdf,marg.data(),(size_t)EH*sizeof(float),cudaMemcpyHostToDevice);
	myEnvW=EW; myEnvH=EH; myEnvFuncInt=(float)total; myEnvReady=true;
	return true;
}

void OptixDemoTOP::getGeneralInfo(TOP_GeneralInfo* g, const OP_Inputs*, void*) { g->cookEveryFrameIfAsked = true; }   // cook every frame WHILE VIEWED (stable); always-cook caused a post-render wedge under continuous reloads

bool
OptixDemoTOP::buildAll()
{
	myLog[0]=0; size_t logSize;

	std::string ptx=readFileW(pluginDirW()+L"\\demo_programs.ptx");
	if (ptx.empty()) { myReadyResult=-100; strcpy_s(myLog,sizeof(myLog),"Failed to read demo_programs.ptx"); return false; }

	OptixModuleCompileOptions mco={}; mco.maxRegisterCount=OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT; mco.optLevel=OPTIX_COMPILE_OPTIMIZATION_DEFAULT; mco.debugLevel=OPTIX_COMPILE_DEBUG_LEVEL_NONE;
	OptixPipelineCompileOptions pco={};
	pco.usesMotionBlur=0; pco.traversableGraphFlags=OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
	pco.numPayloadValues=2; pco.numAttributeValues=1; pco.exceptionFlags=OPTIX_EXCEPTION_FLAG_NONE;
	pco.pipelineLaunchParamsVariableName="params"; pco.usesPrimitiveTypeFlags=OPTIX_PRIMITIVE_TYPE_FLAGS_SPHERE | OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;

	logSize=sizeof(myLog);
	OptixResult r=optixModuleCreate(myOptixContext,&mco,&pco,ptx.c_str(),ptx.size(),myLog,&logSize,&myModule);
	if (r!=OPTIX_SUCCESS){ myReadyResult=(int)r; return false; }

	// built-in hardware sphere intersector module
	{
		OptixBuiltinISOptions bis={}; bis.builtinISModuleType=OPTIX_PRIMITIVE_TYPE_SPHERE; bis.usesMotionBlur=0; bis.buildFlags=OPTIX_BUILD_FLAG_PREFER_FAST_TRACE; bis.curveEndcapFlags=0;
		r=optixBuiltinISModuleGet(myOptixContext,&mco,&pco,&bis,&mySphereIS);
		if (r!=OPTIX_SUCCESS){ myReadyResult=(int)r; strcpy_s(myLog,sizeof(myLog),"sphere IS module failed"); return false; }
	}

	OptixProgramGroupOptions pgo={};
	OptixProgramGroupDesc rg={}; rg.kind=OPTIX_PROGRAM_GROUP_KIND_RAYGEN; rg.raygen.module=myModule; rg.raygen.entryFunctionName="__raygen__rg";
	logSize=sizeof(myLog); r=optixProgramGroupCreate(myOptixContext,&rg,1,&pgo,myLog,&logSize,&myRaygenPG); if(r!=OPTIX_SUCCESS){myReadyResult=(int)r;return false;}
	OptixProgramGroupDesc ms={}; ms.kind=OPTIX_PROGRAM_GROUP_KIND_MISS; ms.miss.module=myModule; ms.miss.entryFunctionName="__miss__ms";
	logSize=sizeof(myLog); r=optixProgramGroupCreate(myOptixContext,&ms,1,&pgo,myLog,&logSize,&myMissPG); if(r!=OPTIX_SUCCESS){myReadyResult=(int)r;return false;}
	OptixProgramGroupDesc hg={}; hg.kind=OPTIX_PROGRAM_GROUP_KIND_HITGROUP; hg.hitgroup.moduleCH=myModule; hg.hitgroup.entryFunctionNameCH="__closesthit__ch"; hg.hitgroup.moduleIS=mySphereIS; hg.hitgroup.entryFunctionNameIS=nullptr;
	logSize=sizeof(myLog); r=optixProgramGroupCreate(myOptixContext,&hg,1,&pgo,myLog,&logSize,&myHitPG); if(r!=OPTIX_SUCCESS){myReadyResult=(int)r;return false;}
	OptixProgramGroupDesc tg={}; tg.kind=OPTIX_PROGRAM_GROUP_KIND_HITGROUP; tg.hitgroup.moduleCH=myModule; tg.hitgroup.entryFunctionNameCH="__closesthit__tri"; tg.hitgroup.moduleIS=nullptr; tg.hitgroup.entryFunctionNameIS=nullptr;
	logSize=sizeof(myLog); r=optixProgramGroupCreate(myOptixContext,&tg,1,&pgo,myLog,&logSize,&myTriHitPG); if(r!=OPTIX_SUCCESS){myReadyResult=(int)r;return false;}

	OptixProgramGroup pgs[4]={myRaygenPG,myMissPG,myHitPG,myTriHitPG};
	OptixPipelineLinkOptions plo={}; plo.maxTraceDepth=1;
	logSize=sizeof(myLog); r=optixPipelineCreate(myOptixContext,&pco,&plo,pgs,4,myLog,&logSize,&myPipeline); if(r!=OPTIX_SUCCESS){myReadyResult=(int)r;return false;}
	optixPipelineSetStackSize(myPipeline,0,0,8192,1);

	EmptyRecord erg,ems,ehg[2]; optixSbtRecordPackHeader(myRaygenPG,&erg); optixSbtRecordPackHeader(myMissPG,&ems); optixSbtRecordPackHeader(myHitPG,&ehg[0]); optixSbtRecordPackHeader(myTriHitPG,&ehg[1]);
	CUdeviceptr d_rg=0,d_ms=0,d_hg=0;
	cudaMalloc((void**)&d_rg,sizeof(EmptyRecord)); cudaMalloc((void**)&d_ms,sizeof(EmptyRecord)); cudaMalloc((void**)&d_hg,2*sizeof(EmptyRecord));
	cudaMemcpy((void*)d_rg,&erg,sizeof(EmptyRecord),cudaMemcpyHostToDevice);
	cudaMemcpy((void*)d_ms,&ems,sizeof(EmptyRecord),cudaMemcpyHostToDevice);
	cudaMemcpy((void*)d_hg,ehg,2*sizeof(EmptyRecord),cudaMemcpyHostToDevice);
	mySbt={}; mySbt.raygenRecord=d_rg;
	mySbt.missRecordBase=d_ms; mySbt.missRecordStrideInBytes=(unsigned)sizeof(EmptyRecord); mySbt.missRecordCount=1;
	mySbt.hitgroupRecordBase=d_hg; mySbt.hitgroupRecordStrideInBytes=(unsigned)sizeof(EmptyRecord); mySbt.hitgroupRecordCount=2;

	// ---- procedural scene -> device ----
	std::vector<float3> C; std::vector<float> R; std::vector<DemoMaterial> Mt; std::vector<float4> Lp, Le;
	genScene(C,R,Mt,Lp,Le);
	mySphereCount=(unsigned)C.size();
	cudaMalloc((void**)&myCenters, C.size()*sizeof(float3));   cudaMemcpy((void*)myCenters, C.data(), C.size()*sizeof(float3), cudaMemcpyHostToDevice);
	cudaMalloc((void**)&myRadii,   R.size()*sizeof(float));    cudaMemcpy((void*)myRadii,   R.data(), R.size()*sizeof(float),  cudaMemcpyHostToDevice);
	cudaMalloc((void**)&myMaterials, Mt.size()*sizeof(DemoMaterial)); cudaMemcpy((void*)myMaterials, Mt.data(), Mt.size()*sizeof(DemoMaterial), cudaMemcpyHostToDevice);

		// ---- NEE light buffers: power-weighted CDF over emissive spheres ----
		myNumLights=(int)Lp.size();
		if(myNumLights>0){
			std::vector<float> cdf((size_t)myNumLights);
			float acc=0.0f;
			for(int i=0;i<myNumLights;i++){
				float lum=0.2126f*Le[i].x + 0.7152f*Le[i].y + 0.0722f*Le[i].z;
				float rr=Lp[i].w; float area=4.0f*3.14159265f*rr*rr;
				acc += lum*area;                                  // total emitted power ~ radiance * area
				cdf[(size_t)i]=acc;
			}
			myTotalPower=acc;
			cudaMalloc((void**)&myLightPos,  Lp.size()*sizeof(float4)); cudaMemcpy((void*)myLightPos, Lp.data(), Lp.size()*sizeof(float4), cudaMemcpyHostToDevice);
			cudaMalloc((void**)&myLightEmit, Le.size()*sizeof(float4)); cudaMemcpy((void*)myLightEmit,Le.data(), Le.size()*sizeof(float4), cudaMemcpyHostToDevice);
			cudaMalloc((void**)&myLightCDF,  cdf.size()*sizeof(float)); cudaMemcpy((void*)myLightCDF, cdf.data(),cdf.size()*sizeof(float),  cudaMemcpyHostToDevice);
		}

	// ---- sphere GAS ----
	OptixBuildInput bi={};
	bi.type=OPTIX_BUILD_INPUT_TYPE_SPHERES;
	bi.sphereArray.vertexBuffers=&myCenters;
	bi.sphereArray.vertexStrideInBytes=sizeof(float3);
	bi.sphereArray.numVertices=mySphereCount;
	bi.sphereArray.radiusBuffers=&myRadii;
	bi.sphereArray.radiusStrideInBytes=sizeof(float);
	bi.sphereArray.singleRadius=0;
	static unsigned int sflags[1]={OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT};
	bi.sphereArray.flags=sflags; bi.sphereArray.numSbtRecords=1;
	OptixAccelBuildOptions ao={}; ao.buildFlags=OPTIX_BUILD_FLAG_PREFER_FAST_TRACE; ao.operation=OPTIX_BUILD_OPERATION_BUILD;
	OptixAccelBufferSizes sizes; r=optixAccelComputeMemoryUsage(myOptixContext,&ao,&bi,1,&sizes); if(r!=OPTIX_SUCCESS){myReadyResult=(int)r;return false;}
	CUdeviceptr tmp=0; cudaMalloc((void**)&tmp,sizes.tempSizeInBytes); cudaMalloc((void**)&myGasBuffer,sizes.outputSizeInBytes);
	r=optixAccelBuild(myOptixContext,myStream,&ao,&bi,1,tmp,sizes.tempSizeInBytes,myGasBuffer,sizes.outputSizeInBytes,&myGasHandle,nullptr,0);
	cudaStreamSynchronize(myStream); cudaFree((void*)tmp);
	if(r!=OPTIX_SUCCESS){myReadyResult=(int)r;return false;}

	cudaMalloc((void**)&myParamsBuf,sizeof(LaunchParamsDemo));

	{   // OptiX AI denoiser (HDR model, albedo + normal guides)
		OptixDenoiserOptions dopt={}; dopt.guideAlbedo=1; dopt.guideNormal=1; dopt.denoiseAlpha=OPTIX_DENOISER_ALPHA_MODE_COPY;
		r=optixDenoiserCreate(myOptixContext, OPTIX_DENOISER_MODEL_KIND_TEMPORAL, &dopt, &myDenoiser);
		if(r!=OPTIX_SUCCESS){ myReadyResult=(int)r; strcpy_s(myLog,sizeof(myLog),"denoiser create failed"); return false; }
	}
	myReadyResult=0; myReady=true;
	return true;
}

bool
OptixDemoTOP::buildInputTriGAS(unsigned int N)
{
	if(N<3 || !myTriVerts) return false;
	N -= (N%3u);                                       // whole triangles only
	OptixBuildInput bi={};
	bi.type=OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
	bi.triangleArray.vertexFormat=OPTIX_VERTEX_FORMAT_FLOAT3;
	bi.triangleArray.vertexStrideInBytes=sizeof(float4);    // RGBA32F texture: read xyz, stride 16
	bi.triangleArray.numVertices=N;
	CUdeviceptr vb=(CUdeviceptr)myTriVerts;
	bi.triangleArray.vertexBuffers=&vb;
	bi.triangleArray.indexFormat=OPTIX_INDICES_FORMAT_NONE; // triangle soup: 3 consecutive verts = 1 triangle
	bi.triangleArray.numIndexTriplets=0; bi.triangleArray.indexBuffer=0;
	static unsigned int tf[1]={OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT};
	bi.triangleArray.flags=tf; bi.triangleArray.numSbtRecords=1;
	// FAST_TRACE BVH; full BUILD on count change, cheap REFIT (UPDATE) for same-count deformation; periodic rebuild keeps quality
	bool canRefit = (myTriGasHandle!=0 && N==myTriGasN && myTriGasRefits<30);
	OptixAccelBuildOptions ao={};
	ao.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE | OPTIX_BUILD_FLAG_ALLOW_UPDATE;
	ao.operation  = canRefit ? OPTIX_BUILD_OPERATION_UPDATE : OPTIX_BUILD_OPERATION_BUILD;
	OptixAccelBufferSizes sizes;
	if(optixAccelComputeMemoryUsage(myOptixContext,&ao,&bi,1,&sizes)!=OPTIX_SUCCESS) return false;
	size_t tempNeed = canRefit ? sizes.tempUpdateSizeInBytes : sizes.tempSizeInBytes;
	if(tempNeed>myTriGasTempSize){ if(myTriGasTemp)cudaFree((void*)myTriGasTemp); cudaMalloc((void**)&myTriGasTemp,tempNeed); myTriGasTempSize=tempNeed; }
	if(!canRefit && sizes.outputSizeInBytes>myTriGasOutSize){ if(myTriGasBuffer)cudaFree((void*)myTriGasBuffer); cudaMalloc((void**)&myTriGasBuffer,sizes.outputSizeInBytes); myTriGasOutSize=sizes.outputSizeInBytes; }
	OptixResult r=optixAccelBuild(myOptixContext,myStream,&ao,&bi,1,myTriGasTemp,myTriGasTempSize,myTriGasBuffer,myTriGasOutSize,&myTriGasHandle,nullptr,0);
	myTriGasN=N; myTriGasRefits = canRefit ? myTriGasRefits+1 : 0;
	return r==OPTIX_SUCCESS;
}

void
OptixDemoTOP::execute(TOP_Output* output, const OP_Inputs* inputs, void*)
{
	myError=nullptr; myExecuteCount++;
	oxdWatchdogStart(); OXD_COOK((unsigned)myExecuteCount); OXD_STEP(0);
	OXD_LOGF("--- execute #%d entry ---", myExecuteCount);

	OP_TextureDesc sug; output->getSuggestedOutputDesc(&sug,nullptr);
	int W=(int)sug.width, H=(int)sug.height; if(W<16)W=16; if(H<16)H=16;

	// read params BEFORE beginCUDAOperations
	int    spp =(int)(inputs->getParDouble("Spp")+0.5);
	int    maxd=(int)(inputs->getParDouble("Maxdepth")+0.5);
	double aperture=inputs->getParDouble("Aperture");
	bool   orbit=inputs->getParDouble("Orbit")>0.5;
	double orbitSpeed=inputs->getParDouble("Orbitspeed");
	double dist=inputs->getParDouble("Distance");
	const char* dn=inputs->getParString("Denoiser"); bool denoise=(dn&&!strcmp(dn,"Optix")), taa=(dn&&!strcmp(dn,"Taa")), rr=(dn&&!strcmp(dn,"Rr"));
	inputs->enablePar("Denoisestr", denoise); inputs->enablePar("Maxhist", taa); inputs->enablePar("Flowscale", rr); inputs->enablePar("Flowinvy", rr); inputs->enablePar("Rrspecmv", rr); inputs->enablePar("Jitter", !rr);  // gray out irrelevant denoiser sliders
	double denoisestr=inputs->getParDouble("Denoisestr"); if(denoisestr<0)denoisestr=0; if(denoisestr>1)denoisestr=1;
	double flowscale=inputs->getParDouble("Flowscale");
	double rrspecmv=inputs->getParDouble("Rrspecmv"); if(rrspecmv<0)rrspecmv=0; if(rrspecmv>1)rrspecmv=1;
	bool   boilDecorrelate=inputs->getParDouble("Boildecorrelate")>0.5;   // free-running RNG seed -> decorrelate noise under motion (off = exact prior)
	double jitter=inputs->getParDouble("Jitter"); if(jitter<0)jitter=0; if(jitter>1)jitter=1;
	/* taa from Denoiser menu */
	double maxhist=inputs->getParDouble("Maxhist"); if(maxhist<1)maxhist=1;
	/* rr from Denoiser menu */
	double firefly=inputs->getParDouble("Firefly"); if(firefly<0)firefly=0;
	bool   flowinvy=inputs->getParDouble("Flowinvy")>0.5;
		bool   nee=inputs->getParDouble("Nee")>0.5;
		bool   useInput=inputs->getParDouble("Useinput")>0.5;
		int    numverts=(int)(inputs->getParDouble("Numverts")+0.5);
		int    nummaterialsP=(int)(inputs->getParDouble("Nummaterials")+0.5); if(nummaterialsP<1) nummaterialsP=1;
		bool   usecamera=inputs->getParDouble("Usecamera")>0.5;
		double eyex=inputs->getParDouble("Eye",0), eyey=inputs->getParDouble("Eye",1), eyez=inputs->getParDouble("Eye",2);
		double fwdx=inputs->getParDouble("Forward",0), fwdy=inputs->getParDouble("Forward",1), fwdz=inputs->getParDouble("Forward",2);
		double camfov=inputs->getParDouble("Camfov");
		double skyzr=inputs->getParDouble("Skyzenith",0), skyzg=inputs->getParDouble("Skyzenith",1), skyzb=inputs->getParDouble("Skyzenith",2);
		double skyhr=inputs->getParDouble("Skyhorizon",0), skyhg=inputs->getParDouble("Skyhorizon",1), skyhb=inputs->getParDouble("Skyhorizon",2);
		double skystr=inputs->getParDouble("Skystrength");
		double sundx=inputs->getParDouble("Sundir",0), sundy=inputs->getParDouble("Sundir",1), sundz=inputs->getParDouble("Sundir",2);
		double suncr=inputs->getParDouble("Suncolor",0), suncg=inputs->getParDouble("Suncolor",1), suncb=inputs->getParDouble("Suncolor",2);
		double sunstr=inputs->getParDouble("Sunstrength");
		double sunang=inputs->getParDouble("Sunangle");
		const char* bgm=inputs->getParString("Backgroundmode");
		int    bgMode=(bgm&&!strcmp(bgm,"Solid"))?1:((bgm&&!strcmp(bgm,"Transparent"))?2:0);
		double bgcr=inputs->getParDouble("Bgcolor",0), bgcg=inputs->getParDouble("Bgcolor",1), bgcb=inputs->getParDouble("Bgcolor",2);
		const char* skym=inputs->getParString("Skymode");
		bool   skyHdri=(skym && !strcmp(skym,"Hdri"));
		bool   skyPhysical=(skym && !strcmp(skym,"Physical"));
		double hdrirot=inputs->getParDouble("Hdrirot");
		bool   envImportanceOn=inputs->getParDouble("Envimportance")>0.5;
		double turbidity=inputs->getParDouble("Turbidity");
		const char* lightdata=inputs->getParString("Lightdata");   // serialized Light COMPs (written by an Execute DAT)
		double lightintensity=inputs->getParDouble("Lightintensity");
		bool   showuv=inputs->getParDouble("Showuv")>0.5;
		double projscale=inputs->getParDouble("Projscale");
		const char* texm=inputs->getParString("Texmode");
		int    texmode=(texm&&!strcmp(texm,"Uv"))?1:((texm&&!strcmp(texm,"Projection"))?2:0);
		bool   fogEnable=inputs->getParDouble("Fogenable")>0.5;
		double fogDensity=inputs->getParDouble("Fogdensity");
		double fogcr=inputs->getParDouble("Fogcolor",0), fogcg=inputs->getParDouble("Fogcolor",1), fogcb=inputs->getParDouble("Fogcolor",2);
		double fogAniso=inputs->getParDouble("Foganisotropy");
		bool   fogEmitNEE=inputs->getParDouble("Fogemitnee")>0.5;
		double fogSkyStr=inputs->getParDouble("Fogskystr");
		bool   fogSingle=inputs->getParDouble("Fogsinglescatter")>0.5;
		int    fogMaxScat=(int)(inputs->getParDouble("Fogmaxscatter")+0.5);
		double fogRRStart=inputs->getParDouble("Fogrrstart");
		double fogStability=inputs->getParDouble("Fogstability");   // 0 = off (exact original), ->1 = motion-robust fog smoothing
		double fogFireflyMax=inputs->getParDouble("Fogfireflyclamp"); if(fogFireflyMax<0)fogFireflyMax=0;   // clamp fog in-scatter spikes (0 = off, exact original)
		// --- God-ray Crispiness (RESEARCH §A) ---
		double crispiness = inputs->getParDouble("Crispiness");
		bool   crispDrives= inputs->getParDouble("Crispdrives")>0.5;
		double fogShaftG  = inputs->getParDouble("Fogshaftg");      // -2 sentinel = "follow Fog Anisotropy (g)"
		double fogContrast= inputs->getParDouble("Fogcontrast");
		if(fogShaftG < -1.0) fogShaftG = fogAniso;                  // sentinel -> bulk fogG -> byte-exact original shafts
		if(crispDrives){                                            // macro owns 5 internal knobs (§A.3 hybrid); fogG/density/color stay manual
			double c = (crispiness<0.0)?0.0:((crispiness>1.0)?1.0:crispiness);
			fogShaftG  = 0.40 + (0.90 - 0.40)*c;                   // milky -> tight forward halo (hard cap 0.90)
			sunang     = 3.0  + (0.30 - 3.0 )*c;                   // 3deg penumbra -> ~solar disk (floor 0.30 deg)
			fogMaxScat = (int)(8.0 + (1.0 - 8.0)*c + 0.5);         // full glow -> single-scatter
			fogSkyStr  = 0.80 + (0.15 - 0.80)*c;                   // strong ambient fill -> minimal (floor 0.15, never 0)
			fogContrast= 1.0  + (1.7  - 1.0 )*c;                   // contrast curve on the in-scatter term
		}

	// Acquire the INPUT CUDA arrays FIRST (handles only; their .cudaArray pointers populate at
	// beginCUDAOperations). The OUTPUT array is created LATER, INSIDE the begin/end block (see the
	// resize-deadlock note below) — NOT here, because createCUDAArray on a resolution change must
	// reallocate the Vulkan output image and that interop has to run inside the managed block.
	// input 0 = triangle-soup positions texture (acquire the CUDA handle BEFORE beginCUDAOperations)
	const OP_TOPInput* posIn = (useInput && inputs->getNumInputs()>0) ? inputs->getInputTOP(0) : nullptr;
	const OP_TOPInput* cdIn  = (useInput && inputs->getNumInputs()>1) ? inputs->getInputTOP(1) : nullptr;   // Cd (color)
	const OP_TOPInput* matIn = (useInput && inputs->getNumInputs()>2) ? inputs->getInputTOP(2) : nullptr;   // Mat (type,rough,ior,emit)
	const OP_TOPInput* nIn   = (useInput && inputs->getNumInputs()>3) ? inputs->getInputTOP(3) : nullptr;   // N (smooth normals)
	const OP_CUDAArrayInfo* inArr=nullptr; const OP_CUDAArrayInfo* cdArr=nullptr; const OP_CUDAArrayInfo* matArr=nullptr; const OP_CUDAArrayInfo* nArr=nullptr;
	unsigned int inW=0,inH=0;
	OXD_LOGF(" acquire: numIn=%d useInput=%d skyHdri=%d conn[0..6]=%d%d%d%d%d%d%d", inputs->getNumInputs(), useInput?1:0, skyHdri?1:0, inputs->getInputTOP(0)?1:0, inputs->getInputTOP(1)?1:0, inputs->getInputTOP(2)?1:0, inputs->getInputTOP(3)?1:0, inputs->getInputTOP(4)?1:0, inputs->getInputTOP(5)?1:0, inputs->getInputTOP(6)?1:0);
	OXD_LOGF("   -> in0 pos  acquire=%d", posIn?1:0);
	if(posIn){ inW=posIn->textureDesc.width; inH=posIn->textureDesc.height; OP_CUDAAcquireInfo acq; acq.stream=myStream; OXD_STEP(10); inArr=posIn->getCUDAArray(acq,nullptr); OXD_STEP(11); }
	OXD_LOGF("   -> in1 Cd   acquire=%d", cdIn?1:0);
	if(cdIn){ OP_CUDAAcquireInfo acq; acq.stream=myStream; OXD_STEP(20); cdArr=cdIn->getCUDAArray(acq,nullptr); OXD_STEP(21); }
	OXD_LOGF("   -> in2 Mat  acquire=%d", matIn?1:0);
	if(matIn){ OP_CUDAAcquireInfo acq; acq.stream=myStream; OXD_STEP(30); matArr=matIn->getCUDAArray(acq,nullptr); OXD_STEP(31); }
	OXD_LOGF("   -> in3 N    acquire=%d", nIn?1:0);
	if(nIn){ OP_CUDAAcquireInfo acq; acq.stream=myStream; OXD_STEP(40); nArr=nIn->getCUDAArray(acq,nullptr); OXD_STEP(41); }
	const OP_TOPInput* bcIn = nullptr;   // input 5 (Movie File In) RETIRED -> base color now arrives via the Texturepop POP (deadlock-safe); never acquire a Vulkan TOP here
	const OP_CUDAArrayInfo* bcArr=nullptr;
	int bcW=0, bcH=0; if(bcIn){ bcW=(int)bcIn->textureDesc.width; bcH=(int)bcIn->textureDesc.height; }   // input-5 dims (read before beginCUDAOperations)
	OXD_LOGF("   -> in5 baseColor conn=%d acquire=%d  <== PRIME SUSPECT", inputs->getInputTOP(5)?1:0, bcIn?1:0);
	if(bcIn){ OP_CUDAAcquireInfo acq; acq.stream=myStream; OXD_STEP(50); bcArr=bcIn->getCUDAArray(acq,nullptr); OXD_STEP(51); }
	OXD_LOGF("   <- in5 baseColor returned (info=%s)", bcArr?"non-null":"null");
	const OP_TOPInput* uvIn = nullptr;   // UV TOP input RETIRED -> UV (Tex) is read POP-direct from Geopop (a 5th Vulkan-backed TOP input deadlocks beginCUDAOperations)
	const OP_CUDAArrayInfo* uvArr=nullptr; int uvW=0,uvH=0; (void)uvIn; (void)uvArr; (void)uvW; (void)uvH;
	// input 4 = equirectangular HDRI (environment); acquire its CUDA handle BEFORE beginCUDAOperations
	const OP_TOPInput* hdriIn = (skyHdri && inputs->getNumInputs()>4) ? inputs->getInputTOP(4) : nullptr;
	const OP_CUDAArrayInfo* hdriArr=nullptr;
	OXD_LOGF("   -> in4 HDRI conn=%d skyHdri=%d acquire=%d", inputs->getInputTOP(4)?1:0, skyHdri?1:0, hdriIn?1:0);
	if(hdriIn){ OP_CUDAAcquireInfo acq; acq.stream=myStream; OXD_STEP(70); hdriArr=hdriIn->getCUDAArray(acq,nullptr); OXD_STEP(71); }

	// --- STAGE 0 de-risk: read a referenced POP's Cd buffer as CUDA, BEFORE begin (texture-as-POP). ---
	// A POP is CUDA-resident (any Vulkan->POP conversion happened in the TOP-to-POP's OWN cook), so this
	// should NOT make begin wait on a Vulkan producer the way a Vulkan TOP input did. The breadcrumb tells
	// us: reach step 76 = getBuffer ok; reach step 81/99 = begin NOT wedged => the approach is sound.
	const OP_POPInput* texPop = inputs->getParPOP("Texturepop");   // texture source (getInputPOP is POP-only). NOTE: this is a toptoPOP of a Vulkan layout TOP -> Vulkan-origin, so CUDA getBuffer DEADLOCKS begin; read it CPU-bounced.
	OP_SmartRef<POP_Buffer> popCdBuf;            // hold the ref in scope; CPU-bounced so getData() is a HOST pointer (valid pre-begin while the ref lives)
	void* popDev=nullptr; long popTotalPts=0;    // device staging (myTexStage) + point count, set after begin
	void* popHost=nullptr;                       // host pointer from the CPU readback; uploaded to myTexStage (HtoD) after begin
	// geometry UV: read the "Tex" (float3 UVW) attribute POP-direct from the Geopop, SAME deadlock-safe pattern
	// as the texture (a 5th Vulkan-backed TOP input deadlocks beginCUDAOperations, so UV must NOT be a TOP input).
	const OP_POPInput* geoPop = inputs->getParPOP("Geopop");
	OP_SmartRef<POP_Buffer> geoTexBuf; void* geoTexDev=nullptr; long geoTexPts=0;
	if(geoPop){ const POP_Attribute* ta=geoPop->getAttribute(POP_AttributeClass::Point,"Tex",nullptr);
		if(!ta) ta=geoPop->getAttribute(POP_AttributeClass::Vertex,"Tex",nullptr);   // Vertex projection methods (cylin/face/equirect) write a VERTEX-class Tex; soup -> vertex order == point order
		if(ta){ POP_GetBufferInfo gi; gi.location=POP_BufferLocation::CUDA; gi.stream=myStream; geoTexBuf=ta->getBuffer(gi,nullptr); } }
	{
		int popAttrFound=0, popPts=0; unsigned long long popBufBytes=0;
		if(texPop){
			const POP_Attribute* a = texPop->getAttribute(POP_AttributeClass::Point, "Color", nullptr);   // toptoPOP (rgba=Color RGBA) names it "Color" = float4
			if(!a) a = texPop->getAttribute(POP_AttributeClass::Point, "Cd", nullptr);                     // fall back to the Houdini/TD "Cd" convention
			if(a){
				popAttrFound=1;
				POP_GetBufferInfo gi; gi.location=POP_BufferLocation::CPU;   // CPU readback, NOT CUDA: getBuffer(CUDA) on this Vulkan-origin toptoPOP wedges the main thread inside the Vulkan->CUDA handoff (breadcrumb step-75 freeze). CPU is a plain Vulkan->host download (no interop semaphore).
				OXD_STEP(75);
				popCdBuf = a->getBuffer(gi, nullptr);
				popHost = popCdBuf ? popCdBuf->getData(nullptr) : nullptr;   // stalls for the GPU->CPU download here (pre-begin, safe); host ptr stays valid while popCdBuf is in scope
				OXD_STEP(76);
				if(popCdBuf){ popBufBytes=(unsigned long long)popCdBuf->info.size; popTotalPts=(long)(popCdBuf->info.size/sizeof(float4)); }
			}
			POP_MaxInfo mi; texPop->getMaxInfo(&mi,nullptr); popPts=(int)mi.points;
		}
		OXD_SET(4, texPop?1:0); OXD_SET(5, popAttrFound); OXD_SET(6, popPts); OXD_SET(7, (long)popBufBytes);
		OXD_LOGF("   -> POP texPop=%d attrFound=%d pts=%d bufBytes=%llu", texPop?1:0, popAttrFound, popPts, popBufBytes);
	}

	// OUTPUT array is created BEFORE beginCUDAOperations. REQUIRED for TD to bind it as this TOP's output
	// texture: creating it AFTER begin renders correctly at the initial size but goes BLACK on a resolution
	// change (the new-size array is never bound as the output).
	// KNOWN LIMITATION — output RESIZE deadlock: when the output dims change, createCUDAArray must DESTROY +
	// REALLOCATE the output Vulkan image, and that Vulkan<->CUDA realloc deadlocks the main thread mid-cook
	// (observed: freeze at breadcrumb step 5 on a width change). NEITHER ordering avoids it: before begin ->
	// step-5 deadlock; after begin -> black. Draining our stream does NOT help (the deadlock is internal to
	// TD's realloc, not our pending work — verified). So the resize is avoided OUTSIDE the C++: the PT_Render
	// wrapper's "Update Resolution" button pauses the COMP (allowCooking=False) while it applies the new size,
	// so TD reallocates the output with no live cook, then resumes. Do NOT change this TOP's resolution while
	// it is cooking. (The end-of-cook stream drain below keeps each cook self-contained, complementing that pause.)
	TOP_CUDAOutputInfo info; info.textureDesc.width=(uint32_t)W; info.textureDesc.height=(uint32_t)H;
	info.textureDesc.texDim=OP_TexDim::e2D; info.textureDesc.pixelFormat=OP_PixelFormat::RGBA32Float; info.stream=myStream;
	OXD_LOGF(" -> createCUDAArray(out) %dx%d", W, H);
	OXD_STEP(5);
	const OP_CUDAArrayInfo* outInfo=output->createCUDAArray(info,nullptr);
	OXD_STEP(6);
	if(!outInfo) return;
	OXD_LOGF(" acquire done; -> beginCUDAOperations");
	OXD_STEP(80);
	if(!myContext->beginCUDAOperations(nullptr)){ OXD_LOGF(" beginCUDAOperations FALSE (early out)"); return; }
	OXD_STEP(81);
	OXD_LOGF(" beginCUDAOperations ok");
	// STAGE 0: the POP CUDA device pointer is filled in during begin (like a TOP cudaArray). Read + record it.
	if(popHost && popTotalPts>0){   // upload the CPU-bounced texels into device staging (myTexStage); popDev then feeds the layered build unchanged
		size_t need=(size_t)popTotalPts*sizeof(float4);
		if(need>myTexStageBytes){ if(myTexStage) cudaFree(myTexStage); if(cudaMalloc(&myTexStage,need)!=cudaSuccess){ myTexStage=nullptr; myTexStageBytes=0; } else myTexStageBytes=need; }
		if(myTexStage){ cudaMemcpy(myTexStage, popHost, need, cudaMemcpyHostToDevice); popDev=myTexStage; }   // sync HtoD orders before the band copies below
		OXD_SET(8, popDev?1:0);
	}
	if(geoTexBuf){ geoTexDev=geoTexBuf->getData(nullptr); geoTexPts=(long)(geoTexBuf->info.size/12); }   // Tex = float3 (12 bytes/point)

	if(!myOptixTried){ myOptixTried=true; OptixResult r=optixInit(); if(r==OPTIX_SUCCESS){ OptixDeviceContextOptions o={}; r=optixDeviceContextCreate(0,&o,&myOptixContext); } myOptixResult=(int)r; }
	if(myOptixContext && !myReady && myReadyResult==-999) buildAll();

	if(myReady)
	{
		// Phase 2: init NGX Ray Reconstruction once, on TD's CUDA context (we are inside beginCUDAOperations).
		if(!myRRInitTried){ myRRInitTried=true; myRR.init(pluginDirW().c_str()); }

		if(!myImage || myW!=W || myH!=H){
			if(myImage)cudaFree(myImage); if(myAccum)cudaFree((void*)myAccum);
			if(myAlbedo)cudaFree(myAlbedo); if(myNormal)cudaFree(myNormal); if(myDenoised)cudaFree(myDenoised);
			if(myFlow)cudaFree(myFlow); if(myPrevDenoised)cudaFree(myPrevDenoised);
			if(myFogAccum)cudaFree((void*)myFogAccum);
			size_t n=(size_t)W*H*sizeof(float4);
			cudaMalloc((void**)&myImage,n); cudaMalloc((void**)&myAccum,n);
			cudaMalloc((void**)&myAlbedo,n); cudaMalloc((void**)&myNormal,n); cudaMalloc((void**)&myDenoised,n);
			cudaMalloc((void**)&myFlow,n); cudaMalloc((void**)&myPrevDenoised,n);
			cudaMalloc((void**)&myFogAccum,n); cudaMemset((void*)myFogAccum,0,n);
			if(myTaaCol0)cudaFree(myTaaCol0); if(myTaaCol1)cudaFree(myTaaCol1);
			cudaMalloc((void**)&myTaaCol0,n); cudaMalloc((void**)&myTaaCol1,n);
			cudaMemset(myTaaCol0,0,n); cudaMemset(myTaaCol1,0,n);
			if(myRRColor)cudaFree(myRRColor); if(myRRSpecAlb)cudaFree(myRRSpecAlb); if(myRRRough)cudaFree(myRRRough);
			if(myRRDepth)cudaFree(myRRDepth); if(myRRMotion)cudaFree(myRRMotion); if(myRRHitDist)cudaFree(myRRHitDist);
			cudaMalloc((void**)&myRRColor,n); cudaMalloc((void**)&myRRSpecAlb,n);
			cudaMalloc((void**)&myRRRough,(size_t)W*H*sizeof(float)); cudaMalloc((void**)&myRRDepth,(size_t)W*H*sizeof(float));
			cudaMalloc((void**)&myRRMotion,(size_t)W*H*sizeof(float2)); cudaMalloc((void**)&myRRHitDist,(size_t)W*H*sizeof(float));
			cudaMemset(myRRHitDist,0,(size_t)W*H*sizeof(float));
			myW=W; myH=H; myFrameIndex=0; myHavePrevDenoised=false; myTaaParity=0; myRRReset=true; myFogReset=true;
			OptixDenoiserSizes ds={}; optixDenoiserComputeMemoryResources(myDenoiser,(unsigned)W,(unsigned)H,&ds);
			if(myDenoiserState)cudaFree((void*)myDenoiserState); if(myDenoiserScratch)cudaFree((void*)myDenoiserScratch);
			cudaMalloc((void**)&myDenoiserState,ds.stateSizeInBytes); myStateSize=ds.stateSizeInBytes;
			cudaMalloc((void**)&myDenoiserScratch,ds.withoutOverlapScratchSizeInBytes); myScratchSize=ds.withoutOverlapScratchSizeInBytes;
			if(!myIntensity) cudaMalloc((void**)&myIntensity,sizeof(float));
			optixDenoiserSetup(myDenoiser,myStream,(unsigned)W,(unsigned)H,myDenoiserState,myStateSize,myDenoiserScratch,myScratchSize);
		}

		// ---- live input geometry: copy soup verts -> device, (re)build the triangle GAS this cook ----
		bool triMode=false, triHaveCd=false, triHaveMat=false, triHaveN=false, triHaveUV=false;
		if(useInput && inArr && inW>0 && inH>0){
			unsigned int padded=inW*inH;
			if((int)padded != myTriVertCap){
				if(myTriVerts)cudaFree((void*)myTriVerts); if(myTriCd)cudaFree((void*)myTriCd); if(myTriMat)cudaFree((void*)myTriMat); if(myTriN)cudaFree((void*)myTriN); if(myTriUV)cudaFree((void*)myTriUV);
				cudaMalloc((void**)&myTriVerts,(size_t)padded*sizeof(float4));
				cudaMalloc((void**)&myTriCd,(size_t)padded*sizeof(float4));
				cudaMalloc((void**)&myTriMat,(size_t)padded*sizeof(float4));
				cudaMalloc((void**)&myTriN,(size_t)padded*sizeof(float4));
				cudaMalloc((void**)&myTriUV,(size_t)padded*sizeof(float4));
				myTriVertCap=(int)padded;
			}
			size_t row=(size_t)inW*sizeof(float4);
			cudaMemcpy2DFromArrayAsync((void*)myTriVerts,row,inArr->cudaArray,0,0,row,inH,cudaMemcpyDeviceToDevice,myStream);
			if(cdArr && cdArr->cudaArray){ cudaMemcpy2DFromArrayAsync((void*)myTriCd, row,cdArr->cudaArray, 0,0,row,inH,cudaMemcpyDeviceToDevice,myStream); triHaveCd=true; }
			if(matArr && matArr->cudaArray){ cudaMemcpy2DFromArrayAsync((void*)myTriMat,row,matArr->cudaArray,0,0,row,inH,cudaMemcpyDeviceToDevice,myStream); triHaveMat=true; }
			if(nArr && nArr->cudaArray){ cudaMemcpy2DFromArrayAsync((void*)myTriN,  row,nArr->cudaArray,  0,0,row,inH,cudaMemcpyDeviceToDevice,myStream); triHaveN=true; }
			if(geoTexDev && myTriUV && geoTexPts>0){   // UV: POP-direct "Tex" (float3) -> myTriUV (float4 xyz=UVW); strided D2D copy
				long n=geoTexPts; if(n>(long)padded) n=(long)padded;
				cudaMemcpy2DAsync((void*)myTriUV, sizeof(float4), geoTexDev, 12, 12, (size_t)n, cudaMemcpyDeviceToDevice, myStream); triHaveUV=true; }
			unsigned int N=(numverts>0 && (unsigned)numverts<=padded) ? (unsigned)numverts : padded;
			if(buildInputTriGAS(N)) triMode=true;

			// ---- emitter glow: build the emissive-triangle index list (v1: gated host readback of triMat) ----
			// Only when the feature is on AND there's material data. One D2H + sync per cook (gated); the
			// pinned/one-frame-latency optimization is deferred to the perf pass. Device computes area/emission.
			myNumEmitTri=0;
			if(triMode && triHaveMat && fogEmitNEE){
				if((int)padded > myEmitCap){
					if(myTriMatHost) free(myTriMatHost);
					if(myEmitIdxHost) free(myEmitIdxHost);
					if(myEmitTriIdx) cudaFree((void*)myEmitTriIdx);
					myTriMatHost=(float4*)malloc((size_t)padded*sizeof(float4));
					myEmitIdxHost=(int*)malloc((size_t)padded*sizeof(int));      // <= padded/3 tris; over-alloc
					cudaMalloc((void**)&myEmitTriIdx,(size_t)padded*sizeof(int));
					myEmitCap=(int)padded;
				}
				if(myTriMatHost && myEmitIdxHost && myEmitTriIdx){
					cudaMemcpyAsync(myTriMatHost, myTriMat, (size_t)padded*sizeof(float4), cudaMemcpyDeviceToHost, myStream);
					cudaStreamSynchronize(myStream);                             // ensure triMat readback complete
					int numTri=(int)(N/3u), cnt=0;
					myDbgScanTri=numTri; myDbgFirstMat=(numTri>0)?myTriMatHost[0].x:-98.0f;   // diag
					for(int t=0;t<numTri;t++){                                   // emissive iff (int)(triMat[3t].x+0.5)%10==3
						int rawtype=(int)(myTriMatHost[3*t].x+0.5f);
						if(rawtype%10==3) myEmitIdxHost[cnt++]=t;
					}
					myNumEmitTri=cnt;
					if(cnt>0) cudaMemcpyAsync(myEmitTriIdx, myEmitIdxHost, (size_t)cnt*sizeof(int), cudaMemcpyHostToDevice, myStream);
				}
			}
		}

		// ---- environment HDRI: wrap input 4's cudaArray as a texture object (rebuild only when the array changes) ----
		cudaArray_t hArr = hdriArr ? hdriArr->cudaArray : nullptr;
		if(hArr != myHdriArray){
			if(myHdriTex){ cudaDestroyTextureObject(myHdriTex); myHdriTex=0; }
			if(hArr){
				cudaResourceDesc rd={}; rd.resType=cudaResourceTypeArray; rd.res.array.array=hArr;
				cudaTextureDesc td={}; td.addressMode[0]=cudaAddressModeWrap; td.addressMode[1]=cudaAddressModeClamp;
				td.filterMode=cudaFilterModeLinear; td.readMode=cudaReadModeElementType; td.normalizedCoords=1;
				if(cudaCreateTextureObject(&myHdriTex,&rd,&td,nullptr)!=cudaSuccess) myHdriTex=0;
			}
			myHdriArray=hArr;   // NB: don't reset accumulation here (TD may hand a fresh array each cook)
		}
		bool hdriOn = (skyHdri && myHdriTex!=0);

		// base color: split the POP's stacked Cd buffer into a LAYERED array (N material layers), keyed on
		// device by triCd.w matID. The POP (getParPOP "Texturepop" -> Cd) is a CUDA-resident linear float4
		// buffer of texel colors, row-major, delivered deadlock-safe (replaces the Vulkan input-5 Movie File In).
		// Per-layer dims come from the point count assuming SQUARE layers (matbake bakes layerH x layerH per
		// material): totalPts = layerH*layerH*nMat. numMaterials==1 => one global map. The Texturepop's Cd MUST
		// be 4-component (float4) so the linear buffer band-copies straight into the float4 layered array.
		int nMat = nummaterialsP;
		int texLayerH=0, texW=0;
		if(popDev && nMat>0 && popTotalPts>0){
			long per = popTotalPts/nMat;
			int s=(int)(sqrt((double)per)+0.5);
			if((long)s*s*nMat==popTotalPts){ texLayerH=s; texW=s; }   // square-layer layout verified
		}
		bool dimsChanged = (texW!=myBcW || texLayerH!=myBcLayerH || nMat!=myBcNumMat);
		if(dimsChanged){
			if(myBaseColorTex){ cudaDestroyTextureObject(myBaseColorTex); myBaseColorTex=0; }
			if(myBaseColorLayered){ cudaFreeArray(myBaseColorLayered); myBaseColorLayered=nullptr; }
			if(texW>0 && texLayerH>0){
				cudaChannelFormatDesc ch = cudaCreateChannelDesc<float4>();
				if(cudaMalloc3DArray(&myBaseColorLayered, &ch, make_cudaExtent(texW, texLayerH, nMat), cudaArrayLayered)!=cudaSuccess) myBaseColorLayered=nullptr;
				if(myBaseColorLayered){
					cudaResourceDesc rd={}; rd.resType=cudaResourceTypeArray; rd.res.array.array=myBaseColorLayered;
					cudaTextureDesc td={}; td.addressMode[0]=cudaAddressModeWrap; td.addressMode[1]=cudaAddressModeWrap;
					td.filterMode=cudaFilterModeLinear; td.readMode=cudaReadModeElementType; td.normalizedCoords=1;
					if(cudaCreateTextureObject(&myBaseColorTex,&rd,&td,nullptr)!=cudaSuccess) myBaseColorTex=0;
				}
			}
			myBcW=texW; myBcLayerH=texLayerH; myBcNumMat=nMat;
		}
		if(popDev && myBaseColorLayered){   // re-copy every cook: the POP re-cooks live -> animated textures
			for(int i=0;i<nMat;i++){          // band i = points [i*texW*texLayerH ..) (row-major float4) -> layer i
				cudaMemcpy3DParms p={};
				p.srcPtr=make_cudaPitchedPtr((char*)popDev + (size_t)i*texW*texLayerH*sizeof(float4), (size_t)texW*sizeof(float4), (size_t)texW*sizeof(float4), (size_t)texLayerH);
				p.dstArray=myBaseColorLayered; p.dstPos=make_cudaPos(0,0,(size_t)i);
				p.extent=make_cudaExtent(texW, texLayerH, 1); p.kind=cudaMemcpyDeviceToDevice;
				cudaMemcpy3D(&p);
			}
		}

		// ENV 3: (re)build the HDRI importance map when needed (throttled to ~every 15 cooks; rebuild on pulse / swap)
		bool wantImp = hdriOn && envImportanceOn;
		if(wantImp){
			bool need = !myEnvReady || (hArr!=myEnvBuiltArray) || myEnvRebuildReq;
			if(need && (myEnvRebuildReq || myEnvLastBuild<0 || (myExecuteCount-myEnvLastBuild)>=15)){
				if(buildEnvCDF(hdriIn,hdriArr)){ myEnvBuiltArray=hArr; myEnvLastBuild=(int)myExecuteCount; }
			}
		}
		myEnvRebuildReq=false;

		// ---- analytic Light COMPs: parse the "Lightdata" string -> device buffer ----
		// format: "N  px py pz dx dy dz cr cg cb type radius cosI cosO   px py pz ..." (13 floats per light)
		myNumPtLights=0;
		if(lightdata && *lightdata){
			char* p=(char*)lightdata;
			long n=strtol(p,&p,10);
			if(n>0 && n<=4096){
				std::vector<PTLight> lv; lv.reserve((size_t)n);
				for(long i=0;i<n;i++){
					float f[13]; bool ok=true;
					for(int k=0;k<13;k++){ char* q=p; f[k]=strtof(p,&p); if(p==q){ ok=false; break; } }
					if(!ok) break;
					PTLight L;
					L.pos=H3(f[0],f[1],f[2]); L.dir=hnorm(H3(f[3],f[4],f[5])); L.radiance=H3(f[6],f[7],f[8]);
					L.type=(int)(f[9]+0.5f); L.radius=f[10]; L.cosInner=f[11]; L.cosOuter=f[12];
					lv.push_back(L);
				}
				int N=(int)lv.size();
				if(N>0){
					if(N>myPtLightCap){ if(myPtLights)cudaFree(myPtLights); cudaMalloc((void**)&myPtLights,(size_t)N*sizeof(PTLight)); myPtLightCap=N; }
					cudaMemcpy(myPtLights,lv.data(),(size_t)N*sizeof(PTLight),cudaMemcpyHostToDevice);
					myNumPtLights=N;
				}
			}
		}

		if(orbit) myOrbitAngle += (float)orbitSpeed*0.01f;
		float  aspect=(float)W/(float)H;
		float3 eye, fwd, up=H3(0,1,0); float focus, fovY;
		if(usecamera){                                  // follow a TD Camera COMP (Eye/Forward/Camfov bound by expr)
			eye=H3((float)eyex,(float)eyey,(float)eyez);
			fwd=hnorm(H3((float)fwdx,(float)fwdy,(float)fwdz));
			float fovH=(float)(camfov*3.14159265/180.0);
			fovY=2.0f*atanf(tanf(fovH*0.5f)/aspect); focus=(float)(dist>0.5?dist:5.0);   // TD fov horizontal -> vertical
		}else{
			float3 target=H3(0.0f,1.0f,0.0f); float D=(float)(dist>0.5?dist:13.0);
			eye=H3(target.x + D*sinf(myOrbitAngle), 2.2f, target.z + D*cosf(myOrbitAngle));
			fwd=hnorm(hsub(target,eye)); focus=hlen(hsub(target,eye));
			float fovHo=(float)(camfov*3.14159265/180.0); fovY=2.0f*atanf(tanf(fovHo*0.5f)/aspect);   // orbit now respects the FOV param too
		}
		float3 right=hnorm(hcross(fwd,up));
		float3 vup=hcross(right,fwd);
		float  halfH=tanf(fovY*0.5f)*focus, halfW=halfH*aspect;
		float3 cu=hscl(right,halfW), cv=hscl(vup,halfH), cw=hscl(fwd,focus);

		// capture previous-frame camera basis BEFORE update (temporal motion vectors)
		bool   vprev=myHavePrev;
		float3 pE=myHavePrev?myPrevEye:eye, pU=myHavePrev?myPrevU:cu, pV=myHavePrev?myPrevV:cv, pW=myHavePrev?myPrevW:cw;

		bool moved = !myHavePrev || hlen(hsub(eye,myPrevEye))>1e-5f;
		bool bgChanged = (bgMode != myPrevBgMode); myPrevBgMode = bgMode;   // switch background -> restart accumulation
		int  curSkyMode = hdriOn?1:(skyPhysical?2:0);
		bool skyChanged = (curSkyMode!=myPrevSkyMode) || (fabsf((float)hdrirot-myPrevHdriRot)>1e-4f);   // sky mode / HDRI rotation
		myPrevSkyMode=curSkyMode; myPrevHdriRot=(float)hdrirot;
		// Projscale/Texmode edit -> reset ONLY the progressive accumulator (myFrameIndex) so it shows live.
		// Do NOT touch myRRReset here: totalCooks-based "geoChanged" fired every frame and reset the RR/DLSS
		// denoiser history every cook, wedging NGX after ~1k frames (the step-91 post-render freeze).
		bool texMapChanged = (fabsf((float)projscale-myPrevProjScale)>1e-6f) || (texmode!=myPrevTexMode);
		myPrevProjScale=(float)projscale; myPrevTexMode=texmode;
		// fog edits -> reset the progressive accumulator so they show live (single signature over all fog params)
		float fogSig = fogEnable ? (1000.0f + (float)fogDensity*131.0f + (float)fogAniso*17.0f + (float)(fogcr*3.0+fogcg*5.0+fogcb*11.0)
		                            + (fogEmitNEE?2000.0f:0.0f) + (float)fogSkyStr*97.0f + (fogSingle?7000.0f:0.0f)
		                            + (float)fogMaxScat*53.0f + (float)fogRRStart*211.0f + (float)fogStability*311.0f
		                            + (float)fogFireflyMax*173.0f
		                            + (float)fogShaftG*149.0f + (float)fogContrast*229.0f) : 0.0f;
		bool fogChanged = fabsf(fogSig - myPrevFogSig) > 1e-6f; myPrevFogSig = fogSig;
		if(fogChanged) myFogReset = true;   // fog look changed -> the fog EMA history is stale (resets EMA, NOT on camera motion)
		if(myResetReq || moved || bgChanged || skyChanged || texMapChanged || fogChanged){ myFrameIndex=0; myResetReq=false; }
		myFrameIndex++;
		myPrevEye=eye; myPrevU=cu; myPrevV=cv; myPrevW=cw; myHavePrev=true;

			// per-frame Halton jitter (free-running so it varies during orbit too) + camera matrices for RR
			unsigned jidx=(unsigned)(myExecuteCount & 15u)+1u;
			auto halton=[](unsigned i,unsigned b){ float f=1.0f,r=0.0f; while(i){ f/=(float)b; r+=f*(float)(i%b); i/=b; } return r; };
			float hx=halton(jidx,2), hy=halton(jidx,3);
			float view16[16], proj16[16];
			{
				float dRE=right.x*eye.x+right.y*eye.y+right.z*eye.z;
				float dUE=vup.x*eye.x+vup.y*eye.y+vup.z*eye.z;
				float dFE=fwd.x*eye.x+fwd.y*eye.y+fwd.z*eye.z;
				view16[0]=right.x; view16[1]=right.y; view16[2]=right.z; view16[3]=-dRE;   // world->view (row-major, M*v)
				view16[4]=vup.x;   view16[5]=vup.y;   view16[6]=vup.z;   view16[7]=-dUE;
				view16[8]=fwd.x;   view16[9]=fwd.y;   view16[10]=fwd.z;  view16[11]=-dFE;
				view16[12]=0; view16[13]=0; view16[14]=0; view16[15]=1;
				float zn=0.1f, zf=1000.0f, ty=1.0f/tanf(fovY*0.5f), tx=ty/aspect;          // view->clip (LH, z in [0,1])
				for(int k=0;k<16;k++) proj16[k]=0.0f;
				proj16[0]=tx; proj16[5]=ty; proj16[10]=zf/(zf-zn); proj16[11]=-zn*zf/(zf-zn); proj16[14]=1.0f;
			}
			if(rr && !myPrevRR) myRRReset=true; myPrevRR=rr;
			int rrReset = myRRReset?1:0; myRRReset=false;

		LaunchParamsDemo lp={};
		lp.accum=(float4*)myAccum; lp.image=myImage; lp.albedo=myAlbedo; lp.normal=myNormal; lp.width=(unsigned)W; lp.height=(unsigned)H; lp.frameIndex=myFrameIndex;
		lp.sampleSeed = boilDecorrelate ? (unsigned)myExecuteCount : 0u;   // 0 = frameIndex (exact prior); free-running cook counter never resets on motion
		lp.spp=(unsigned)(spp<1?1:spp); lp.maxDepth=(unsigned)(maxd<1?1:maxd); lp.handle=triMode?myTriGasHandle:myGasHandle; lp.useInput=triMode?1:0; lp.triVerts=(float4*)myTriVerts;
		lp.triCd=(triMode&&triHaveCd)?(float4*)myTriCd:nullptr; lp.triMat=(triMode&&triHaveMat)?(float4*)myTriMat:nullptr; lp.triN=(triMode&&triHaveN)?(float4*)myTriN:nullptr;
		lp.triUV=(triMode&&triHaveUV)?(float4*)myTriUV:nullptr; lp.showUV=showuv?1:0;
		lp.baseColorTex=myBaseColorTex; lp.useBaseColor=(triMode && myBaseColorTex)?1:0; lp.projScale=(float)projscale; lp.texMode=texmode; lp.numMaterials=myBcNumMat;
		lp.cam_eye=eye; lp.cam_u=cu; lp.cam_v=cv; lp.cam_w=cw;
		lp.prev_eye=pE; lp.prev_u=pU; lp.prev_v=pV; lp.prev_w=pW; lp.validPrev=vprev?1:0; lp.flow=(float4*)myFlow;
		lp.aperture=(float)aperture;
		lp.materials=(DemoMaterial*)myMaterials;
		lp.neeEnable=(nee && myNumLights>0 && !triMode)?1:0; lp.numLights=myNumLights; lp.totalPower=myTotalPower;
		lp.lightPos=(float4*)myLightPos; lp.lightEmit=(float4*)myLightEmit; lp.lightCDF=(float*)myLightCDF;
		lp.ptLights=myPtLights; lp.numPtLights=myNumPtLights; lp.lightMaster=(float)lightintensity;
		lp.sunDir=hnorm(H3((float)sundx,(float)sundy,(float)sundz));
		lp.sunColor=H3((float)(suncr*sunstr),(float)(suncg*sunstr),(float)(suncb*sunstr));
		lp.sunAngle=(float)(sunang*0.01745329252);   // degrees -> radians (soft-shadow cone half-angle)
		lp.bgMode=bgMode; lp.bgSolid=H3((float)bgcr,(float)bgcg,(float)bgcb);
		lp.skyStrength=(float)skystr; lp.skyZenith=H3((float)skyzr,(float)skyzg,(float)skyzb); lp.skyHorizon=H3((float)skyhr,(float)skyhg,(float)skyhb);
		lp.skyMode=hdriOn?1:(skyPhysical?2:0); lp.hdriTex=myHdriTex; lp.hdriRot=(float)(hdrirot/360.0);   // rotation degrees -> turns
		lp.envImportance=(wantImp && myEnvReady)?1:0;
		lp.envW=myEnvW; lp.envH=myEnvH; lp.envCondCdf=myEnvCondCdf; lp.envMargCdf=myEnvMargCdf; lp.envFunc=myEnvFunc; lp.envFuncInt=myEnvFuncInt;
		// ENV 4: Preetham physical-sky coefficients (from sun elevation + turbidity; used only when skyMode==2)
		{
			float T=(float)turbidity; if(T<1.7f)T=1.7f;
			float3 sd=hnorm(H3((float)sundx,(float)sundy,(float)sundz));
			float ct=fminf(fmaxf(sd.y,0.0f),1.0f), thetaS=acosf(ct);
			float Ax=-0.0193f*T-0.2592f,Bx=-0.0665f*T+0.0008f,Cx=-0.0004f*T+0.2125f,Dx=-0.0641f*T-0.8989f,Ex=-0.0033f*T+0.0452f;
			float Ay=-0.0167f*T-0.2608f,By=-0.0950f*T+0.0092f,Cy=-0.0079f*T+0.2102f,Dy=-0.0441f*T-1.6537f,Ey=-0.0109f*T+0.0529f;
			float AY= 0.1787f*T-1.4630f,BY=-0.3554f*T+0.4275f,CY=-0.0227f*T+5.3251f,DY= 0.1206f*T-2.5771f,EY=-0.0670f*T+0.3703f;
			float ts2=thetaS*thetaS, ts3=ts2*thetaS;
			float xz=( 0.00166f*ts3-0.00375f*ts2+0.00209f*thetaS)*T*T+(-0.02903f*ts3+0.06377f*ts2-0.03202f*thetaS+0.00394f)*T+( 0.11693f*ts3-0.21196f*ts2+0.06052f*thetaS+0.25886f);
			float yz=( 0.00275f*ts3-0.00610f*ts2+0.00317f*thetaS)*T*T+(-0.04214f*ts3+0.08970f*ts2-0.04153f*thetaS+0.00516f)*T+( 0.15346f*ts3-0.26756f*ts2+0.06670f*thetaS+0.26688f);
			float chi=(4.0f/9.0f-T/120.0f)*(3.14159265f-2.0f*thetaS);
			float Yz=(4.0453f*T-4.9710f)*tanf(chi)-0.2155f*T+2.4192f; if(Yz<0.0f)Yz=0.0f;
			float Fx0=fmaxf((1.0f+Ax*expf(Bx))*(1.0f+Cx*expf(Dx*thetaS)+Ex*ct*ct),1e-3f);
			float Fy0=fmaxf((1.0f+Ay*expf(By))*(1.0f+Cy*expf(Dy*thetaS)+Ey*ct*ct),1e-3f);
			float FY0=fmaxf((1.0f+AY*expf(BY))*(1.0f+CY*expf(DY*thetaS)+EY*ct*ct),1e-3f);
			lp.perezA=H3(Ax,Ay,AY); lp.perezB=H3(Bx,By,BY); lp.perezC=H3(Cx,Cy,CY); lp.perezD=H3(Dx,Dy,DY); lp.perezE=H3(Ex,Ey,EY);
			lp.perezZenith=H3(xz,yz,Yz); lp.perezNorm=H3(Fx0,Fy0,FY0);
		}
		lp.jitter=(float)jitter; lp.fireflyMax=(float)firefly;
		lp.fogEnable=fogEnable?1:0; lp.fogDensity=(float)fogDensity; lp.fogColor=H3((float)fogcr,(float)fogcg,(float)fogcb); lp.fogG=(float)fogAniso;
		lp.fogEmitNEE=(fogEnable && fogEmitNEE && triMode && myNumEmitTri>0)?1:0; lp.numEmitTri=myNumEmitTri; lp.emitTriIdx=myEmitTriIdx;
		lp.fogSkyStr=(float)fogSkyStr; lp.fogSingleScatter=fogSingle?1:0; lp.fogMaxScatter=fogMaxScat; lp.fogRRStart=(float)fogRRStart; lp.fogFireflyMax=(float)fogFireflyMax;
		lp.fogAccum=(float4*)myFogAccum; lp.fogStability=(float)fogStability; lp.fogReset=myFogReset?1:0; myFogReset=false;   // motion-robust fog smoothing
		lp.fogShaftG=(float)fogShaftG; lp.fogContrast=(float)fogContrast;   // god-ray crispiness (shaft g + in-scatter contrast)
		lp.taaEnable=taa?1:0; lp.taaMaxHist=(float)maxhist;
		if(myTaaParity==0){ lp.taaPrevCol=(float4*)myTaaCol1; lp.taaCurCol=(float4*)myTaaCol0; }
		else            { lp.taaPrevCol=(float4*)myTaaCol0; lp.taaCurCol=(float4*)myTaaCol1; }
		myTaaParity ^= 1;
		lp.rrEnable=rr?1:0; lp.rrColor=(float4*)myRRColor; lp.rrSpecAlbedo=(float4*)myRRSpecAlb; lp.rrRoughness=myRRRough; lp.rrDepth=myRRDepth; lp.rrMotion=myRRMotion;
			lp.rrHitDist=myRRHitDist; lp.rrJitterX=hx; lp.rrJitterY=hy; lp.rrSpecMV=(float)rrspecmv;

		cudaMemcpyAsync((void*)myParamsBuf,&lp,sizeof(lp),cudaMemcpyHostToDevice,myStream);
		OXD_LOGF(" -> optixLaunch %ux%u", (unsigned)W,(unsigned)H);
		OXD_STEP(90);
		OptixResult lr=optixLaunch(myPipeline,myStream,myParamsBuf,sizeof(lp),&mySbt,(unsigned)W,(unsigned)H,1);
		OXD_STEP(91);
		OXD_LOGF(" <- optixLaunch %s", lr==OPTIX_SUCCESS?"ok":"FAILED");
		if(lr!=OPTIX_SUCCESS) myError="optixLaunch failed";

		float4* result=myImage; bool rrDone=false;
		if(rr && myRR.initialized())
		{
			myRR.ensure(W,H);
			float jrrx=-(hx-0.5f), jrry=-(hy-0.5f);   // RR jitter convention (reference: -(jitter-0.5))
			if(myRR.evaluate((float4*)myRRColor,(float4*)myAlbedo,(float4*)myRRSpecAlb,(float4*)myNormal,
			                 myRRRough,myRRDepth,myRRMotion,myRRHitDist,
			                 (float4*)myDenoised, view16, proj16, jrrx, jrry, rrReset,
			                 (float)flowscale, (float)flowscale, flowinvy?1:0, myStream))
				{ result=myDenoised; rrDone=true; }
		}
		// graceful fallback: the OptiX denoiser also runs when Rr was requested but DLSS-RR was
		// unavailable or failed this frame (rrDone==false) -> non-RTX / old-driver users still get
		// a denoised (not raw-noisy) image instead of falling through to the raw path-trace.
		if(!rrDone && (denoise || rr) && myDenoiser && myDenoiserState)
		{
			OptixImage2D inI=mkImg((CUdeviceptr)myImage,W,H), albI=mkImg((CUdeviceptr)myAlbedo,W,H), nrmI=mkImg((CUdeviceptr)myNormal,W,H), outI=mkImg((CUdeviceptr)myDenoised,W,H);
			optixDenoiserComputeIntensity(myDenoiser,myStream,&inI,(CUdeviceptr)myIntensity,(CUdeviceptr)myDenoiserScratch,myScratchSize);
			OptixDenoiserParams dp={}; dp.hdrIntensity=(CUdeviceptr)myIntensity;
			dp.blendFactor = 1.0f - (float)denoisestr;   // strength 1 = full AI denoise, 0 = raw path-trace
			dp.temporalModeUsePreviousLayers = myHavePrevDenoised ? 1u : 0u;
			dp.flowMulX = (float)flowscale; dp.flowMulY = (float)flowscale;   // tune / flip (-1) the motion vectors
			OptixDenoiserGuideLayer guide={}; guide.albedo=albI; guide.normal=nrmI; guide.flow=mkImg((CUdeviceptr)myFlow,W,H,OPTIX_PIXEL_FORMAT_FLOAT2);
			OptixDenoiserLayer layer={}; layer.input=inI; layer.output=outI;
			layer.previousOutput = myHavePrevDenoised ? mkImg((CUdeviceptr)myPrevDenoised,W,H) : inI;
			OptixResult dr=optixDenoiserInvoke(myDenoiser,myStream,&dp,(CUdeviceptr)myDenoiserState,myStateSize,&guide,&layer,1,0,0,(CUdeviceptr)myDenoiserScratch,myScratchSize);
			if(dr==OPTIX_SUCCESS){ result=myDenoised; cudaMemcpyAsync((void*)myPrevDenoised,(void*)myDenoised,(size_t)W*H*sizeof(float4),cudaMemcpyDeviceToDevice,myStream); myHavePrevDenoised=true; }
		}
		else if(!rrDone) { myHavePrevDenoised=false; }
		if(bgMode==2 && result!=myImage)   // Transparent: denoisers reconstruct RGB only -> restore the path-traced matte (.w)
			cudaMemcpy2DAsync((char*)result+12,sizeof(float4),(char*)myImage+12,sizeof(float4),sizeof(float),(size_t)W*H,cudaMemcpyDeviceToDevice,myStream);
		cudaMemcpy2DToArrayAsync(outInfo->cudaArray,0,0,(void*)result,(size_t)W*sizeof(float4),(size_t)W*sizeof(float4),(size_t)H,cudaMemcpyDeviceToDevice,myStream);
	}
	else
	{
		myError = (myReadyResult!=-999) ? (myLog[0]?myLog:"OptiX init/build failed") : "initializing OptiX...";
	}

	// Drain our stream BEFORE endCUDAOperations so each cook is self-contained — no async write to the
	// output array crosses into the next cook. This does NOT by itself prevent the output-resize deadlock
	// (handled TD-side by the "Update Resolution" button pausing the COMP — see createCUDAArray above); it
	// COMPLEMENTS that pause: when the COMP resumes after a resize, the resume-cook's createCUDAArray sees
	// no in-flight work referencing the old output. Only LEGAL place to sync (still inside begin/end). The
	// CPU waits for GPU work it would block on at display anyway, so the steady-state cost is negligible.
	OXD_STEP(97);
	if(myStream) cudaStreamSynchronize(myStream);
	OXD_STEP(98);
	myContext->endCUDAOperations(nullptr);
	OXD_STEP(99);
	OXD_LOGF("=== execute #%d end (cook complete) ===", myExecuteCount);
}

int32_t OptixDemoTOP::getNumInfoCHOPChans(void*) { return 13; }
void OptixDemoTOP::getInfoCHOPChan(int32_t i, OP_InfoCHOPChan* c, void*)
{
	if(i==0){ c->name->setString("executeCount"); c->value=(float)myExecuteCount; }
	else if(i==1){ c->name->setString("optixOK"); c->value=(myOptixResult==0)?1.0f:0.0f; }
	else if(i==2){ c->name->setString("ready"); c->value=myReady?1.0f:0.0f; }
	else if(i==3){ c->name->setString("frameIndex"); c->value=(float)myFrameIndex; }
	else if(i==4){ c->name->setString("spheres"); c->value=(float)mySphereCount; }
	else if(i==5){ c->name->setString("rrInit"); c->value=myRR.initialized()?1.0f:0.0f; }
	else if(i==6){ c->name->setString("rrResult"); c->value=(float)myRR.lastResult(); }
	else if(i==7){ c->name->setString("lights"); c->value=(float)myNumLights; }
	else if(i==8){ c->name->setString("envReady"); c->value=myEnvReady?1.0f:0.0f; }
	else if(i==9){ c->name->setString("ptLights"); c->value=(float)myNumPtLights; }
	else if(i==10){ c->name->setString("emitTris"); c->value=(float)myNumEmitTri; }
	else if(i==11){ c->name->setString("scanTris"); c->value=(float)myDbgScanTri; }
	else if(i==12){ c->name->setString("firstMat"); c->value=myDbgFirstMat; }
}
void OptixDemoTOP::getErrorString(OP_String* e, void*) { e->setString(myError); }

void OptixDemoTOP::setupParameters(OP_ParameterManager* m, void*)
{
	// ---------- Render ----------
	{ OP_NumericParameter np; np.name="Spp"; np.label="Samples / Pixel"; np.page="Render"; np.defaultValues[0]=4.0; np.minValues[0]=1.0; np.maxValues[0]=64.0; np.minSliders[0]=1.0; np.maxSliders[0]=16.0; np.clampMins[0]=true; np.clampMaxes[0]=false; m->appendFloat(np); }
	{ OP_NumericParameter np; np.name="Maxdepth"; np.label="Max Bounces"; np.page="Render"; np.defaultValues[0]=8.0; np.minValues[0]=1.0; np.maxValues[0]=32.0; np.minSliders[0]=1.0; np.maxSliders[0]=16.0; np.clampMins[0]=true; np.clampMaxes[0]=true; m->appendFloat(np); }
	{ OP_NumericParameter np; np.name="Firefly"; np.label="Firefly Clamp (0=off)"; np.page="Render"; np.defaultValues[0]=10.0; np.minValues[0]=0.0; np.maxValues[0]=2000.0; np.minSliders[0]=0.0; np.maxSliders[0]=50.0; np.clampMins[0]=true; np.clampMaxes[0]=false; m->appendFloat(np); }
	{ OP_NumericParameter np; np.name="Nee"; np.label="Direct Light Sampling (NEE)"; np.page="Render"; np.defaultValues[0]=0.0; m->appendToggle(np); }
	{ OP_NumericParameter np; np.name="Reset"; np.label="Reset Accumulation"; np.page="Render"; m->appendPulse(np); }

	// ---------- Denoise ----------
	{ OP_StringParameter sp; sp.name="Denoiser"; sp.label="Denoiser"; sp.page="Denoise"; sp.defaultValue="Optix";
	  const char* nm[]={"None","Optix","Taa","Rr"}; const char* lb[]={"None (raw)","OptiX AI","TAA (temporal)","DLSS Ray Reconstruction"};
	  m->appendMenu(sp, 4, nm, lb); }
	{ OP_NumericParameter np; np.name="Denoisestr"; np.label="OptiX Denoise Strength"; np.page="Denoise"; np.defaultValues[0]=0.7; np.minValues[0]=0.0; np.maxValues[0]=1.0; np.minSliders[0]=0.0; np.maxSliders[0]=1.0; np.clampMins[0]=true; np.clampMaxes[0]=true; m->appendFloat(np); }
	{ OP_NumericParameter np; np.name="Maxhist"; np.label="TAA Max History"; np.page="Denoise"; np.defaultValues[0]=32.0; np.minValues[0]=1.0; np.maxValues[0]=512.0; np.minSliders[0]=1.0; np.maxSliders[0]=128.0; np.clampMins[0]=true; np.clampMaxes[0]=true; m->appendFloat(np); }
	{ OP_NumericParameter np; np.name="Jitter"; np.label="AA Jitter"; np.page="Denoise"; np.defaultValues[0]=1.0; np.minValues[0]=0.0; np.maxValues[0]=1.0; np.minSliders[0]=0.0; np.maxSliders[0]=1.0; np.clampMins[0]=true; np.clampMaxes[0]=true; m->appendFloat(np); }
	{ OP_NumericParameter np; np.name="Flowscale"; np.label="RR Motion Scale"; np.page="Denoise"; np.defaultValues[0]=-1.0; np.minValues[0]=-2.0; np.maxValues[0]=2.0; np.minSliders[0]=-2.0; np.maxSliders[0]=2.0; np.clampMins[0]=false; np.clampMaxes[0]=false; m->appendFloat(np); }
	{ OP_NumericParameter np; np.name="Flowinvy"; np.label="RR Flip MV Y"; np.page="Denoise"; np.defaultValues[0]=1.0; m->appendToggle(np); }
	{ OP_NumericParameter np; np.name="Rrspecmv"; np.label="RR Specular MV"; np.page="Denoise"; np.defaultValues[0]=1.0; np.minValues[0]=0.0; np.maxValues[0]=1.0; np.minSliders[0]=0.0; np.maxSliders[0]=1.0; np.clampMins[0]=true; np.clampMaxes[0]=true; m->appendFloat(np); }
	{ OP_NumericParameter np; np.name="Boildecorrelate"; np.label="Decorrelate Noise (anti-boil)"; np.page="Denoise"; np.defaultValues[0]=0.0; m->appendToggle(np); }

	// ---------- Camera ----------
	{ OP_NumericParameter np; np.name="Usecamera"; np.label="Use Camera COMP"; np.page="Camera"; np.defaultValues[0]=0.0; m->appendToggle(np); }
	{ OP_StringParameter sp; sp.name="Camera"; sp.label="Camera COMP"; sp.page="Camera"; sp.defaultValue="pcam"; m->appendString(sp); }
	{ OP_NumericParameter np; np.name="Orbit"; np.label="Orbit (fallback cam)"; np.page="Camera"; np.defaultValues[0]=0.0; m->appendToggle(np); }
	{ OP_NumericParameter np; np.name="Orbitspeed"; np.label="Orbit Speed"; np.page="Camera"; np.defaultValues[0]=1.0; np.minValues[0]=-10.0; np.maxValues[0]=10.0; np.minSliders[0]=-5.0; np.maxSliders[0]=5.0; np.clampMins[0]=false; np.clampMaxes[0]=false; m->appendFloat(np); }
	{ OP_NumericParameter np; np.name="Distance"; np.label="Orbit Distance / Focus"; np.page="Camera"; np.defaultValues[0]=13.0; np.minValues[0]=4.0; np.maxValues[0]=40.0; np.minSliders[0]=6.0; np.maxSliders[0]=25.0; np.clampMins[0]=true; np.clampMaxes[0]=true; m->appendFloat(np); }
	{ OP_NumericParameter np; np.name="Aperture"; np.label="DOF Aperture"; np.page="Camera"; np.defaultValues[0]=0.0; np.minValues[0]=0.0; np.maxValues[0]=0.6; np.minSliders[0]=0.0; np.maxSliders[0]=0.3; np.clampMins[0]=true; np.clampMaxes[0]=true; m->appendFloat(np); }
	{ OP_NumericParameter np; np.name="Eye"; np.label="Cam Eye (driven)"; np.page="Camera"; for(int i=0;i<3;i++){ np.defaultValues[i]=0.0; np.minValues[i]=-1000.0; np.maxValues[i]=1000.0; np.minSliders[i]=-20.0; np.maxSliders[i]=20.0; np.clampMins[i]=false; np.clampMaxes[i]=false; } m->appendXYZ(np); }
	{ OP_NumericParameter np; np.name="Forward"; np.label="Cam Forward (driven)"; np.page="Camera"; for(int i=0;i<3;i++){ np.defaultValues[i]=(i==2?-1.0:0.0); np.minValues[i]=-1.0; np.maxValues[i]=1.0; np.minSliders[i]=-1.0; np.maxSliders[i]=1.0; np.clampMins[i]=false; np.clampMaxes[i]=false; } m->appendXYZ(np); }
	{ OP_NumericParameter np; np.name="Camfov"; np.label="Cam FOV (driven)"; np.page="Camera"; np.defaultValues[0]=45.0; np.minValues[0]=1.0; np.maxValues[0]=170.0; np.minSliders[0]=10.0; np.maxSliders[0]=120.0; np.clampMins[0]=true; np.clampMaxes[0]=true; m->appendFloat(np); }

	// ---------- Input ----------
	{ OP_NumericParameter np; np.name="Useinput"; np.label="Use Input Geometry"; np.page="Input"; np.defaultValues[0]=0.0; m->appendToggle(np); }
	{ OP_NumericParameter np; np.name="Numverts"; np.label="Input Vert Count"; np.page="Input"; np.defaultValues[0]=0; np.minValues[0]=0; np.maxValues[0]=5000000; np.minSliders[0]=0; np.maxSliders[0]=300000; np.clampMins[0]=true; np.clampMaxes[0]=false; m->appendInt(np); }
	{ OP_NumericParameter np; np.name="Showuv"; np.label="Show UVs (debug)"; np.page="Input"; np.defaultValues[0]=0.0; m->appendToggle(np); }
	{ OP_NumericParameter np; np.name="Nummaterials"; np.label="Num Materials (tex layers)"; np.page="Input"; np.defaultValues[0]=1; np.minValues[0]=1; np.maxValues[0]=64; np.minSliders[0]=1; np.maxSliders[0]=16; np.clampMins[0]=true; np.clampMaxes[0]=true; m->appendInt(np); }
	{ OP_StringParameter sp; sp.name="Texturepop"; sp.label="Texture POP (CUDA base color)"; sp.page="Input"; m->appendPOP(sp); }
	{ OP_StringParameter sp; sp.name="Geopop"; sp.label="Geometry POP (CUDA UV/Tex)"; sp.page="Input"; sp.defaultValue="soupFacet"; m->appendPOP(sp); }

	// ---------- Environment ----------
	{ OP_NumericParameter np; np.name="Skyzenith"; np.label="Sky Zenith"; np.page="Environment"; double zd[3]={0.45,0.62,1.0}; for(int i=0;i<3;i++){ np.defaultValues[i]=zd[i]; np.minValues[i]=0.0; np.maxValues[i]=1.0; np.minSliders[i]=0.0; np.maxSliders[i]=1.0; np.clampMins[i]=true; np.clampMaxes[i]=false; } m->appendRGB(np); }
	{ OP_NumericParameter np; np.name="Skyhorizon"; np.label="Sky Horizon"; np.page="Environment"; for(int i=0;i<3;i++){ np.defaultValues[i]=1.0; np.minValues[i]=0.0; np.maxValues[i]=1.0; np.minSliders[i]=0.0; np.maxSliders[i]=1.0; np.clampMins[i]=true; np.clampMaxes[i]=false; } m->appendRGB(np); }
	{ OP_NumericParameter np; np.name="Skystrength"; np.label="Sky Strength"; np.page="Environment"; np.defaultValues[0]=1.0; np.minValues[0]=0.0; np.maxValues[0]=10.0; np.minSliders[0]=0.0; np.maxSliders[0]=3.0; np.clampMins[0]=true; np.clampMaxes[0]=false; m->appendFloat(np); }
	{ OP_NumericParameter np; np.name="Sundir"; np.label="Sun Direction"; np.page="Environment"; double sd[3]={0.4,0.85,0.3}; for(int i=0;i<3;i++){ np.defaultValues[i]=sd[i]; np.minValues[i]=-1.0; np.maxValues[i]=1.0; np.minSliders[i]=-1.0; np.maxSliders[i]=1.0; np.clampMins[i]=false; np.clampMaxes[i]=false; } m->appendXYZ(np); }
	{ OP_NumericParameter np; np.name="Suncolor"; np.label="Sun Color"; np.page="Environment"; double sc[3]={1.0,0.95,0.85}; for(int i=0;i<3;i++){ np.defaultValues[i]=sc[i]; np.minValues[i]=0.0; np.maxValues[i]=1.0; np.minSliders[i]=0.0; np.maxSliders[i]=1.0; np.clampMins[i]=true; np.clampMaxes[i]=false; } m->appendRGB(np); }
	{ OP_NumericParameter np; np.name="Sunstrength"; np.label="Sun Strength"; np.page="Environment"; np.defaultValues[0]=8.0; np.minValues[0]=0.0; np.maxValues[0]=200.0; np.minSliders[0]=0.0; np.maxSliders[0]=50.0; np.clampMins[0]=true; np.clampMaxes[0]=false; m->appendFloat(np); }
	{ OP_NumericParameter np; np.name="Sunangle"; np.label="Sun Angle (soft shadow)"; np.page="Environment"; np.defaultValues[0]=0.5; np.minValues[0]=0.0; np.maxValues[0]=20.0; np.minSliders[0]=0.0; np.maxSliders[0]=5.0; np.clampMins[0]=true; np.clampMaxes[0]=false; m->appendFloat(np); }
	{ OP_StringParameter sp; sp.name="Backgroundmode"; sp.label="Background"; sp.page="Environment"; sp.defaultValue="Environment";
	  const char* nm[]={"Environment","Solid","Transparent"}; const char* lb[]={"Environment (sky)","Solid Color","Transparent (alpha)"};
	  m->appendMenu(sp, 3, nm, lb); }
	{ OP_NumericParameter np; np.name="Bgcolor"; np.label="Background Color"; np.page="Environment"; for(int i=0;i<3;i++){ np.defaultValues[i]=0.0; np.minValues[i]=0.0; np.maxValues[i]=1.0; np.minSliders[i]=0.0; np.maxSliders[i]=1.0; np.clampMins[i]=true; np.clampMaxes[i]=false; } m->appendRGB(np); }
	{ OP_StringParameter sp; sp.name="Skymode"; sp.label="Sky Mode"; sp.page="Environment"; sp.defaultValue="Gradient";
	  const char* nm[]={"Gradient","Hdri","Physical"}; const char* lb[]={"Gradient (sky + sun)","HDRI (equirect, input 5)","Physical sky (Preetham)"};
	  m->appendMenu(sp, 3, nm, lb); }
	{ OP_NumericParameter np; np.name="Hdrirot"; np.label="HDRI Rotation"; np.page="Environment"; np.defaultValues[0]=0.0; np.minValues[0]=-360.0; np.maxValues[0]=360.0; np.minSliders[0]=-180.0; np.maxSliders[0]=180.0; np.clampMins[0]=false; np.clampMaxes[0]=false; m->appendFloat(np); }
	{ OP_NumericParameter np; np.name="Envimportance"; np.label="HDRI Importance Sampling"; np.page="Environment"; np.defaultValues[0]=1.0; m->appendToggle(np); }
	{ OP_NumericParameter np; np.name="Rebuildenv"; np.label="Rebuild Env Map"; np.page="Environment"; m->appendPulse(np); }
	{ OP_NumericParameter np; np.name="Turbidity"; np.label="Turbidity (Physical)"; np.page="Environment"; np.defaultValues[0]=2.5; np.minValues[0]=1.7; np.maxValues[0]=10.0; np.minSliders[0]=1.7; np.maxSliders[0]=8.0; np.clampMins[0]=true; np.clampMaxes[0]=true; m->appendFloat(np); }

	// ---------- Lights ----------
	{ OP_NumericParameter np; np.name="Lightintensity"; np.label="Light Intensity (master)"; np.page="Lights"; np.defaultValues[0]=2.0; np.minValues[0]=0.0; np.maxValues[0]=1000.0; np.minSliders[0]=0.0; np.maxSliders[0]=10.0; np.clampMins[0]=true; np.clampMaxes[0]=false; m->appendFloat(np); }
	{ OP_StringParameter sp; sp.name="Lightdata"; sp.label="Light Data (auto-filled)"; sp.page="Lights"; sp.defaultValue=""; m->appendString(sp); }

	// ---------- Texture (base color map = the 6th input; optional UV = the 7th) ----------
	{ OP_StringParameter sp; sp.name="Texmode"; sp.label="Texture Coords"; sp.page="Texture"; sp.defaultValue="Auto";
	  const char* nm[]={"Auto","Uv","Projection"}; const char* lb[]={"Auto (UV if present, else projection)","Force UV","Force Projection (triplanar)"};
	  m->appendMenu(sp, 3, nm, lb); }
	{ OP_NumericParameter np; np.name="Projscale"; np.label="Projection Scale (units/tile)"; np.page="Texture"; np.defaultValues[0]=1.5; np.minValues[0]=0.01; np.maxValues[0]=100.0; np.minSliders[0]=0.1; np.maxSliders[0]=8.0; np.clampMins[0]=true; np.clampMaxes[0]=false; m->appendFloat(np); }

	// ---------- Fog (Phase 1: homogeneous volumetric single scattering) ----------
	{ OP_NumericParameter np; np.name="Fogenable"; np.label="Volumetric Fog"; np.page="Fog"; np.defaultValues[0]=0.0; m->appendToggle(np); }
	{ OP_NumericParameter np; np.name="Fogdensity"; np.label="Fog Density (sigma_t)"; np.page="Fog"; np.defaultValues[0]=0.04; np.minValues[0]=0.0; np.maxValues[0]=5.0; np.minSliders[0]=0.0; np.maxSliders[0]=0.4; np.clampMins[0]=true; np.clampMaxes[0]=false; m->appendFloat(np); }
	{ OP_NumericParameter np; np.name="Fogcolor"; np.label="Fog Scatter Color"; np.page="Fog"; for(int i=0;i<3;i++){ np.defaultValues[i]=0.9; np.minValues[i]=0.0; np.maxValues[i]=1.0; np.minSliders[i]=0.0; np.maxSliders[i]=1.0; np.clampMins[i]=true; np.clampMaxes[i]=true; } m->appendRGB(np); }
	{ OP_NumericParameter np; np.name="Foganisotropy"; np.label="Fog Anisotropy (g, fwd=sun halo)"; np.page="Fog"; np.defaultValues[0]=0.0; np.minValues[0]=-0.95; np.maxValues[0]=0.95; np.minSliders[0]=-0.9; np.maxSliders[0]=0.9; np.clampMins[0]=true; np.clampMaxes[0]=true; m->appendFloat(np); }
	{ OP_NumericParameter np; np.name="Fogskystr"; np.label="Sky In-scatter"; np.page="Fog"; np.defaultValues[0]=1.0; np.minValues[0]=0.0; np.maxValues[0]=4.0; np.minSliders[0]=0.0; np.maxSliders[0]=2.0; np.clampMins[0]=true; np.clampMaxes[0]=false; m->appendFloat(np); }
	{ OP_NumericParameter np; np.name="Fogemitnee"; np.label="Emitter Glow (NEE emissive geo)"; np.page="Fog"; np.defaultValues[0]=1.0; m->appendToggle(np); }
	// --- quality / performance ---
	{ OP_NumericParameter np; np.name="Fogsinglescatter"; np.label="Single Scatter Only (fast, biased)"; np.page="Fog"; np.defaultValues[0]=0.0; m->appendToggle(np); }
	{ OP_NumericParameter np; np.name="Fogmaxscatter"; np.label="Max Scatter Events (0=uncapped)"; np.page="Fog"; np.defaultValues[0]=4.0; np.minValues[0]=0.0; np.maxValues[0]=64.0; np.minSliders[0]=0.0; np.maxSliders[0]=16.0; np.clampMins[0]=true; np.clampMaxes[0]=true; m->appendFloat(np); }
	{ OP_NumericParameter np; np.name="Fogrrstart"; np.label="Fog RR Start (lower=kill sooner)"; np.page="Fog"; np.defaultValues[0]=0.5; np.minValues[0]=0.02; np.maxValues[0]=1.0; np.minSliders[0]=0.05; np.maxSliders[0]=1.0; np.clampMins[0]=true; np.clampMaxes[0]=true; m->appendFloat(np); }
	{ OP_NumericParameter np; np.name="Fogstability"; np.label="Fog Stability (motion smooth)"; np.page="Fog"; np.defaultValues[0]=0.0; np.minValues[0]=0.0; np.maxValues[0]=1.0; np.minSliders[0]=0.0; np.maxSliders[0]=1.0; np.clampMins[0]=true; np.clampMaxes[0]=true; m->appendFloat(np); }
	{ OP_NumericParameter np; np.name="Fogfireflyclamp"; np.label="Fog Firefly Clamp (0=off)"; np.page="Fog"; np.defaultValues[0]=0.0; np.minValues[0]=0.0; np.maxValues[0]=200.0; np.minSliders[0]=0.0; np.maxSliders[0]=20.0; np.clampMins[0]=true; np.clampMaxes[0]=false; m->appendFloat(np); }
	// --- God-ray Crispiness: one macro slider + an advanced override (RESEARCH §A.5) ---
	{ OP_NumericParameter np; np.name="Crispiness"; np.label="God-Ray Crispiness (macro)"; np.page="Fog"; np.defaultValues[0]=0.0; np.minValues[0]=0.0; np.maxValues[0]=1.0; np.minSliders[0]=0.0; np.maxSliders[0]=1.0; np.clampMins[0]=true; np.clampMaxes[0]=true; m->appendFloat(np); }
	{ OP_NumericParameter np; np.name="Crispdrives"; np.label="Crispiness Drives Fog Knobs"; np.page="Fog"; np.defaultValues[0]=0.0; m->appendToggle(np); }
	{ OP_NumericParameter np; np.name="Fogshaftg"; np.label="Shaft Anisotropy (g; -2 = follow Fog g)"; np.page="Fog"; np.defaultValues[0]=-2.0; np.minValues[0]=-2.0; np.maxValues[0]=0.95; np.minSliders[0]=-2.0; np.maxSliders[0]=0.95; np.clampMins[0]=true; np.clampMaxes[0]=true; m->appendFloat(np); }
	{ OP_NumericParameter np; np.name="Fogcontrast"; np.label="Shaft Contrast"; np.page="Fog"; np.defaultValues[0]=1.0; np.minValues[0]=1.0; np.maxValues[0]=3.0; np.minSliders[0]=1.0; np.maxSliders[0]=2.0; np.clampMins[0]=true; np.clampMaxes[0]=false; m->appendFloat(np); }
}

void OptixDemoTOP::pulsePressed(const char* name, void*) {
	if(!strcmp(name,"Reset")){ myResetReq=true; myFrameIndex=0; myRRReset=true; }
	else if(!strcmp(name,"Rebuildenv")){ myEnvRebuildReq=true; }
}
