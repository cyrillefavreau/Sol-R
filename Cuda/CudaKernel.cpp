/* 
 * Copyright (C) 2011-2012 Cyrille Favreau <cyrille_favreau@hotmail.com>
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Library General Public
 * License as published by the Free Software Foundation; either
 * version 2 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Library General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>. 
 */

/*
 * Author: Cyrille Favreau <cyrille_favreau@hotmail.com>
 *
 */

#define _CRT_SECURE_NO_WARNINGS

#include <math.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include <time.h>
#include <sstream>
#include <stdlib.h>
#include <vector>

#include <cuda_runtime.h>
#ifdef LOGGING
#include <ETWLoggingModule.h>
#include <ETWResources.h>
#else
#define LOG_INFO( msg ) std::cout << msg << std::endl;
#define LOG_ERROR( msg ) std::cerr << msg << std::endl;
#endif

#include "CudaRayTracer.h"
#include "CudaKernel.h"
#include "../Consts.h"

const long MAX_SOURCE_SIZE = 65535;
const long MAX_DEVICES = 10;

#ifndef WIN32
typedef struct BITMAPFILEHEADER {
  short bfType;
  int bfSize;
  short Reserved1;
  short Reserved2;
  int bfOffBits;
};

typedef struct BITMAPINFOHEADER {
  int biSizeImage;
  long biWidth;
  long biHeight;
};
#endif

/*
________________________________________________________________________________

CudaKernel::CudaKernel
________________________________________________________________________________
*/
CudaKernel::CudaKernel() : GPUKernel(),
   m_blockSize(1024,1024,1),
   m_sharedMemSize(256)
{
#ifdef LOGGING
	// Initialize Log
	LOG_INITIALIZE_ETW(
		&GPU_CudaRAYTRACERMODULE,
		&GPU_CudaRAYTRACERMODULE_EVENT_DEBUG,
		&GPU_CudaRAYTRACERMODULE_EVENT_VERBOSE,
		&GPU_CudaRAYTRACERMODULE_EVENT_INFO, 
		&GPU_CudaRAYTRACERMODULE_EVENT_WARNING,
		&GPU_CudaRAYTRACERMODULE_EVENT_ERROR);
#endif // NDEBUG

#if USE_KINECT
	// Initialize Kinect
	NuiInitialize( NUI_INITIALIZE_FLAG_USES_DEPTH_AND_PLAYER_INDEX | NUI_INITIALIZE_FLAG_USES_SKELETON | NUI_INITIALIZE_FLAG_USES_COLOR);

	m_hNextDepthFrameEvent = CreateEvent( NULL, TRUE, FALSE, NULL ); 
	m_hNextVideoFrameEvent = CreateEvent( NULL, TRUE, FALSE, NULL ); 
	m_hNextSkeletonEvent   = CreateEvent( NULL, TRUE, FALSE, NULL );

	m_skeletons = CreateEvent( NULL, TRUE, FALSE, NULL );			 
	NuiSkeletonTrackingEnable( m_skeletons, NUI_SKELETON_TRACKING_FLAG_ENABLE_SEATED_SUPPORT );

	NuiImageStreamOpen( NUI_IMAGE_TYPE_COLOR,                  NUI_IMAGE_RESOLUTION_640x480, 0, 2, m_hNextVideoFrameEvent, &m_pVideoStreamHandle );
	NuiImageStreamOpen( NUI_IMAGE_TYPE_DEPTH_AND_PLAYER_INDEX, NUI_IMAGE_RESOLUTION_320x240, 0, 2, m_hNextDepthFrameEvent, &m_pDepthStreamHandle );

	NuiCameraElevationSetAngle( 0 );
#endif // USE_KINECT

   // Eye position
	m_viewPos.x =   0.f;
	m_viewPos.y =   0.f;
	m_viewPos.z = -40.f;

	// Rotation angles
	m_angles.x = 0.f;
	m_angles.y = 0.f;
	m_angles.z = 0.f;
	m_angles.w = 0.f;

   setPostProcessingInfo( ppe_none, 0.f, 40.f, 50 );
   float4 bkColor = {0.f, 0.f, 0.f, 0.f};
   setSceneInfo( 512, 512, 0.f, true, 10000.f, 0.9f, 5, bkColor, false, 0.f, false, 0, 1000 );
}

/*
________________________________________________________________________________

CudaKernel::~CudaKernel
________________________________________________________________________________
*/
CudaKernel::~CudaKernel()
{
   // Clean up
   releaseDevice();

#if USE_KINECT
   CloseHandle(m_skeletons);
   CloseHandle(m_hNextDepthFrameEvent); 
   CloseHandle(m_hNextVideoFrameEvent); 
   CloseHandle(m_hNextSkeletonEvent);
   NuiShutdown();
#endif // USE_KINECT
}

void CudaKernel::initBuffers() 
{
   GPUKernel::initBuffers();
   deviceQuery();
	initializeDevice();
}

/*
________________________________________________________________________________

Initialize CPU & GPU resources
________________________________________________________________________________
*/
void CudaKernel::initializeDevice()
{
	initialize_scene( m_sceneInfo.width.x, m_sceneInfo.height.x, NB_MAX_PRIMITIVES, NB_MAX_LAMPS, NB_MAX_MATERIALS, NB_MAX_TEXTURES );
}

void CudaKernel::resetBoxesAndPrimitives()
{
   m_nbActiveBoxes = -1;
   m_nbActivePrimitives = -1;
}

/*
________________________________________________________________________________

Release CPU & GPU resources
________________________________________________________________________________
*/
void CudaKernel::releaseDevice()
{
	LOG_INFO("Release device memory\n");
	finalize_scene();
}

/*
________________________________________________________________________________

Execute GPU kernel
________________________________________________________________________________
*/
void CudaKernel::render_begin( const float timer )
{
#if USE_KINECT
	// Video
	const NUI_IMAGE_FRAME* pImageFrame = 0;
	WaitForSingleObject (m_hNextVideoFrameEvent,INFINITE); 
	HRESULT status = NuiImageStreamGetNextFrame( m_pVideoStreamHandle, 0, &pImageFrame ); 
	if(( status == S_OK) && pImageFrame ) 
	{
		INuiFrameTexture* pTexture = pImageFrame->pFrameTexture;
		NUI_LOCKED_RECT LockedRect;
		pTexture->LockRect( 0, &LockedRect, NULL, 0 ) ; 
		if( LockedRect.Pitch != 0 ) 
		{
			m_hVideo = (char*) LockedRect.pBits;
		}
	}

	// Depth
	const NUI_IMAGE_FRAME* pDepthFrame = 0;
	WaitForSingleObject (m_hNextDepthFrameEvent,INFINITE); 
	status = NuiImageStreamGetNextFrame( m_pDepthStreamHandle, 0, &pDepthFrame ); 
	if(( status == S_OK) && pDepthFrame ) 
	{
		INuiFrameTexture* pTexture = pDepthFrame->pFrameTexture;
		if( pTexture ) 
		{
			NUI_LOCKED_RECT LockedRectDepth;
			pTexture->LockRect( 0, &LockedRectDepth, NULL, 0 ) ; 
			if( LockedRectDepth.Pitch != 0 ) 
			{
				m_hDepth = (char*) LockedRectDepth.pBits;
			}
		}
	}
	NuiImageStreamReleaseFrame( m_pVideoStreamHandle, pImageFrame ); 
	NuiImageStreamReleaseFrame( m_pDepthStreamHandle, pDepthFrame ); 

   // copy kinect data to GPU
   h2d_kinect( 
      m_hVideo, gKinectVideo*gKinectVideoHeight*gKinectVideoWidth,
      m_hDepth, gKinectDepth*gKinectDepthHeight*gKinectDepthWidth );
#endif // USE_KINECT

	// CPU -> GPU Data transfers
	h2d_scene( m_hBoundingBoxes, m_nbActiveBoxes+1, m_hBoxPrimitivesIndex, m_hPrimitives, m_nbActivePrimitives+1, m_hLamps, m_nbActiveLamps+1 );
	if( !m_texturedTransfered )
	{
		h2d_materials( 
         m_hMaterials, m_nbActiveMaterials+1, 
         m_hTextures,  m_nbActiveTextures, 
         m_hRandoms,   m_sceneInfo.width.x*m_sceneInfo.height.x);
		m_texturedTransfered = true;
	}

   // Kernel execution
   Ray ray;
   memset(&ray, 0, sizeof(Ray));
   ray.origin    = m_viewPos;
   ray.direction = m_viewDir;

   int4 objects;
   objects.x = m_nbActiveBoxes+1;
   objects.y = m_nbActivePrimitives+1;
   objects.z = m_nbActiveLamps+1;
   objects.w = 0;
	cudaRender(
      m_blockSize, m_sharedMemSize,
      m_sceneInfo, objects,
      m_postProcessingInfo,
      timer,
		ray, 
      m_angles );
}

void CudaKernel::render_end( char* bitmap)
{
   // GPU -> CPU Data transfers
   d2h_bitmap( bitmap, m_sceneInfo);
}

void CudaKernel::deviceQuery()
{
   std::cout << " CUDA Device Query (Runtime API) version (CUDART static linking)" << std::endl;

   int deviceCount = 0;
   cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

   if (error_id != cudaSuccess)
   {
      std::cout << "cudaGetDeviceCount returned " << (int)error_id << " -> " << cudaGetErrorString(error_id) << std::endl;
   }

   // This function call returns 0 if there are no CUDA capable devices.
   if (deviceCount == 0)
      std::cout << "There is no device supporting CUDA" << std::endl;
   else
      std::cout << "Found " << deviceCount << " CUDA Capable device(s)" << std::endl;

   int dev, driverVersion = 0, runtimeVersion = 0;

   for (dev = 0; dev < deviceCount; ++dev)
   {
      cudaSetDevice(dev);
      cudaDeviceProp deviceProp;
      cudaGetDeviceProperties(&deviceProp, dev);

      std::cout << "Device :" << dev <<", " << deviceProp.name << std::endl;

#if CUDART_VERSION >= 2020
      // Console log
      cudaDriverGetVersion(&driverVersion);
      cudaRuntimeGetVersion(&runtimeVersion);
      std::cout << "  CUDA Driver Version / Runtime Version          " << driverVersion/1000 << 
         "." << (driverVersion%100)/10 << 
         " / " << runtimeVersion/1000 << "." 
         << (runtimeVersion%100)/10 << std::endl;
#endif
      std::cout << "  CUDA Capability Major/Minor version number:    " << deviceProp.major << "." << deviceProp.minor << std::endl;

      std::cout << "  Total amount of global memory: " << 
         (float)deviceProp.totalGlobalMem/1048576.0f << "MBytes (" << 
         (unsigned long long) deviceProp.totalGlobalMem << " bytes)" << std::endl;

#if CUDART_VERSION >= 4000

      std::cout << "  Max Texture Dimension Size (x,y,z)             1D=(" << deviceProp.maxTexture1D << 
         "), 2D=(" << deviceProp.maxTexture2D[0] << "," << deviceProp.maxTexture2D[1] << 
         "), 3D=(" << deviceProp.maxTexture3D[0] << "," << deviceProp.maxTexture3D[1] << "," << deviceProp.maxTexture3D[2] << ")" << std::endl;
      std::cout << "  Max Layered Texture Size (dim) x layers        1D=(" << deviceProp.maxTexture1DLayered[0] <<
         ") x " << deviceProp.maxTexture1DLayered[1] 
         << ", 2D=(" << deviceProp.maxTexture2DLayered[0] << "," << deviceProp.maxTexture2DLayered[1] << 
         ") x " << deviceProp.maxTexture2DLayered[2] << std::endl;
#endif
      std::cout << "  Total amount of constant memory:               " << deviceProp.totalConstMem << "bytes" << std::endl;
      std::cout << "  Total amount of shared memory per block:       " << deviceProp.sharedMemPerBlock << "bytes" << std::endl;
      std::cout << "  Total number of registers available per block: " << deviceProp.regsPerBlock << std::endl;
      std::cout << "  Warp size:                                     " << deviceProp.warpSize << std::endl;
      std::cout << "  Maximum number of threads per multiprocessor:  " << deviceProp.maxThreadsPerMultiProcessor << std::endl;
      std::cout << "  Maximum number of threads per block:           " << deviceProp.maxThreadsPerBlock << std::endl;
      std::cout << "  Maximum sizes of each dimension of a block:    " <<
         deviceProp.maxThreadsDim[0] << " x " <<
         deviceProp.maxThreadsDim[1] << " x " <<
         deviceProp.maxThreadsDim[2] << std::endl;

      m_blockSize.x = 16;
      m_blockSize.y = 16;
      m_blockSize.z = 1;

      std::cout << "  Maximum sizes of each dimension of a grid:     " <<
         deviceProp.maxGridSize[0] << " x " <<
         deviceProp.maxGridSize[1] << " x " <<
         deviceProp.maxGridSize[2] << std::endl;
      std::cout << "  Maximum memory pitch:                          " << deviceProp.memPitch << "bytes" << std::endl;
      std::cout << "  Texture alignment:                             " << deviceProp.textureAlignment  << "bytes" << std::endl;

#if CUDART_VERSION >= 4000
      std::cout << "  Concurrent copy and execution:                 " << (deviceProp.deviceOverlap ? "Yes" : "No") << " with " << deviceProp.asyncEngineCount << "copy engine(s)" << std::endl;
#else
      std::cout << "  Concurrent copy and execution:                 " << (deviceProp.deviceOverlap ? "Yes" : "No") << std::endl;
#endif

#if CUDART_VERSION >= 2020
      std::cout << "  Run time limit on kernels:                     " << (deviceProp.kernelExecTimeoutEnabled ? "Yes" : "No") << std::endl;
      std::cout << "  Integrated GPU sharing Host Memory:            " << (deviceProp.integrated ? "Yes" : "No") << std::endl;
      std::cout << "  Support host page-locked memory mapping:       " << (deviceProp.canMapHostMemory ? "Yes" : "No") << std::endl;
#endif
#if CUDART_VERSION >= 3000
      std::cout << "  Concurrent kernel execution:                   " << (deviceProp.concurrentKernels ? "Yes" : "No") << std::endl;
      std::cout << "  Alignment requirement for Surfaces:            " << (deviceProp.surfaceAlignment ? "Yes" : "No") << std::endl;
#endif
#if CUDART_VERSION >= 3010
      std::cout << "  Device has ECC support enabled:                " << (deviceProp.ECCEnabled ? "Yes" : "No") << std::endl;
#endif
#if CUDART_VERSION >= 3020
      std::cout << "  Device is using TCC driver mode:               " << (deviceProp.tccDriver ? "Yes" : "No") << std::endl;
#endif
#if CUDART_VERSION >= 4000
      std::cout << "  Device supports Unified Addressing (UVA):      " << (deviceProp.unifiedAddressing ? "Yes" : "No") << std::endl;
      std::cout << "  Device PCI Bus ID / PCI location ID:           " << deviceProp.pciBusID << "/" << deviceProp.pciDeviceID << std::endl;
#endif

#if CUDART_VERSION >= 2020
      const char *sComputeMode[] =
      {
         "Default (multiple host threads can use ::cudaSetDevice() with device simultaneously)",
         "Exclusive (only one host thread in one process is able to use ::cudaSetDevice() with this device)",
         "Prohibited (no host thread can use ::cudaSetDevice() with this device)",
         "Exclusive Process (many threads in one process is able to use ::cudaSetDevice() with this device)",
         "Unknown",
         NULL
      };
      std::cout << "  Compute Mode:" << std::endl;
      std::cout << "     < " << sComputeMode[deviceProp.computeMode] << " >" << std::endl;
#endif
   }

   // csv masterlog info
   // *****************************
   // exe and CUDA driver name
   std::cout << std::endl;
   std::string sProfileString = "deviceQuery, CUDA Driver = CUDART";
   char cTemp[10];

   // driver version
   sProfileString += ", CUDA Driver Version = ";
#ifdef WIN32
   sprintf_s(cTemp, 10, "%d.%d", driverVersion/1000, (driverVersion%100)/10);
#else
   sprintf(cTemp, "%d.%d", driverVersion/1000, (driverVersion%100)/10);
#endif
   sProfileString +=  cTemp;

   // Runtime version
   sProfileString += ", CUDA Runtime Version = ";
#ifdef WIN32
   sprintf_s(cTemp, 10, "%d.%d", runtimeVersion/1000, (runtimeVersion%100)/10);
#else
   sprintf(cTemp, "%d.%d", runtimeVersion/1000, (runtimeVersion%100)/10);
#endif
   sProfileString +=  cTemp;

   // Device count
   sProfileString += ", NumDevs = ";
#ifdef WIN32
   sprintf_s(cTemp, 10, "%d", deviceCount);
#else
   sprintf(cTemp, "%d", deviceCount);
#endif
   sProfileString += cTemp;

   // First 2 device names, if any
   for (dev = 0; dev < ((deviceCount > 2) ? 2 : deviceCount); ++dev)
   {
      cudaDeviceProp deviceProp;
      cudaGetDeviceProperties(&deviceProp, dev);
      sProfileString += ", Device = ";
      sProfileString += deviceProp.name;
   }

   sProfileString += "\n";
   std::cout << sProfileString.c_str() << std::endl;
}
