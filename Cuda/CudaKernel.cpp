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

#include <GL/freeglut.h>

#include <math.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include <time.h>
#include <sstream>
#include <stdlib.h>
#include <vector>

#include <cuda_runtime.h>
#include "CudaRayTracer.h"
#include "CudaKernel.h"
#include "../Consts.h"
#include "../Logging.h"

const long MAX_SOURCE_SIZE = 65535;
const long MAX_DEVICES = 10;

#ifndef WIN32
typedef struct stBITMAPFILEHEADER
{
  short bfType;
  int bfSize;
  short Reserved1;
  short Reserved2;
  int bfOffBits;
} BITMAPFILEHEADER;

typedef struct stBITMAPINFOHEADER
{
  int biSizeImage;
  long biWidth;
  long biHeight;
} BITMAPINFOHEADER;
#endif

/*
________________________________________________________________________________

CudaKernel::CudaKernel
________________________________________________________________________________
*/
CudaKernel::CudaKernel( bool activeLogging, int optimalNbOfPrimmitivesPerBox, int platform, int device ) 
 : GPUKernel( activeLogging, optimalNbOfPrimmitivesPerBox, platform, device ),
   m_sharedMemSize(256),
   m_imageCount(0)
{
   LOG_INFO(3,"CudaKernel::CudaKernel(" << platform << "," << device << ")");
   m_blockSize.x = 0;
   m_blockSize.y = 0;
   m_blockSize.z = 0;
   m_blockSize.w = 0;

   m_gpuDescription = "CUDA device";

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
   std::cout << "----------------------------" << std::endl;
   std::cout << "                         O  " << std::endl;
   std::cout << "                       --+--" << std::endl;
   std::cout << "                         |  " << std::endl;
   std::cout << "Kinect initialization   / \\" << std::endl;
	HRESULT err=NuiInitialize(NUI_INITIALIZE_FLAG_USES_DEPTH_AND_PLAYER_INDEX | NUI_INITIALIZE_FLAG_USES_SKELETON | NUI_INITIALIZE_FLAG_USES_COLOR);
   m_kinectEnabled = (err==S_OK);

   if( m_kinectEnabled )
   {
	   m_hNextDepthFrameEvent = CreateEvent( NULL, TRUE, FALSE, NULL ); 
	   m_hNextVideoFrameEvent = CreateEvent( NULL, TRUE, FALSE, NULL ); 
	   m_hNextSkeletonEvent   = CreateEvent( NULL, TRUE, FALSE, NULL );

	   m_skeletons = CreateEvent( NULL, TRUE, FALSE, NULL );			 
	   NuiSkeletonTrackingEnable( m_skeletons, NUI_SKELETON_TRACKING_FLAG_ENABLE_SEATED_SUPPORT );

	   NuiImageStreamOpen( NUI_IMAGE_TYPE_COLOR,                  NUI_IMAGE_RESOLUTION_640x480, 0, 2, m_hNextVideoFrameEvent, &m_pVideoStreamHandle );
	   NuiImageStreamOpen( NUI_IMAGE_TYPE_DEPTH_AND_PLAYER_INDEX, NUI_IMAGE_RESOLUTION_320x240, 0, 2, m_hNextDepthFrameEvent, &m_pDepthStreamHandle );

	   NuiCameraElevationSetAngle( 0 );
      std::cout << "    SUCCESS" << std::endl;
   }
   else
   {
      std::cout << "*** FAILED  ***" << std::endl;
   }
   std::cout << "----------------------------" << std::endl;
#endif // USE_KINECT
}

/*
________________________________________________________________________________

CudaKernel::~CudaKernel
________________________________________________________________________________
*/
CudaKernel::~CudaKernel()
{
   LOG_INFO(3,"CudaKernel::~CudaKernel");
   // Clean up
   releaseDevice();

#if USE_KINECT
   if( m_kinectEnabled ) 
   {
      CloseHandle(m_skeletons);
      CloseHandle(m_hNextDepthFrameEvent); 
      CloseHandle(m_hNextVideoFrameEvent); 
      CloseHandle(m_hNextSkeletonEvent);
      NuiShutdown();
   }
#endif // USE_KINECT
}

void CudaKernel::initBuffers() 
{
   GPUKernel::initBuffers();
   deviceQuery();
	initializeDevice();
}

void CudaKernel::cleanup()
{
   GPUKernel::cleanup();
   releaseDevice();
}


/*
________________________________________________________________________________

Initialize CPU & GPU resources
________________________________________________________________________________
*/
void CudaKernel::initializeDevice()
{
   LOG_INFO(1,"CudaKernel::initializeDevice");
	initialize_scene( m_sceneInfo.width.x, m_sceneInfo.height.x, NB_MAX_PRIMITIVES, NB_MAX_LAMPS, NB_MAX_MATERIALS, NB_MAX_TEXTURES );
}

void CudaKernel::resetBoxesAndPrimitives()
{
   LOG_INFO(3,"CudaKernel::resetBoxesAndPrimitives");
}

/*
________________________________________________________________________________

Release CPU & GPU resources
________________________________________________________________________________
*/
void CudaKernel::releaseDevice()
{
	finalize_scene();
}

/*
________________________________________________________________________________

Execute GPU GPUKernel
________________________________________________________________________________
*/
void CudaKernel::render_begin( const float timer )
{
#if USE_KINECT
   if( m_kinectEnabled )
   {
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
      h2d_kinect( m_hVideo, m_hDepth );
   }
#endif // USE_KINECT

   if( m_refresh )
   {
	   // CPU -> GPU Data transfers
      int nbBoxes      = m_nbActiveBoxes;
      int nbPrimitives = m_nbActivePrimitives;
      int nbMaterials  = m_nbActiveMaterials+1;
      int nbLamps      = m_nbActiveLamps;

      LOG_INFO(3,"Data sizes: " << nbBoxes << ", " << nbPrimitives << ", " << nbMaterials << ", " << nbLamps);

      if( !m_primitivesTransfered )
      {
	      h2d_scene( 
            m_hBoundingBoxes, nbBoxes, 
            m_hPrimitives, nbPrimitives, 
            m_hLamps, nbLamps );

         h2d_lightInformation( 
            m_lightInformation, m_lightInformationSize );
         m_primitivesTransfered = true;
      }
	
      if( !m_materialsTransfered )
	   {
		   h2d_materials( 
            m_hMaterials, nbMaterials, 
            m_hRandoms,   m_sceneInfo.width.x*m_sceneInfo.height.x);
         LOG_INFO(3, "Transfering " << nbMaterials << " materials");
		   m_materialsTransfered = true;
	   }

      if( !m_texturesTransfered )
	   {
         LOG_INFO(3, "Transfering " << m_nbActiveTextures << " textures, and " << m_lightInformationSize << " light information");
		   h2d_textures( m_hTextures,  m_nbActiveTextures);
		   m_texturesTransfered = true;
      }

      // Kernel execution
      int4 objects;
      objects.x = nbBoxes;
      objects.y = nbPrimitives;
      objects.z = nbLamps;
      objects.w = m_lightInformationSize;

	   cudaRender(
         m_blockSize,
         m_sceneInfo, objects,
         m_postProcessingInfo,
         m_viewPos,
		   m_viewDir, 
         m_angles );
   }
   m_refresh = (m_sceneInfo.pathTracingIteration.x<m_sceneInfo.maxPathTracingIterations.x);
}

void CudaKernel::render_end()
{
   // GPU -> CPU Data transfers
   d2h_bitmap( m_bitmap, m_hPrimitivesXYIds, m_sceneInfo);

   if( m_sceneInfo.misc.x == 0 )
   {
      ::glEnable(GL_TEXTURE_2D);
      ::glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
      ::glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
      ::glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
      ::glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_DECAL);
      ::glTexImage2D(GL_TEXTURE_2D, 0, 3, m_sceneInfo.width.x, m_sceneInfo.height.x, 0, GL_RGB, GL_UNSIGNED_BYTE, m_bitmap);
      ::glBegin(GL_QUADS);
      ::glTexCoord2f(1.f, 1.f);
      ::glVertex3f(-1.f, 1.f, 0.f);
      ::glTexCoord2f(0.f, 1.f);
      ::glVertex3f( 1.f, 1.f, 0.f);
      ::glTexCoord2f(0.f, 0.f);
      ::glVertex3f( 1.f,-1.f, 0.f);
      ::glTexCoord2f(1.f, 0.f);
      ::glVertex3f(-1.f,-1.f, 0.f);
      ::glEnd();
      ::glDisable(GL_TEXTURE_2D);
   }
}

void CudaKernel::deviceQuery()
{
   LOG_INFO(3,"CUDA Device Query (Runtime API) version (CUDART static linking)");

   int deviceCount = 0;
   cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

   if (error_id != cudaSuccess)
   {
      LOG_INFO(3,"cudaGetDeviceCount returned " << (int)error_id << " -> " << cudaGetErrorString(error_id));
   }

   // This function call returns 0 if there are no CUDA capable devices.
   if (deviceCount == 0)
   {
      LOG_INFO(3,"There is no device supporting CUDA");
   }
   else
   {
      LOG_INFO(3,"Found " << deviceCount << " CUDA Capable device(s)");
   }

   int dev, driverVersion = 0, runtimeVersion = 0;

   for (dev = 0; dev < deviceCount; ++dev)
   {
      cudaSetDevice(dev);
      cudaDeviceProp deviceProp;
      cudaGetDeviceProperties(&deviceProp, dev);

      LOG_INFO(1,"--------------------------------------------------------------------------------" );
      LOG_INFO(1,"Device :" << dev <<", " << deviceProp.name );

      m_gpuDescription = deviceProp.name;

      cudaDriverGetVersion(&driverVersion);
      cudaRuntimeGetVersion(&runtimeVersion);
      LOG_INFO(1,"  CUDA Driver Version / Runtime Version          " << driverVersion/1000 << "." << (driverVersion%100)/10 << " / " << runtimeVersion/1000 << "." << (runtimeVersion%100)/10);
      LOG_INFO(1,"  CUDA Capability Major/Minor version number:    " << deviceProp.major << "." << deviceProp.minor);
      LOG_INFO(1,"  Total amount of global memory: " << (float)deviceProp.totalGlobalMem/1048576.0f << "MBytes (" << (unsigned long long) deviceProp.totalGlobalMem << " bytes)");
      LOG_INFO(1,"  Max Texture Dimension Size (x,y,z)             1D=(" << deviceProp.maxTexture1D << "), 2D=(" << deviceProp.maxTexture2D[0] << "," << deviceProp.maxTexture2D[1] << "), 3D=(" << deviceProp.maxTexture3D[0] << "," << deviceProp.maxTexture3D[1] << "," << deviceProp.maxTexture3D[2] << ")");
      LOG_INFO(1,"  Max Layered Texture Size (dim) x layers        1D=(" << deviceProp.maxTexture1DLayered[0] << ") x " << deviceProp.maxTexture1DLayered[1] << ", 2D=(" << deviceProp.maxTexture2DLayered[0] << "," << deviceProp.maxTexture2DLayered[1] << ") x " << deviceProp.maxTexture2DLayered[2]);
      LOG_INFO(1,"  Total amount of constant memory:               " << deviceProp.totalConstMem << "bytes");
      LOG_INFO(1,"  Total amount of shared memory per block:       " << deviceProp.sharedMemPerBlock << "bytes");
      LOG_INFO(1,"  Total number of registers available per block: " << deviceProp.regsPerBlock);
      LOG_INFO(1,"  Warp size:                                     " << deviceProp.warpSize);
      LOG_INFO(1,"  Maximum number of threads per multiprocessor:  " << deviceProp.maxThreadsPerMultiProcessor);
      LOG_INFO(1,"  Maximum number of threads per block:           " << deviceProp.maxThreadsPerBlock);
      LOG_INFO(1,"  Maximum sizes of each dimension of a block:    " << deviceProp.maxThreadsDim[0] << " x " << deviceProp.maxThreadsDim[1] << " x " << deviceProp.maxThreadsDim[2]);
      LOG_INFO(1,"  Maximum sizes of each dimension of a grid:     " << deviceProp.maxGridSize[0] << " x " << deviceProp.maxGridSize[1] << " x " << deviceProp.maxGridSize[2]);
      LOG_INFO(1,"  Maximum memory pitch:                          " << deviceProp.memPitch << "bytes");
      LOG_INFO(1,"  Texture alignment:                             " << deviceProp.textureAlignment  << "bytes");
      LOG_INFO(1,"  Concurrent copy and execution:                 " << (deviceProp.deviceOverlap ? "Yes" : "No") << " with " << deviceProp.asyncEngineCount << "copy engine(s)");
      LOG_INFO(1,"  Run time limit on kernels:                     " << (deviceProp.kernelExecTimeoutEnabled ? "Yes" : "No"));
      LOG_INFO(1,"  Integrated GPU sharing Host Memory:            " << (deviceProp.integrated ? "Yes" : "No"));
      LOG_INFO(1,"  Support host page-locked memory mapping:       " << (deviceProp.canMapHostMemory ? "Yes" : "No"));
      LOG_INFO(1,"  Concurrent GPUKernel execution:                   " << (deviceProp.concurrentKernels ? "Yes" : "No"));
      LOG_INFO(1,"  Alignment requirement for Surfaces:            " << (deviceProp.surfaceAlignment ? "Yes" : "No"));
      LOG_INFO(1,"  Device has ECC support enabled:                " << (deviceProp.ECCEnabled ? "Yes" : "No"));
      LOG_INFO(1,"  Device is using TCC driver mode:               " << (deviceProp.tccDriver ? "Yes" : "No"));
      LOG_INFO(1,"  Device supports Unified Addressing (UVA):      " << (deviceProp.unifiedAddressing ? "Yes" : "No"));
      LOG_INFO(1,"  Device PCI Bus ID / PCI location ID:           " << deviceProp.pciBusID << "/" << deviceProp.pciDeviceID);

      const char *sComputeMode[] =
      {
         "Default (multiple host threads can use ::cudaSetDevice() with device simultaneously)",
         "Exclusive (only one host thread in one process is able to use ::cudaSetDevice() with this device)",
         "Prohibited (no host thread can use ::cudaSetDevice() with this device)",
         "Exclusive Process (many threads in one process is able to use ::cudaSetDevice() with this device)",
         "Unknown",
         NULL
      };
      LOG_INFO(1,"  Compute Mode:");
      LOG_INFO(1,"     < " << sComputeMode[deviceProp.computeMode] << " >");
      LOG_INFO(1,"--------------------------------------------------------------------------------" );

      m_blockSize.x = 8;
      m_blockSize.y = 8;
      m_blockSize.z = 1;

   }

   // exe and CUDA driver name
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
   LOG_INFO(3, sProfileString.c_str());
}
