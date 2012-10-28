/* 
 * OpenCL Raytracer
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

#include "RayTracingEngineStub.h"

#include <fstream>

#if USE_CUDA
#include "Cuda/CudaKernel.h"
typedef CudaKernel GPUKERNEL;
GPUKERNEL* gpuKernel = nullptr;
#endif // USE_OPENCL

#if USE_OPENCL
#include "OpenCL/OpenCLKernel.h"
typedef OpenCLKernel GPUKERNEL;
GPUKERNEL* gpuKernel = nullptr;
#endif // USE_OPENCL

SceneInfo gSceneInfo;
PostProcessingInfo gPostProcessingInfo;

// --------------------------------------------------------------------------------
// Implementation
// --------------------------------------------------------------------------------
extern "C" RAYTRACINGENGINE_API long RayTracer_SetSceneInfo(
   int width, int height,
   bool shadowsEnabled, int nbRayIterations, double transparentColor,
   double viewDistance, double shadowIntensity,
   bool supportFor3DVision, double width3DVision,
   double bgColorR, double bgColorG, double bgColorB,
   bool renderBoxes, int pathTracingIteration, int maxPathTracingIterations )
{
   gSceneInfo.width.x                   = width;
   gSceneInfo.height.x                  = height;
   gSceneInfo.shadowsEnabled.x          = shadowsEnabled;
   gSceneInfo.nbRayIterations.x         = nbRayIterations;
   gSceneInfo.transparentColor.x        = static_cast<float>(transparentColor);
   gSceneInfo.viewDistance.x            = static_cast<float>(viewDistance);
   gSceneInfo.shadowIntensity.x         = static_cast<float>(shadowIntensity);
   gSceneInfo.width3DVision.x           = static_cast<float>(width3DVision);
   gSceneInfo.backgroundColor.x         = static_cast<float>(bgColorR);
   gSceneInfo.backgroundColor.y         = static_cast<float>(bgColorG);
   gSceneInfo.backgroundColor.z         = static_cast<float>(bgColorB);
   gSceneInfo.backgroundColor.w         = 0.f;
   gSceneInfo.supportFor3DVision.x      = static_cast<int>(supportFor3DVision);
   gSceneInfo.renderBoxes.x             = static_cast<int>(renderBoxes);
   gSceneInfo.pathTracingIteration.x    = pathTracingIteration;
   gSceneInfo.maxPathTracingIterations.x= maxPathTracingIterations;
   return 0;
}

extern "C" RAYTRACINGENGINE_API long RayTracer_SetPostProcessingInfo(
   int type, double param1, double param2, int param3 )
{
   gPostProcessingInfo.type.x   = type;
   gPostProcessingInfo.param1.x = static_cast<float>(param1);
   gPostProcessingInfo.param2.x = static_cast<float>(param2);
   gPostProcessingInfo.param3.x = param3;
   return 0;
}

extern "C" RAYTRACINGENGINE_API long RayTracer_InitializeKernel( int platform, int device )
{
	gpuKernel = new GPUKERNEL( platform, device );
   if( gpuKernel == nullptr ) return -1;
	gpuKernel->setSceneInfo( gSceneInfo );
   gpuKernel->initBuffers();
   return 0;
}

// --------------------------------------------------------------------------------
extern "C" RAYTRACINGENGINE_API 
   long RayTracer_FinalizeKernel()
{
   if( gpuKernel ) delete gpuKernel;
   return 0;   
}

// --------------------------------------------------------------------------------
extern "C" RAYTRACINGENGINE_API 
   void RayTracer_SetCamera( 
   double eye_x,   double eye_y,   double eye_z,
   double dir_x,   double dir_y,   double dir_z,
   double angle_x, double angle_y, double angle_z )
{
   float4 eye     = { static_cast<float>(eye_x),   static_cast<float>(eye_y),   static_cast<float>(eye_z),   0.f };
   float4 dir     = { static_cast<float>(dir_x),   static_cast<float>(dir_y),   static_cast<float>(dir_z),   0.f };
   float4 angles  = { static_cast<float>(angle_x), static_cast<float>(angle_y), static_cast<float>(angle_z), 0.f };
   gpuKernel->setCamera( eye, dir, angles );
}

// --------------------------------------------------------------------------------
extern "C" RAYTRACINGENGINE_API long RayTracer_RunKernel( double timer, char* image )
{
	gpuKernel->render_begin( static_cast<float>(timer) );
   gpuKernel->render_end( image );
   return 0;
}

// --------------------------------------------------------------------------------
extern "C" RAYTRACINGENGINE_API 
   long RayTracer_AddPrimitive( int type )
{
   return gpuKernel->addPrimitive( static_cast<PrimitiveType>(type) );
}

// --------------------------------------------------------------------------------
extern "C" RAYTRACINGENGINE_API 
   long RayTracer_SetPrimitive( 
   int    index,
   int    boxId,
   double x, double y, double z, 
   double width,
   double height,
   double depth,
   int    materialId, 
   int    materialPaddingX, 
   int    materialPaddingY )
{
   gpuKernel->setPrimitive(
      index, boxId,
      static_cast<float>(x), static_cast<float>(y), static_cast<float>(z), 
      static_cast<float>(width), static_cast<float>(height), static_cast<float>(depth), 
      materialId, materialPaddingX, materialPaddingY );
   return 0;
}

extern "C" RAYTRACINGENGINE_API long RayTracer_RotatePrimitive( 
   int index, int boxId,
   double x, double y, double z)
{
   float4 angles = { static_cast<float>(x), static_cast<float>(y),  static_cast<float>(z), 0.f };

   gpuKernel->rotatePrimitive( index, boxId, angles );
   return 0;
}

extern "C" RAYTRACINGENGINE_API long RayTracer_SetPrimitiveMaterial( 
   int    index,
   int    materialId)
{
   gpuKernel->setPrimitiveMaterial( index,  materialId );
   return 0;
}

// --------------------------------------------------------------------------------
extern "C" RAYTRACINGENGINE_API long RayTracer_UpdateSkeletons( 
   double center_x, double  center_y, double center_z, 
   double size,
   double radius,       int materialId,
   double head_radius,  int head_materialId,
   double hands_radius, int hands_materialId,
   double feet_radius,  int feet_materialId)
{
#if USE_KINECT
   return gpuKernel->updateSkeletons(
      center_x, center_y, center_z, 
      size,
      radius,       materialId,
      head_radius,  head_materialId,
      hands_radius, hands_materialId,
      feet_radius,  feet_materialId);
#else
   return 0;
#endif // USE_KINECT
}

// --------------------------------------------------------------------------------
extern "C" RAYTRACINGENGINE_API long RayTracer_AddTexture( char* filename )
{
   return gpuKernel->addTexture( filename );
}

// --------------------------------------------------------------------------------
extern "C" RAYTRACINGENGINE_API long RayTracer_SetTexture( int index, HANDLE texture )
{
   gpuKernel->setTexture( 
      index, 
      static_cast<char*>(texture) );
   return 0;
}

// ---------- Materials ----------
extern "C" RAYTRACINGENGINE_API long RayTracer_AddMaterial()
{
   return gpuKernel->addMaterial();
}

extern "C" RAYTRACINGENGINE_API long RayTracer_SetMaterial(
   int    index,
   double color_r, 
   double color_g, 
   double color_b,
   double noise,
   double reflection,
   double refraction,
   int    textured,
   double transparency,
   int    textureId,
   double specValue, 
   double specPower, 
   double specCoef,
   double innerIllumination)
{
   gpuKernel->setMaterial(
      index, 
      static_cast<float>(color_r), 
      static_cast<float>(color_g), 
      static_cast<float>(color_b), 
      static_cast<float>(noise),
      static_cast<float>(reflection), 
      static_cast<float>(refraction),
      static_cast<int>(textured), 
      static_cast<float>(transparency), 
      textureId,
      static_cast<float>(specValue),
      static_cast<float>(specPower), 
      static_cast<float>(innerIllumination),
      static_cast<float>(specCoef )
      );
   return 0;
}
