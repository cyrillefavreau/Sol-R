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
 * aint with this program.  If not, see <http://www.gnu.org/licenses/>. 
 */

/*
 * Author: Cyrille Favreau <cyrille_favreau@hotmail.com>
 *
 */

#include "RayTracingEngineStub.h"
#include "Consts.h"

#include "PDBReader.h"
#include "OBJReader.h"
#include "FileMarshaller.h"

#include <fstream>

#if USE_CUDA
#include "Cuda/CudaKernel.h"
CudaKernel* kernel = NULL;
#endif // USE_OPENCL

#if USE_OPENCL
#include "OpenCL/OpenCLKernel.h"
typedef OpenCLKernel kernel;
kernel* kernel = NULL;
#endif // USE_OPENCL

SceneInfo gSceneInfo;
PostProcessingInfo gPostProcessingInfo;

// --------------------------------------------------------------------------------
// Implementation
// --------------------------------------------------------------------------------
extern "C" RAYTRACINGENGINE_API int RayTracer_SetSceneInfo(
   int width, int height,
   bool shadowsEnabled, int nbRayIterations, double transparentColor,
   double viewDistance, double shadowIntensity,
   int supportFor3DVision, double width3DVision,
   double bgColorR, double bgColorG, double bgColorB,
   bool renderBoxes, int pathTracingIteration, int maxPathTracingIterations,
   int outputType, int timer, int fogEffect, int isometric3D)
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
   gSceneInfo.supportFor3DVision.x      = supportFor3DVision;
   gSceneInfo.renderBoxes.x             = static_cast<int>(renderBoxes);
   gSceneInfo.pathTracingIteration.x    = pathTracingIteration;
   gSceneInfo.maxPathTracingIterations.x= maxPathTracingIterations;
   gSceneInfo.misc.x                    = outputType;
   gSceneInfo.misc.y                    = timer;
   gSceneInfo.misc.z                    = fogEffect;
   gSceneInfo.misc.w                    = isometric3D;
   return 0;
}

extern "C" RAYTRACINGENGINE_API int RayTracer_SetPostProcessingInfo(
   int type, double param1, double param2, int param3 )
{
   gPostProcessingInfo.type.x   = type;
   gPostProcessingInfo.param1.x = static_cast<float>(param1);
   gPostProcessingInfo.param2.x = static_cast<float>(param2);
   gPostProcessingInfo.param3.x = param3;
   return 0;
}

extern "C" RAYTRACINGENGINE_API int RayTracer_InitializeKernel( 
   bool activeLogging, 
   int platform, 
   int device )
{
	kernel = new CudaKernel( activeLogging, platform, device );
   if( kernel == NULL ) return -1;
	kernel->setSceneInfo( gSceneInfo );
   kernel->initBuffers();
   return 0;
}

// --------------------------------------------------------------------------------
extern "C" RAYTRACINGENGINE_API 
   int RayTracer_FinalizeKernel()
{
   if( kernel ) delete kernel;
   kernel = 0;
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
   kernel->setCamera( eye, dir, angles );
}

// --------------------------------------------------------------------------------
extern "C" RAYTRACINGENGINE_API int RayTracer_RunKernel( double timer, char* image )
{
	kernel->setSceneInfo( gSceneInfo );
   kernel->setPostProcessingInfo( gPostProcessingInfo );
	kernel->render_begin( static_cast<float>(timer) );
   kernel->render_end( image );
   return 0;
}

// --------------------------------------------------------------------------------
extern "C" RAYTRACINGENGINE_API 
   int RayTracer_AddPrimitive( int type )
{
   return kernel->addPrimitive( static_cast<PrimitiveType>(type) );
}

// --------------------------------------------------------------------------------
extern "C" RAYTRACINGENGINE_API 
   int RayTracer_SetPrimitive( 
   int    index,
   double p0_x, double p0_y, double p0_z, 
   double p1_x, double p1_y, double p1_z, 
   double p2_x, double p2_y, double p2_z, 
   double size_x, double size_y, double size_z,
   int    materialId, 
   double materialPaddingX, 
   double materialPaddingY )
{
   kernel->setPrimitive(
      index,
      static_cast<float>(p0_x), static_cast<float>(p0_y), static_cast<float>(p0_z), 
      static_cast<float>(p1_x), static_cast<float>(p1_y), static_cast<float>(p1_z), 
      static_cast<float>(p2_x), static_cast<float>(p2_y), static_cast<float>(p2_z), 
      static_cast<float>(size_x), static_cast<float>(size_y), static_cast<float>(size_z), 
      materialId, 
      static_cast<float>(materialPaddingX), 
      static_cast<float>(materialPaddingY));
   return 0;
}

extern "C" RAYTRACINGENGINE_API int RayTracer_GetPrimitive( 
   int index,
   double& p0_x, double& p0_y, double& p0_z, 
   double& p1_x, double& p1_y, double& p1_z, 
   double& p2_x, double& p2_y, double& p2_z, 
   double& size_x, double& size_y, double& size_z,
   int& materialId, double& materialPaddingX, double& materialPaddingY )
{
   CPUPrimitive* primitive = kernel->getPrimitive(index);
   if( primitive != NULL )
   {
      p0_x = primitive->p0.x;
      p0_y = primitive->p0.y;
      p0_z = primitive->p0.z;
      p1_x = primitive->p1.x;
      p1_y = primitive->p1.y;
      p1_z = primitive->p1.z;
      p2_x = primitive->p2.x;
      p2_y = primitive->p2.y;
      p2_z = primitive->p2.z;
      size_x = primitive->size.x;
      size_y = primitive->size.y;
      size_z = primitive->size.z;
      materialId = primitive->materialId;
      materialPaddingX = primitive->materialInfo.x;
      materialPaddingY = primitive->materialInfo.y;
      return 0;
   }
   return -1;
}

extern "C" RAYTRACINGENGINE_API int RayTracer_GetPrimitiveAt( int x, int y )
{
   return kernel->getPrimitiveAt(x,y);
}

extern "C" RAYTRACINGENGINE_API int RayTracer_GetPrimitiveCenter( int index, double& x, double& y, double& z)
{
   float cx,cy,cz, cw;
   kernel->getPrimitiveCenter( index, cx, cy, cz, cw );
   x = static_cast<double>(cx);
   y = static_cast<double>(cy);
   z = static_cast<double>(cz);
   return 0;
}

extern "C" RAYTRACINGENGINE_API int RayTracer_RotatePrimitive( 
   int index,
   double rx, double ry, double rz,
   double ax, double ay, double az)
{
   float4 rotationCenter = { static_cast<float>(rx), static_cast<float>(ry),  static_cast<float>(rz), 0.f };
   float4 angles = { static_cast<float>(ax), static_cast<float>(ay),  static_cast<float>(az), 0.f };

   //kernel->rotatePrimitive( index, boxId, rotationCenter, angles ); // TODO!!
   return 0;
}

extern "C" RAYTRACINGENGINE_API int RayTracer_RotatePrimitives(
   int fromBoxId, int toBoxId,
   double rx, double ry, double rz,
   double ax, double ay, double az)
{
   try
   {
      float4 rotationCenter = { static_cast<float>(rx), static_cast<float>(ry),  static_cast<float>(rz), 0.f };
      float4 angles = { static_cast<float>(ax), static_cast<float>(ay),  static_cast<float>(az), 0.f };

      kernel->rotatePrimitives( rotationCenter, angles, fromBoxId, toBoxId );
   }
   catch( ... )
   {
   }
   return 0;
}

extern "C" RAYTRACINGENGINE_API int RayTracer_SetPrimitiveMaterial( 
   int    index,
   int    materialId)
{
   kernel->setPrimitiveMaterial( index,  materialId );
   return 0;
}

extern "C" RAYTRACINGENGINE_API int RayTracer_GetPrimitiveMaterial( int index)
{
   return kernel->getPrimitiveMaterial( index );
}

// --------------------------------------------------------------------------------
extern "C" RAYTRACINGENGINE_API int RayTracer_UpdateSkeletons( 
   int index,
   double p0_x, double  p0_y, double p0_z, 
   double size,
   double radius,       int materialId,
   double head_radius,  int head_materialId,
   double hands_radius, int hands_materialId,
   double feet_radius,  int feet_materialId)
{
#if USE_KINECT
   float4 position = { static_cast<float>(p0_x), static_cast<float>(p0_y), static_cast<float>(p0_z), 0.f };
   return kernel->updateSkeletons(
      index,
      position,
      static_cast<float>(size),
      static_cast<float>(radius),
      materialId,
      static_cast<float>(head_radius),  head_materialId,
      static_cast<float>(hands_radius), hands_materialId,
      static_cast<float>(feet_radius),  feet_materialId);
#else
   return 0;
#endif // USE_KINECT
}

// --------------------------------------------------------------------------------
extern "C" RAYTRACINGENGINE_API int RayTracer_AddTexture( char* filename )
{
   return kernel->addTexture( filename );
}

// --------------------------------------------------------------------------------
extern "C" RAYTRACINGENGINE_API int RayTracer_SetTexture( int index, HANDLE texture )
{
   kernel->setTexture( 
      index, 
      static_cast<char*>(texture) );
   return 0;
}

// ---------- Materials ----------
extern "C" RAYTRACINGENGINE_API int RayTracer_AddMaterial()
{
   return kernel->addMaterial();
}

// --------------------------------------------------------------------------------
extern "C" RAYTRACINGENGINE_API int RayTracer_SetMaterial(
   int    index,
   double color_r, double color_g, double color_b, 
   double noise,
   double reflection,
   double refraction,
   bool   procedural,
   bool   wireframe, int wireframeDepth,
   double transparency,
   int    textureId,
   double specValue, double specPower, double specCoef, double innerIllumination,
   bool   fastTransparency)
{
   kernel->setMaterial(
      index, 
      static_cast<float>(color_r), 
      static_cast<float>(color_g), 
      static_cast<float>(color_b), 
      static_cast<float>(noise),
      static_cast<float>(reflection), 
      static_cast<float>(refraction),
      procedural, 
      wireframe, 
      static_cast<int>(wireframeDepth),
      static_cast<float>(transparency), 
      static_cast<int>(textureId),
      static_cast<float>(specValue),
      static_cast<float>(specPower), 
      static_cast<float>(specCoef ),
      static_cast<float>(innerIllumination),
      fastTransparency
      );
   return 0;
}

// --------------------------------------------------------------------------------
extern "C" RAYTRACINGENGINE_API int RayTracer_GetMaterial(
   int     in_index,
   double& out_color_r, 
   double& out_color_g, 
   double& out_color_b,
   double& out_noise,
   double& out_reflection,
   double& out_refraction,
   int&    out_procedural,
   int&    out_wireframe,
   int&    out_wireframeDepth,
   double& out_transparency,
   int&    out_textureId,
   double& out_specValue, 
   double& out_specPower, 
   double& out_specCoef,
   double& out_innerIllumination)
{
   float color_r,color_g,color_b, noise, reflection, refraction, transparency;
   float specValue, specPower, specCoef, innerIllumination;
   bool  procedural;
   bool  wireframe;
   int   wireframeDepth;
   int   textureId;
   int returnValue = kernel->getMaterialAttributes(
      in_index, 
      color_r, color_g, color_b,
      noise, reflection, refraction, procedural, wireframe, wireframeDepth, transparency, 
      textureId, specValue, specPower, specCoef, innerIllumination );

   out_color_r = static_cast<double>(color_r);
   out_color_g = static_cast<double>(color_g);
   out_color_b = static_cast<double>(color_b);
   out_noise = static_cast<double>(noise);
   out_reflection = static_cast<double>(reflection);
   out_refraction = static_cast<double>(refraction);
   out_transparency = static_cast<double>(transparency);
   out_textureId = static_cast<int>(textureId);
   out_procedural = procedural ? 1 : 0;
   out_wireframe = wireframe ? 1 : 0;
   out_wireframeDepth = wireframeDepth;
   out_specValue = static_cast<double>(specValue);
   out_specPower = static_cast<double>(specPower);
   out_specCoef = static_cast<double>(specCoef);
   out_innerIllumination = static_cast<double>(innerIllumination);
   return returnValue;
}

// --------------------------------------------------------------------------------
extern "C" RAYTRACINGENGINE_API int RayTracer_LoadMolecule( 
   char*  filename,
   int    geometryType,
   double defaultAtomSize,
   double defaultStickSize,
   int    atomMaterialType,
   double scale)
{
   // PDB
	PDBReader prbReader;
	float4 minPos = prbReader.loadAtomsFromFile(
      filename, *kernel,
      static_cast<GeometryType>(geometryType),
      static_cast<float>(defaultAtomSize), 
      static_cast<float>(defaultStickSize),
      atomMaterialType, 
      static_cast<float>(scale) );
   kernel->compactBoxes(true);
   return kernel->getNbActivePrimitives();
}

// --------------------------------------------------------------------------------
extern "C" RAYTRACINGENGINE_API int RayTracer_LoadOBJModel( 
   char*  filename,
   int    materialId,
   double scale)
{
   float4 center={0.f,0.f,0.f,0.f};
   // PDB
	OBJReader objReader;
	float4 minPos = objReader.loadModelFromFile( 
      filename,
      *kernel,
      center,
      static_cast<float>(scale), 
      materialId );
   return kernel->getNbActivePrimitives();
}

// --------------------------------------------------------------------------------
extern "C" RAYTRACINGENGINE_API int RayTracer_SaveToFile( char* filename)
{
   FileMarshaller fm;
   fm.saveToFile( *kernel, filename );
   return kernel->getNbActivePrimitives();
}

// --------------------------------------------------------------------------------
extern "C" RAYTRACINGENGINE_API int RayTracer_LoadFromFile( char* filename)
{
   FileMarshaller fm;
   fm.loadFromFile( *kernel, filename );
   return kernel->getNbActivePrimitives();
}

// --------------------------------------------------------------------------------
extern "C" RAYTRACINGENGINE_API int RayTracer_CompactBoxes( bool update )
{
   return kernel->compactBoxes(update);
}

// --------------------------------------------------------------------------------
extern "C" RAYTRACINGENGINE_API int RayTracer_GetLight( int index )
{
   return kernel->getLight(index);
}

// --------------------------------------------------------------------------------
extern "C" RAYTRACINGENGINE_API 
   int RayTracer_SetLight( 
   int    index,
   double p0_x, double p0_y, double p0_z, 
   double p1_x, double p1_y, double p1_z, 
   double p2_x, double p2_y, double p2_z, 
   double size_x, double size_y, double size_z,
   int    materialId, 
   double materialPaddingX, 
   double materialPaddingY )
{
   kernel->setLight(
      index,
      static_cast<float>(p0_x), static_cast<float>(p0_y), static_cast<float>(p0_z), 
      static_cast<float>(p1_x), static_cast<float>(p1_y), static_cast<float>(p1_z), 
      static_cast<float>(p2_x), static_cast<float>(p2_y), static_cast<float>(p2_z), 
      static_cast<float>(size_x), static_cast<float>(size_y), static_cast<float>(size_z), 
      materialId, 
      static_cast<float>(materialPaddingX), 
      static_cast<float>(materialPaddingY));
   return 0;
}
