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

#ifdef USE_CUDA
#include "Cuda/CudaKernel.h"
typedef CudaKernel GenericGPUKernel;
#else
   #ifdef USE_OPENCL
      #include "OpenCL/OpenCLKernel.h"
      typedef OpenCLKernel GenericGPUKernel;
   #else
      #include "CPUKernel.h"
      typedef CPUKernel GenericGPUKernel;
   #endif // USE_OPENCL
#endif // USE_CUDA
GenericGPUKernel* gKernel = NULL;

SceneInfo gSceneInfoStub;
PostProcessingInfo gPostProcessingInfoStub;

// --------------------------------------------------------------------------------
// Implementation
// --------------------------------------------------------------------------------
extern "C" RAYTRACINGENGINE_API int RayTracer_SetSceneInfo(
   int width, int height,
   int graphicsLevel, int nbRayIterations, double transparentColor,
   double viewDistance, double shadowIntensity,
   int renderingType, double width3DVision,
   double bgColorR, double bgColorG, double bgColorB,
   bool renderBoxes, int pathTracingIteration, int maxPathTracingIterations,
   int outputType, int timer, int fogEffect, int isometric3D)
{
   gSceneInfoStub.width.x                   = width;
   gSceneInfoStub.height.x                  = height;
   gSceneInfoStub.graphicsLevel.x           = graphicsLevel;
   gSceneInfoStub.nbRayIterations.x         = nbRayIterations;
   gSceneInfoStub.transparentColor.x        = static_cast<float>(transparentColor);
   gSceneInfoStub.viewDistance.x            = static_cast<float>(viewDistance);
   gSceneInfoStub.shadowIntensity.x         = static_cast<float>(shadowIntensity);
   gSceneInfoStub.width3DVision.x           = static_cast<float>(width3DVision);
   gSceneInfoStub.backgroundColor.x         = static_cast<float>(bgColorR);
   gSceneInfoStub.backgroundColor.y         = static_cast<float>(bgColorG);
   gSceneInfoStub.backgroundColor.z         = static_cast<float>(bgColorB);
   gSceneInfoStub.backgroundColor.w         = 0.f;
   gSceneInfoStub.renderingType.x           = renderingType;
   gSceneInfoStub.renderBoxes.x             = static_cast<int>(renderBoxes);
   gSceneInfoStub.pathTracingIteration.x    = pathTracingIteration;
   gSceneInfoStub.maxPathTracingIterations.x= maxPathTracingIterations;
   gSceneInfoStub.misc.x                    = outputType;
   gSceneInfoStub.misc.y                    = timer;
   gSceneInfoStub.misc.z                    = fogEffect;
   gSceneInfoStub.misc.w                    = isometric3D;
   return 0;
}

extern "C" RAYTRACINGENGINE_API int RayTracer_SetPostProcessingInfo(
   int type, double param1, double param2, int param3 )
{
   gPostProcessingInfoStub.type.x   = type;
   gPostProcessingInfoStub.param1.x = static_cast<float>(param1);
   gPostProcessingInfoStub.param2.x = static_cast<float>(param2);
   gPostProcessingInfoStub.param3.x = param3;
   return 0;
}

extern "C" RAYTRACINGENGINE_API int RayTracer_InitializeKernel( 
   bool activeLogging, 
   int platform, 
   int device )
{
   if( gKernel == nullptr )
   {
	   gKernel = new GenericGPUKernel( activeLogging, NB_MAX_BOXES, platform, device );
      gSceneInfoStub.pathTracingIteration.x = 0; 
	   gKernel->setSceneInfo( gSceneInfoStub );
      gKernel->initBuffers();
      gKernel->setFrame(0);
      gKernel->setOptimalNbOfBoxes(16384);
      return 0;
   }
   return -1;
}

// --------------------------------------------------------------------------------
extern "C" RAYTRACINGENGINE_API 
   int RayTracer_FinalizeKernel()
{
   if( gKernel ) 
   {
      delete gKernel;
      gKernel = nullptr;
   }
   return 0;   
}

// --------------------------------------------------------------------------------
extern "C" RAYTRACINGENGINE_API 
   int RayTracer_ResetKernel()
{
   if( gKernel ) 
   {
      gKernel->resetAll();
      return 0;   
   }
   return -1;
}


// --------------------------------------------------------------------------------
extern "C" RAYTRACINGENGINE_API 
   void RayTracer_SetCamera( 
   double eye_x,   double eye_y,   double eye_z,
   double dir_x,   double dir_y,   double dir_z,
   double angle_x, double angle_y, double angle_z )
{
   float3 eye     = { static_cast<float>(eye_x),   static_cast<float>(eye_y),   static_cast<float>(eye_z)};
   float3 dir     = { static_cast<float>(dir_x),   static_cast<float>(dir_y),   static_cast<float>(dir_z)};
   float3 angles  = { static_cast<float>(angle_x), static_cast<float>(angle_y), static_cast<float>(angle_z)};
   gKernel->setCamera( eye, dir, angles );
}

// --------------------------------------------------------------------------------
extern "C" RAYTRACINGENGINE_API int RayTracer_RunKernel( double timer, BitmapBuffer* image )
{
	gKernel->setSceneInfo( gSceneInfoStub );
   gKernel->setPostProcessingInfo( gPostProcessingInfoStub );
	gKernel->render_begin( static_cast<float>(timer) );
   gKernel->render_end();
   memcpy(image,gKernel->getBitmap(),gSceneInfoStub.width.x*gSceneInfoStub.height.x*3);
   return 0;
}

// --------------------------------------------------------------------------------
extern "C" RAYTRACINGENGINE_API 
   int RayTracer_AddPrimitive( int type )
{
   return gKernel->addPrimitive( static_cast<PrimitiveType>(type) );
}

// --------------------------------------------------------------------------------
extern "C" RAYTRACINGENGINE_API 
   int RayTracer_SetPrimitive( 
   int    index,
   double p0_x, double p0_y, double p0_z, 
   double p1_x, double p1_y, double p1_z, 
   double p2_x, double p2_y, double p2_z, 
   double size_x, double size_y, double size_z,
   int    materialId )
{
   gKernel->setPrimitive(
      index,
      static_cast<float>(p0_x), static_cast<float>(p0_y), static_cast<float>(p0_z), 
      static_cast<float>(p1_x), static_cast<float>(p1_y), static_cast<float>(p1_z), 
      static_cast<float>(p2_x), static_cast<float>(p2_y), static_cast<float>(p2_z), 
      static_cast<float>(size_x), static_cast<float>(size_y), static_cast<float>(size_z), 
      materialId);
   return 0;
}

extern "C" RAYTRACINGENGINE_API int RayTracer_GetPrimitive( 
   int index,
   double& p0_x, double& p0_y, double& p0_z, 
   double& p1_x, double& p1_y, double& p1_z, 
   double& p2_x, double& p2_y, double& p2_z, 
   double& size_x, double& size_y, double& size_z,
   int& materialId )
{
   CPUPrimitive* primitive = gKernel->getPrimitive(index);
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
      return 0;
   }
   return -1;
}

extern "C" RAYTRACINGENGINE_API int RayTracer_GetPrimitiveAt( int x, int y )
{
   return gKernel->getPrimitiveAt(x,y);
}

extern "C" RAYTRACINGENGINE_API int RayTracer_GetPrimitiveCenter( int index, double& x, double& y, double& z)
{
   float3 center = gKernel->getPrimitiveCenter( index );
   x = static_cast<double>(center.x);
   y = static_cast<double>(center.y);
   z = static_cast<double>(center.z);
   return 0;
}

extern "C" RAYTRACINGENGINE_API int RayTracer_RotatePrimitive( 
   int index,
   double rx, double ry, double rz,
   double ax, double ay, double az)
{
   float4 rotationCenter = { static_cast<float>(rx), static_cast<float>(ry),  static_cast<float>(rz), 0.f };
   float4 angles = { static_cast<float>(ax), static_cast<float>(ay),  static_cast<float>(az), 0.f };

   //gKernel->rotatePrimitive( index, boxId, rotationCenter, angles ); // TODO!!
   return 0;
}

extern "C" RAYTRACINGENGINE_API int RayTracer_RotatePrimitives(
   int fromBoxId, int toBoxId,
   double rx, double ry, double rz,
   double ax, double ay, double az)
{
   try
   {
      float3 rotationCenter = { static_cast<float>(rx), static_cast<float>(ry),  static_cast<float>(rz) };
      float3 angles = { static_cast<float>(ax), static_cast<float>(ay),  static_cast<float>(az) };

      gKernel->rotatePrimitives( rotationCenter, angles, fromBoxId, toBoxId );
      gKernel->compactBoxes(false);
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
   gKernel->setPrimitiveMaterial( index,  materialId );
   return 0;
}

extern "C" RAYTRACINGENGINE_API int RayTracer_GetPrimitiveMaterial( int index)
{
   return gKernel->getPrimitiveMaterial( index );
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
   float3 position = { static_cast<float>(p0_x), static_cast<float>(p0_y), static_cast<float>(p0_z) };
   return gKernel->updateSkeletons(
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
extern "C" RAYTRACINGENGINE_API int RayTracer_LoadTextureFromFile( int index, char* filename )
{
   return gKernel->loadTextureFromFile( index, filename );
}

// --------------------------------------------------------------------------------
extern "C" RAYTRACINGENGINE_API int RayTracer_SetTexture( int index, HANDLE texture )
{
   /* TODO
   CPUTextureInformation texInfo;
   gKernel->setTexture( 
      index, 
      static_cast<char*>(texture) );
   */
   return 0;
}

extern "C" RAYTRACINGENGINE_API int RayTracer_GetNbTextures( int& nbTextures )
{
   return NB_MAX_TEXTURES;// gKernel->getNbActiveTextures();
}

// ---------- Materials ----------
extern "C" RAYTRACINGENGINE_API int RayTracer_AddMaterial()
{
   return gKernel->addMaterial();
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
   double specValue, double specPower, double specCoef, 
   double innerIllumination, double illuminationDiffusion, double illuminationPropagation, 
   bool   fastTransparency)
{
   gKernel->setMaterial(
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
      static_cast<float>(illuminationDiffusion),
      static_cast<float>(illuminationPropagation),
      fastTransparency
      );
   return 0;
}

// --------------------------------------------------------------------------------
extern "C" RAYTRACINGENGINE_API int RayTracer_GetMaterial(
   int     in_index,
   double& out_color_r, double& out_color_g, double& out_color_b,
   double& out_noise,
   double& out_reflection,
   double& out_refraction,
   int&    out_procedural,
   int&    out_wireframe, int&    out_wireframeDepth,
   double& out_transparency,
   int&    out_textureId,
   double& out_specValue, double& out_specPower, double& out_specCoef,
   double& out_innerIllumination, double& out_illuminationDiffusion, double& out_illuminationPropagation,
   int&    out_fastTransparency)
{
   float color_r,color_g,color_b, noise, reflection, refraction, transparency;
   float specValue, specPower, specCoef, innerIllumination, illuminationDiffusion, illuminationPropagation;
   bool  procedural;
   bool  wireframe;
   int   wireframeDepth;
   int   textureId;
   bool  fastTransparency;
   int returnValue = gKernel->getMaterialAttributes(
      in_index, 
      color_r, color_g, color_b,
      noise, reflection, refraction, procedural, wireframe, wireframeDepth, transparency, 
      textureId, specValue, specPower, specCoef, innerIllumination, illuminationDiffusion, illuminationPropagation, fastTransparency );

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
   out_illuminationDiffusion = static_cast<double>(illuminationDiffusion);
   out_illuminationPropagation = static_cast<double>(illuminationPropagation);
   out_fastTransparency = fastTransparency ? 1 : 0;
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
   float s(static_cast<float>(scale));
   float3 objectScale = {s,s,s};
	float3 minPos = prbReader.loadAtomsFromFile(
      filename, *gKernel,
      static_cast<GeometryType>(geometryType),
      static_cast<float>(defaultAtomSize), 
      static_cast<float>(defaultStickSize),
      atomMaterialType, 
      objectScale );
   gKernel->compactBoxes(true);
   return gKernel->getNbActivePrimitives();
}

// --------------------------------------------------------------------------------
extern "C" RAYTRACINGENGINE_API int RayTracer_LoadOBJModel( 
   char*  filename,
   int    materialId,
   double scale)
{
   float3 center={0.f,0.f,0.f};
   // PDB
	OBJReader objReader;
   float s(static_cast<float>(scale));
   float3 objectScale = {s,s,s};
	float3 minPos = objReader.loadModelFromFile(
      filename,
      *gKernel,
      center,
      true,
      objectScale,
      true,
      materialId,
      false);
   return gKernel->getNbActivePrimitives();
}

// --------------------------------------------------------------------------------
extern "C" RAYTRACINGENGINE_API int RayTracer_SaveToFile( char* filename)
{
   FileMarshaller fm;
   fm.saveToFile( *gKernel, filename );
   return gKernel->getNbActivePrimitives();
}

// --------------------------------------------------------------------------------
extern "C" RAYTRACINGENGINE_API int RayTracer_LoadFromFile( char* filename, double scale )
{
   float3 center = { 0.f,0.f,0.f };
   FileMarshaller fm;
   fm.loadFromFile( *gKernel, filename, center, static_cast<float>(scale) );
   return gKernel->getNbActivePrimitives();
}

// --------------------------------------------------------------------------------
extern "C" RAYTRACINGENGINE_API int RayTracer_CompactBoxes( bool update )
{
   return gKernel->compactBoxes(update);
}

// --------------------------------------------------------------------------------
extern "C" RAYTRACINGENGINE_API int RayTracer_GetLight( int index )
{
   return gKernel->getLight(index);
}

