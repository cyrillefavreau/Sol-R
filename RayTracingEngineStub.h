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

#pragma once

#ifdef WIN32
#include <windows.h>
#else
typedef char* HANDLE;
#endif 

#include "DLL_API.h"
#include "GPUKernel.h"

// ---------- Scene ----------
extern "C" RAYTRACINGENGINE_API int RayTracer_SetSceneInfo(
   int width, int height,
   bool shadowsEnabled, int nbRayIterations, double transparentColor,
   double viewDistance, double shadowIntensity,
   int supportFor3DVision, double width3DVision,
   double bgColorR, double bgColorG, double bgColorB,
   bool renderBoxes, int pathTracingIteration, int maxPathTracingIterations,
   int outputType, int timer, int fogEffect, int isometric3D );

extern "C" RAYTRACINGENGINE_API int RayTracer_SetPostProcessingInfo(
   int type, double param1, double param2, int param3 );

extern "C" RAYTRACINGENGINE_API int RayTracer_InitializeKernel( bool activeLogging, int platform, int device, char* filename );
extern "C" RAYTRACINGENGINE_API int RayTracer_FinalizeKernel();

// ---------- Camera ----------
extern "C" RAYTRACINGENGINE_API void RayTracer_SetCamera( 
   double eye_x,   double eye_y,   double eye_z,
   double dir_x,   double dir_y,   double dir_z,
   double angle_x, double angle_y, double angle_z);

// ---------- Rendering ----------
extern "C" RAYTRACINGENGINE_API int RayTracer_RunKernel( double timer, char* image );

// ---------- Primitives ----------
extern "C" RAYTRACINGENGINE_API int RayTracer_AddPrimitive( int type );

extern "C" RAYTRACINGENGINE_API int RayTracer_SetPrimitive( 
   int index,
   double p0_x, double p0_y, double p0_z, 
   double p1_x, double p1_y, double p1_z, 
   double p2_x, double p2_y, double p2_z, 
   double size_x, double size_y, double size_z,
   int materialId, double materialPaddingX, double materialPaddingY );

extern "C" RAYTRACINGENGINE_API int RayTracer_GetPrimitive( 
   int index,
   double& p0_x, double& p0_y, double& p0_z, 
   double& p1_x, double& p1_y, double& p1_z, 
   double& p2_x, double& p2_y, double& p2_z, 
   double& size_x, double& size_y, double& size_z,
   int& materialId, double& materialPaddingX, double& materialPaddingY );

extern "C" RAYTRACINGENGINE_API int RayTracer_GetPrimitiveAt( int x, int y );

extern "C" RAYTRACINGENGINE_API int RayTracer_GetPrimitiveCenter( int index, double& x, double& y, double& z);

extern "C" RAYTRACINGENGINE_API int RayTracer_RotatePrimitive( 
   int index,
   double rx, double ry, double rz,
   double ax, double ay, double az);
extern "C" RAYTRACINGENGINE_API int RayTracer_RotatePrimitives( 
   int fromBoxId, int toBoxId,
   double rx, double ry, double rz,
   double ax, double ay, double az);


extern "C" RAYTRACINGENGINE_API int RayTracer_SetPrimitiveMaterial( 
   int    index,
   int    materialId);
extern "C" RAYTRACINGENGINE_API int RayTracer_GetPrimitiveMaterial( 
   int    index);

// ---------- Materials ----------
extern "C" RAYTRACINGENGINE_API int RayTracer_AddMaterial();
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
   double innerIllumination,
   bool   fastTransparency);

extern "C" RAYTRACINGENGINE_API int RayTracer_GetMaterial(
   int     index,
   double& color_r, double& color_g, double& color_b,
   double& noise,
   double& reflection,
   double& refraction,
   int&    procedural,
   int&    wireframe, int& wireframeDepth,
   double& transparency,
   int&    textureId,
   double& specValue, double& specPower, double& specCoef,
   double& innerIllumination);

// Boxes
extern "C" RAYTRACINGENGINE_API int RayTracer_CompactBoxes( bool update );

// ---------- Lights ----------
extern "C" RAYTRACINGENGINE_API int RayTracer_GetLight( int index );

extern "C" RAYTRACINGENGINE_API 
   int RayTracer_SetLight( 
   int    index,
   double p0_x, double p0_y, double p0_z, 
   double p1_x, double p1_y, double p1_z, 
   double p2_x, double p2_y, double p2_z, 
   double size_x, double size_y, double size_z,
   int    materialId, 
   double materialPaddingX, 
   double materialPaddingY );

// ---------- Textures ----------
extern "C" RAYTRACINGENGINE_API int RayTracer_AddTexture( char* filename );
extern "C" RAYTRACINGENGINE_API int RayTracer_SetTexture( int index, HANDLE texture );

// ---------- Molecules ----------
extern "C" RAYTRACINGENGINE_API int RayTracer_LoadMolecule( 
   char*  filename,
   int    geometryType,
   double defaultAtomSize,
   double defaultStickSize,
   int    atomMaterialType,
   double scale);

// ---------- OBJ models ----------
extern "C" RAYTRACINGENGINE_API int RayTracer_LoadOBJModel( 
   char*  filename,
   int    materialId,
   double scale);

#ifdef USE_KINECT
// ---------- Kinect ----------
extern "C" RAYTRACINGENGINE_API int RayTracer_UpdateSkeletons(
   int index,
   double center_x, double  center_y, double  center_z, 
   double size,
   double radius,       int materialId,
   double head_radius,  int head_materialId,
   double hands_radius, int hands_materialId,
   double feet_radius,  int feet_materialId);
#endif // USE_KINECT
