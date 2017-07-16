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

#pragma once

#ifdef WIN32
#include <windows.h>
#endif 

#include "DLL_API.h"
#include "GPUKernel.h"

// ---------- Scene ----------
extern "C" RAYTRACINGENGINE_API long RayTracer_SetSceneInfo(
   int width, int height,
   bool shadowsEnabled, int nbRayIterations, double transparentColor,
   double viewDistance, double shadowIntensity,
   bool supportFor3DVision, double width3DVision,
   double bgColorR, double bgColorG, double bgColorB,
   bool renderBoxes, int pathTracingIteration, int maxPathTracingIterations );

extern "C" RAYTRACINGENGINE_API long RayTracer_SetPostProcessingInfo(
   int type, double param1, double param2, int param3 );

extern "C" RAYTRACINGENGINE_API long RayTracer_InitializeKernel( bool activeLogging, int platform, int device );
extern "C" RAYTRACINGENGINE_API long RayTracer_FinalizeKernel();

// ---------- Camera ----------
extern "C" RAYTRACINGENGINE_API void RayTracer_SetCamera( 
   double eye_x,   double eye_y,   double eye_z,
   double dir_x,   double dir_y,   double dir_z,
   double angle_x, double angle_y, double angle_z);

// ---------- Rendering ----------
extern "C" RAYTRACINGENGINE_API long RayTracer_RunKernel( double timer, char* image );

// ---------- Primitives ----------
extern "C" RAYTRACINGENGINE_API long RayTracer_AddPrimitive( int type );
extern "C" RAYTRACINGENGINE_API long RayTracer_SetPrimitive( 
   int index, int boxId,
   double x, double y, double z, 
   double width, double height, double depth,
   int materialId, int materialPaddingX, int materialPaddingY );

extern "C" RAYTRACINGENGINE_API long RayTracer_RotatePrimitive( 
   int index, int boxId, 
   double x, double y, double z);

extern "C" RAYTRACINGENGINE_API long RayTracer_SetPrimitiveMaterial( 
   int    index,
   int    materialId);

// ---------- Materials ----------
extern "C" RAYTRACINGENGINE_API long RayTracer_AddMaterial();
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
   double innerIllumination);

// ---------- Textures ----------
extern "C" RAYTRACINGENGINE_API long RayTracer_AddTexture( char* filename );
extern "C" RAYTRACINGENGINE_API long RayTracer_SetTexture( int index, HANDLE texture );

#ifdef USE_KINECT
// ---------- Kinect ----------
extern "C" RAYTRACINGENGINE_API long RayTracer_UpdateSkeletons(
   int index, int boxId,
   double center_x, double  center_y, double  center_z, 
   double size,
   double radius,       int materialId,
   double head_radius,  int head_materialId,
   double hands_radius, int hands_materialId,
   double feet_radius,  int feet_materialId);
#endif // USE_KINECT
