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

#pragma once

#include "CudaDataTypes.h"

extern "C" void initialize_scene( 
	int width, int height, int nbPrimitives, int nbints, int nbMaterials );

extern "C" void finalize_scene();

extern "C" void h2d_scene(
   BoundingBox* boundingBoxes, int nbActiveBoxes,
	Primitive* primitives, int nbPrimitives,
	int* ints, int nbints );

extern "C" void h2d_materials( 
	Material*  materials, int nbActiveMaterials,
   float*     randoms,   int nbRandoms );

extern "C" void h2d_textures( 
	const int activeTextures, TextureInformation* textureInfos );

extern "C" void h2d_lightInformation( 
	LightInformation* lightInformation, int lightInformationSize);

extern "C" void d2h_bitmap( unsigned char* bitmap, int4* primitivesXYIds, const SceneInfo sceneInfo );

extern "C" void cudaRender(
   int4 blockSize,
   SceneInfo sceneInfo,
   int4 objects,
   PostProcessingInfo PostProcessingInfo,
   float3 origin, 
   float3 direction, 
   float3 angles);

#ifdef USE_KINECT
extern "C" void h2d_kinect( char* video, char* depth );
#endif // USE_KINECT
