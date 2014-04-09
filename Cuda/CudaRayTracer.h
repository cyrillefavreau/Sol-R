/* 
* Copyright (C) 2011-2014 Cyrille Favreau <cyrille_favreau@hotmail.com>
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

#include "../Consts.h"
#include "CudaDataTypes.h"

extern "C" void initialize_scene( 
   int2            occupancyParameters,
   SceneInfo sceneInfo,
   int       nbPrimitives,
   int       nbLamps,
   int       nbMaterials
#ifdef USE_MANAGED_MEMORY
   ,BoundingBox*&   boundingBoxes
   ,Primitive*&     primitives
#endif
   );

extern "C" void finalize_scene(
   int2   occupancyParameters
#ifdef USE_MANAGED_MEMORY
   ,BoundingBox* boundingBoxes
   ,Primitive*   primitives
#endif
   );

extern "C" void reshape_scene(
   int2 occupancyParameters,
   SceneInfo sceneInfo );
   
extern "C" void h2d_scene(
   int2   occupancyParameters,
   BoundingBox* boundingBoxes, 
   int    nbActiveBoxes,
	Primitive*   primitives, 
   int    nbPrimitives,
	Lamp*        lamps, 
   int    nbLamps );

extern "C" void h2d_materials( 
   int2    occupancyParameters,
	Material*     materials, 
   int     nbActiveMaterials,
   RandomBuffer* randoms,   
   int     nbRandoms );

extern "C" void h2d_textures( 
   int2          occupancyParameters,
	int           activeTextures,
   TextureInformation* textureInfos );

extern "C" void h2d_lightInformation( 
   int2        occupancyParameters,
	LightInformation* lightInformation, 
   int               lightInformationSize);

extern "C" void d2h_bitmap( 
   int2           occupancyParameters,
   SceneInfo      sceneInfo,
   BitmapBuffer*        bitmap, 
   PrimitiveXYIdBuffer* primitivesXYIds);

extern "C" void cudaRender(
   int2               occupancyParameters,
   int4               blockSize,
   SceneInfo          sceneInfo,
   int4               objects,
   PostProcessingInfo PostProcessingInfo,
   Vertex             origin,
   Vertex             direction,
   Vertex             angles
#ifdef USE_MANAGED_MEMORY
   ,BoundingBox*            boundingBoxes
	,Primitive*              primitives
#endif
   );

#ifdef USE_KINECT
extern "C" void h2d_kinect( 
   int2    occupancyParameters,
   BitmapBuffer* video, 
   BitmapBuffer* depth );
#endif // USE_KINECT
