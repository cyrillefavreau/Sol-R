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
   int2&           occupancyParameters,
   const SceneInfo sceneInfo,
   const int       nbPrimitives, 
   const int       nbLamps, 
   const int       nbMaterials );

extern "C" void finalize_scene(
   const int2   occupancyParameters );

extern "C" void h2d_scene(
   const int2   occupancyParameters,
   BoundingBox* boundingBoxes, 
   const int    nbActiveBoxes,
	Primitive*   primitives, 
   const int    nbPrimitives,
	Lamp*        lamps, 
   const int    nbLamps );

extern "C" void h2d_materials( 
   const int2    occupancyParameters,
	Material*     materials, 
   const int     nbActiveMaterials,
   RandomBuffer* randoms,   
   const int     nbRandoms );

extern "C" void h2d_textures( 
   const int2          occupancyParameters,
	const int           activeTextures, 
   TextureInformation* textureInfos );

extern "C" void h2d_lightInformation( 
   const int2        occupancyParameters,
	LightInformation* lightInformation, 
   int               lightInformationSize);

extern "C" void d2h_bitmap( 
   const int2           occupancyParameters,
   const SceneInfo      sceneInfo,
   BitmapBuffer*        bitmap, 
   PrimitiveXYIdBuffer* primitivesXYIds);

extern "C" void cudaRender(
   const int2               occupancyParameters,
   const int4               blockSize,
   const SceneInfo          sceneInfo,
   const int4               objects,
   const PostProcessingInfo PostProcessingInfo,
   const Vertex             origin, 
   const Vertex             direction, 
   const Vertex             angles);

#ifdef USE_KINECT
extern "C" void h2d_kinect( 
   const int2    occupancyParameters,
   BitmapBuffer* video, 
   BitmapBuffer* depth );
#endif // USE_KINECT
