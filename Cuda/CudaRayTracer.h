/* 
 * Copyright (C) 2014 Cyrille Favreau - All Rights Reserved
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 * Written by Cyrille Favreau <cyrille_favreau@hotmail.com>
 */

#pragma once

#include "../Consts.h"
#include "CudaDataTypes.h"

extern "C" void initialize_scene( 
   int2 occupancyParameters,
   SceneInfo sceneInfo,
   const int nbPrimitives,
   const int nbLamps,
   const int nbMaterials,
   const int nbBoxes
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
   int     nbActiveMaterials);

extern "C" void h2d_randoms( 
   int2 occupancyParameters,
	float*     randoms);

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
