/* Copyright (c) 2011-2014, Cyrille Favreau
 * All rights reserved. Do not distribute without permission.
 * Responsible Author: Cyrille Favreau <cyrille_favreau@hotmail.com>
 *
 * This file is part of Sol-R <https://github.com/cyrillefavreau/Sol-R>
 *
 * This library is free software; you can redistribute it and/or modify it under
 * the terms of the GNU Lesser General Public License version 3.0 as published
 * by the Free Software Foundation.
 *
 * This library is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this library; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */

#pragma once

#include "../Consts.h"
#include "CudaDataTypes.h"

extern "C" void initialize_scene(int2 occupancyParameters, SceneInfo sceneInfo,
                                 int nbPrimitives, int nbLamps, int nbMaterials
#ifdef USE_MANAGED_MEMORY
                                 ,
                                 BoundingBox *&boundingBoxes,
                                 Primitive *&primitives
#endif
                                 );

extern "C" void finalize_scene(int2 occupancyParameters
#ifdef USE_MANAGED_MEMORY
                               ,
                               BoundingBox *boundingBoxes, Primitive *primitives
#endif
                               );

extern "C" void reshape_scene(int2 occupancyParameters, SceneInfo sceneInfo);

extern "C" void h2d_scene(int2 occupancyParameters, BoundingBox *boundingBoxes,
                          int nbActiveBoxes, Primitive *primitives,
                          int nbPrimitives, Lamp *lamps, int nbLamps);

extern "C" void h2d_materials(int2 occupancyParameters, Material *materials,
                              int nbActiveMaterials);

extern "C" void h2d_randoms(int2 occupancyParameters, float *randoms);

extern "C" void h2d_textures(int2 occupancyParameters, int activeTextures,
                             TextureInformation *textureInfos);

extern "C" void h2d_lightInformation(int2 occupancyParameters,
                                     LightInformation *lightInformation,
                                     int lightInformationSize);

extern "C" void d2h_bitmap(int2 occupancyParameters, SceneInfo sceneInfo,
                           BitmapBuffer *bitmap,
                           PrimitiveXYIdBuffer *primitivesXYIds);

extern "C" void cudaRender(int2 occupancyParameters, int4 blockSize,
                           SceneInfo sceneInfo, int4 objects,
                           PostProcessingInfo PostProcessingInfo, Vertex origin,
                           Vertex direction, Vertex angles
#ifdef USE_MANAGED_MEMORY
                           ,
                           BoundingBox *boundingBoxes, Primitive *primitives
#endif
                           );

#ifdef USE_KINECT
extern "C" void h2d_kinect(int2 occupancyParameters, BitmapBuffer *video,
                           BitmapBuffer *depth);
#endif // USE_KINECT
