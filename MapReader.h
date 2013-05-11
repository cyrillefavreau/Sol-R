/* 
* Raytracing Engine
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

#include <string>

#include "GPUKernel.h"

//<COLOURSCHEME 0>
//  <COLOUR 0.696078 0.666667 0.372549>
//  <COLOUR 0.176471 0.058824 0.431372>
//  <COLOUR 0.000000 0.529412 0.725490>
//  <COLOUR 0.549020 0.000000 0.500000>

struct MapMaterialScheme
{
   unsigned int index;
   float3 material[4];
};

// <BLOCK LOCATION 8 0 0 DIMENSION 53 36 0 NORTH 0 SOUTH 0 EAST 0 WEST 0 SPLIT 0 HOLLOW 0>
struct MapBlock
{
   int zone;
   int type;
   float3 location;
   float3 dimension;
   int north;
   int south;
   int east;
   int west;
   int split;
   int hollow;
};

// <ZONE LOCATION 0 0 0 DIMENSIONS 54 37 23 COLOURSCHEME 0>
struct MapZone
{
   float3 location;
   float3 dimension;
   int    colorScene;
};

// <JEWEL LOCATION 1 6 0>
struct MapJewel
{
   float3 location;
};

class RAYTRACINGENGINE_API MapReader
{
public:
   MapReader(void);
   ~MapReader(void);

   float3 loadFromFile( 
      const std::string& filename,
      GPUKernel& cudaKernel);
};
